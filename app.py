import streamlit as st
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
import numpy as np
import plotly.graph_objects as go

# Cấu hình trang Streamlit (Full width để xem 3D tốt hơn)
st.set_page_config(layout="wide", page_title="DEM 3D Visualizer")

# -----------------------------------------------------------------------------
# 1. CÁC HÀM XỬ LÝ (PROCESSING FUNCTIONS)
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_and_downsample_dem(file_content, max_dim=500):
    """
    Đọc file DEM từ upload, downsample nếu kích thước lớn hơn max_dim.
    Trả về: data (numpy array), transform, crs, nodata_val
    """
    with MemoryFile(file_content) as memfile:
        with memfile.open() as dataset:
            # Lấy thông tin metadata gốc
            profile = dataset.profile
            orig_width = dataset.width
            orig_height = dataset.height
            
            # Tính toán scale factor để giới hạn kích thước (tối ưu hiệu năng)
            scale = 1
            if orig_width > max_dim or orig_height > max_dim:
                scale = max_dim / max(orig_width, orig_height)
            
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            # Đọc dữ liệu với resampling (downsample ngay lúc đọc)
            data = dataset.read(
                1,
                out_shape=(new_height, new_width),
                resampling=Resampling.bilinear
            )
            
            # Cập nhật transform cho kích thước mới
            transform = dataset.transform * dataset.transform.scale(
                (dataset.width / data.shape[1]),
                (dataset.height / data.shape[0])
            )
            
            # Xử lý nodata: Chuyển thành np.nan để Plotly không vẽ điểm đó
            if dataset.nodata is not None:
                data = data.astype('float32')
                data[data == dataset.nodata] = np.nan
            
            return data, transform, dataset.crs, dataset.nodata

def reproject_to_metric(data, transform, src_crs):
    """
    Chuyển đổi DEM từ Geographic CRS (Degree) sang Projected CRS (Metric).
    Sử dụng Web Mercator (EPSG:3857) làm đích để đảm bảo đơn vị là mét.
    """
    dst_crs = 'EPSG:3857' # Web Mercator (đơn vị mét)

    # Tính toán transform và kích thước mới sau khi reproject
    new_transform, new_width, new_height = calculate_default_transform(
        src_crs, dst_crs, data.shape[1], data.shape[0], 
        left=transform[2], bottom=transform[5] + transform[4]*data.shape[0],
        right=transform[2] + transform[0]*data.shape[1], top=transform[5]
    )

    destination = np.zeros((new_height, new_width), dtype=np.float32)

    reproject(
        source=data,
        destination=destination,
        src_transform=transform,
        src_crs=src_crs,
        dst_transform=new_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear
    )

    # Đặt lại nodata thành nan cho mảng mới
    destination[destination == 0] = np.nan 
    
    return destination, new_transform, dst_crs

def prepare_xyz(data, transform):
    """
    Tạo lưới tọa độ X, Y từ transform và trả về X, Y, Z.
    Z chính là giá trị elevation từ data.
    """
    rows, cols = data.shape
    
    # Tạo vector chỉ số hàng và cột
    x_cols = np.arange(cols)
    y_rows = np.arange(rows)
    
    # Chuyển đổi chỉ số pixel sang tọa độ không gian thực
    # Affine transform: x_geo = a * col + b * row + c ...
    # Rasterio transform: (a, b, c, d, e, f)
    # Thông thường với file north-up: x = c + a*col, y = f + e*row
    
    # Tạo lưới coordinate 2D
    x_coords, y_coords = rasterio.transform.xy(transform, y_rows, x_cols, offset='center')
    
    # rasterio.transform.xy trả về tuple danh sách tọa độ 1D nếu input là 1D
    # Cần tạo meshgrid để map với Z (data)
    # Lưu ý: xy trả về (xs, ys), ta cần meshgrid
    
    # Cách nhanh hơn: Dùng linspace dựa trên bounds
    # minx = transform[2]
    # maxy = transform[5]
    # pixel_width = transform[0]
    # pixel_height = transform[4] # thường là âm
    
    # Tạo X, Y meshgrid
    # X tăng dần theo cột, Y giảm dần theo hàng (thường là vậy với GeoTIFF)
    xs = np.linspace(transform[2], transform[2] + transform[0] * cols, cols)
    ys = np.linspace(transform[5], transform[5] + transform[4] * rows, rows)
    
    X, Y = np.meshgrid(xs, ys)
    Z = data
    
    return X, Y, Z

def plot_3d_surface(X, Y, Z, colormap='Viridis', z_scale=1.0, show_grid=True, z_min=None, z_max=None):
    """
    Vẽ biểu đồ 3D sử dụng Plotly.
    """
    
    # Áp dụng Z scaling (Vertical Exaggeration)
    # Lưu ý: Ta không nhân trực tiếp vào dữ liệu Z để hiển thị tooltip đúng giá trị thực,
    # nhưng Plotly Surface không có tham số scale riêng biệt dễ dàng ngoài việc chỉnh aspect ratio.
    # Để đơn giản và trực quan: Ta nhân Z hiển thị, nhưng ghi chú giá trị thực.
    # Ở đây: Nhân Z để tạo hình khối.
    
    Z_plot = Z * z_scale
    
    # Cắt lọc giá trị Z theo min/max slider
    if z_min is not None:
        Z_plot[Z_plot < z_min * z_scale] = z_min * z_scale
    if z_max is not None:
        Z_plot[Z_plot > z_max * z_scale] = z_max * z_scale

    fig = go.Figure(data=[go.Surface(
        z=Z_plot,
        x=X,
        y=Y,
        colorscale=colormap,
        cmin=np.nanmin(Z_plot),
        cmax=np.nanmax(Z_plot),
        colorbar=dict(title='Elevation (m)')
    )])

    # Cấu hình hiển thị
    fig.update_layout(
        title='Interactive 3D Terrain',
        autosize=True,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(title='X (Easting)', showgrid=show_grid, visible=show_grid),
            yaxis=dict(title='Y (Northing)', showgrid=show_grid, visible=show_grid),
            zaxis=dict(title='Z (Elevation)', showgrid=show_grid, visible=show_grid),
            aspectmode='data' # Quan trọng: Giữ tỷ lệ 1:1:1
        )
    )
    return fig

# -----------------------------------------------------------------------------
# 2. GIAO DIỆN CHÍNH (STREAMLIT UI)
# -----------------------------------------------------------------------------

st.title("🏔️ GIS 3D DEM Visualizer")
st.markdown("Upload GeoTIFF, xử lý nodata, và hiển thị bề mặt 3D tương tác.")

# --- Sidebar: Input & Settings ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload DEM (GeoTIFF)", type=['tif', 'tiff'])

if uploaded_file is not None:
    # --- Sidebar: Performance Settings ---
    st.sidebar.header("2. Performance")
    max_pixels = st.sidebar.slider(
        "Max Resolution (px)", 
        min_value=100, 
        max_value=2000, 
        value=500,
        step=100,
        help="Giảm độ phân giải để render 3D mượt mà hơn."
    )

    # --- Load Data ---
    with st.spinner("Đang đọc và xử lý file Raster..."):
        # Đọc file (có downsample)
        data, transform, crs, nodata = load_and_downsample_dem(uploaded_file, max_dim=max_pixels)
        
    st.success(f"Đã load file thành công! Kích thước lưới hiển thị: {data.shape}")
    
    # Hiển thị thông tin metadata
    with st.expander("Thông tin Metadata Raster"):
        st.write(f"**CRS:** {crs}")
        st.write(f"**Dimensions:** {data.shape}")
        st.write(f"**Min/Max Z:** {np.nanmin(data):.2f} / {np.nanmax(data):.2f}")
        st.write(f"**Nodata Value:** {nodata}")

    # --- Check CRS (Geographic vs Projected) ---
    is_geographic = False
    if crs:
        is_geographic = crs.is_geographic

    if is_geographic:
        st.warning(f"⚠️ Cảnh báo: File đang sử dụng hệ tọa độ địa lý (Degrees). Để hiển thị 3D đúng tỷ lệ 1:1:1, cần chuyển sang mét.")
        reproject_opt = st.checkbox("🔄 Tự động chuyển sang hệ mét (EPSG:3857)", value=True)
        
        if reproject_opt:
            with st.spinner("Đang reproject sang EPSG:3857..."):
                data, transform, crs = reproject_to_metric(data, transform, crs)
            st.info(f"Đã chuyển sang: {crs}")

    # --- Sidebar: Visualization Controls ---
    st.sidebar.header("3. Visualization")
    
    # Colormap
    cmap_options = ['Earth', 'Viridis', 'Plasma', 'Turbo', 'Gray', 'RdBu', 'Jet']
    selected_cmap = st.sidebar.selectbox("Colormap", cmap_options, index=0)
    
    # Vertical Exaggeration
    z_scale = st.sidebar.slider("Vertical Exaggeration (Z-Scale)", 0.1, 10.0, 1.0, 0.1)
    
    # Z Limits
    min_z_val = float(np.nanmin(data))
    max_z_val = float(np.nanmax(data))
    
    z_range = st.sidebar.slider(
        "Giới hạn độ cao (Clip Z)",
        min_value=min_z_val,
        max_value=max_z_val,
        value=(min_z_val, max_z_val)
    )
    
    # Grid Toggle
    show_grid = st.sidebar.checkbox("Hiển thị lưới tọa độ", value=True)

    # --- Render 3D ---
    st.markdown("---")
    st.subheader("Interactive 3D Surface")
    
    with st.spinner("Đang dựng hình 3D..."):
        # Chuẩn bị dữ liệu XYZ
        X, Y, Z = prepare_xyz(data, transform)
        
        # Vẽ biểu đồ
        fig = plot_3d_surface(
            X, Y, Z, 
            colormap=selected_cmap, 
            z_scale=z_scale, 
            show_grid=show_grid,
            z_min=z_range[0],
            z_max=z_range[1]
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Vui lòng upload file DEM (.tif) ở thanh bên trái để bắt đầu.")
    st.markdown("""
    ### Hướng dẫn:
    1. Chuẩn bị file DEM format GeoTIFF.
    2. Upload file vào sidebar.
    3. Điều chỉnh các thông số hiển thị.
    4. Tương tác với biểu đồ (Xoay, Zoom, Pan).
    """)