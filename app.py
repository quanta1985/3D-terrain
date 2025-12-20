import streamlit as st
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.crs import CRS
import numpy as np
import plotly.graph_objects as go

# Cấu hình trang Streamlit
st.set_page_config(layout="wide", page_title="DEM 3D Visualizer (TIF, ASC, TXT)")

# -----------------------------------------------------------------------------
# 1. CÁC HÀM XỬ LÝ (PROCESSING FUNCTIONS)
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_and_downsample_dem(file_content, filename, max_dim=500):
    """
    Đọc file DEM (TIF, ASC, TXT).
    filename được truyền vào để rasterio nhận diện driver (AAIGrid cho .asc/.txt).
    """
    try:
        with MemoryFile(file_content, filename=filename) as memfile:
            with memfile.open() as dataset:
                profile = dataset.profile
                orig_width = dataset.width
                orig_height = dataset.height
                
                # Tính toán scale factor để downsample
                scale = 1
                if orig_width > max_dim or orig_height > max_dim:
                    scale = max_dim / max(orig_width, orig_height)
                
                new_width = int(orig_width * scale)
                new_height = int(orig_height * scale)
                
                # Đọc dữ liệu
                data = dataset.read(
                    1,
                    out_shape=(new_height, new_width),
                    resampling=Resampling.bilinear
                )
                
                # Cập nhật transform
                transform = dataset.transform * dataset.transform.scale(
                    (dataset.width / data.shape[1]),
                    (dataset.height / data.shape[0])
                )
                
                # Xử lý nodata
                if dataset.nodata is not None:
                    data = data.astype('float32')
                    data[data == dataset.nodata] = np.nan
                
                # Lấy CRS gốc (thường None với asc/txt)
                raw_crs = dataset.crs
                
                return data, transform, raw_crs, dataset.nodata
    except Exception as e:
        # Bắt lỗi nếu file txt không đúng định dạng ESRI Grid
        st.error(f"Lỗi khi đọc file: {e}")
        st.error("Lưu ý: File .txt/.asc phải theo chuẩn ESRI ASCII Grid (có header ncols, nrows...).")
        return None, None, None, None

def reproject_to_metric(data, transform, src_crs):
    """
    Chuyển đổi DEM sang EPSG:3857 (Web Mercator - mét).
    """
    dst_crs = 'EPSG:3857' 

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

    destination[destination == 0] = np.nan 
    return destination, new_transform, dst_crs

def prepare_xyz(data, transform):
    rows, cols = data.shape
    xs = np.linspace(transform[2], transform[2] + transform[0] * cols, cols)
    ys = np.linspace(transform[5], transform[5] + transform[4] * rows, rows)
    X, Y = np.meshgrid(xs, ys)
    Z = data
    return X, Y, Z

def plot_3d_surface(X, Y, Z, colormap='Viridis', z_scale=1.0, show_grid=True, z_min=None, z_max=None):
    Z_plot = Z * z_scale
    
    if z_min is not None:
        Z_plot[Z_plot < z_min * z_scale] = z_min * z_scale
    if z_max is not None:
        Z_plot[Z_plot > z_max * z_scale] = z_max * z_scale

    fig = go.Figure(data=[go.Surface(
        z=Z_plot, x=X, y=Y, colorscale=colormap,
        cmin=np.nanmin(Z_plot), cmax=np.nanmax(Z_plot),
        colorbar=dict(title='Elevation')
    )])

    fig.update_layout(
        title='Interactive 3D Terrain', autosize=True, height=700,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(title='X', showgrid=show_grid, visible=show_grid),
            yaxis=dict(title='Y', showgrid=show_grid, visible=show_grid),
            zaxis=dict(title='Z', showgrid=show_grid, visible=show_grid),
            aspectmode='data'
        )
    )
    return fig

# -----------------------------------------------------------------------------
# 2. GIAO DIỆN CHÍNH (UI)
# -----------------------------------------------------------------------------

st.title("🏔️ GIS 3D DEM Visualizer")
st.markdown("Hỗ trợ GeoTIFF (.tif), ESRI ASCII (.asc) và Text Grid (.txt).")

# --- Sidebar: Input ---
st.sidebar.header("1. Data Input")

# Cập nhật: Thêm 'txt' vào danh sách hỗ trợ
uploaded_file = st.sidebar.file_uploader(
    "Upload DEM", 
    type=['tif', 'tiff', 'asc', 'txt'],
    help="Với file .txt/.asc, đảm bảo định dạng là ESRI ASCII Grid."
)

if uploaded_file is not None:
    st.sidebar.header("2. Settings")
    max_pixels = st.sidebar.slider("Max Resolution (px)", 100, 2000, 500, 100)

    # Load Data
    with st.spinner("Đang đọc file..."):
        data, transform, raw_crs, nodata = load_and_downsample_dem(
            uploaded_file.getvalue(), 
            uploaded_file.name, 
            max_dim=max_pixels
        )
    
    # Nếu load lỗi thì dừng
    if data is None:
        st.stop()

    # --- CRS Handling Logic ---
    st.markdown("### 🛠️ Cấu hình tọa độ (CRS)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"CRS gốc từ file: `{raw_crs}`")
    
    use_crs = raw_crs
    
    with col2:
        # Nếu file không có CRS (thường gặp ở .asc/.txt), gợi ý người dùng nhập
        default_crs_input = ""
        if raw_crs is None:
            st.warning("⚠️ File này thiếu thông tin tọa độ (CRS). Vui lòng nhập mã EPSG.")
            default_crs_input = "EPSG:4326" 
        
        user_crs_input = st.text_input(
            "Nhập/Ghi đè CRS (VD: EPSG:4326, EPSG:32648)", 
            value=default_crs_input
        )

    if user_crs_input:
        try:
            use_crs = CRS.from_string(user_crs_input)
            st.success(f"Đang sử dụng CRS: `{use_crs}`")
        except Exception as e:
            st.error(f"Mã CRS không hợp lệ: {e}")
            use_crs = None

    if use_crs is None:
        st.error("⛔ Không thể dựng hình 3D nếu không có hệ tọa độ.")
        st.stop()

    # --- Reproject Logic ---
    is_geographic = use_crs.is_geographic
    final_data = data
    final_transform = transform

    if is_geographic:
        st.warning(f"File đang dùng hệ tọa độ địa lý (Độ). Sẽ tự động chuyển sang Mét (EPSG:3857).")
        with st.spinner("Đang reproject sang hệ mét..."):
            final_data, final_transform, dst_crs = reproject_to_metric(data, transform, use_crs)

    # --- Sidebar Visuals ---
    st.sidebar.header("3. Visualization Controls")
    cmap = st.sidebar.selectbox("Colormap", ['Earth', 'Viridis', 'Plasma', 'Turbo', 'Gray'], index=0)
    z_scale = st.sidebar.slider("Vertical Exaggeration", 0.1, 10.0, 1.0, 0.1)
    
    min_z, max_z = float(np.nanmin(final_data)), float(np.nanmax(final_data))
    z_range = st.sidebar.slider("Z Range", min_z, max_z, (min_z, max_z))
    show_grid = st.sidebar.checkbox("Show Grid", True)

    # --- Render ---
    st.markdown("---")
    with st.spinner("Đang dựng hình 3D..."):
        X, Y, Z = prepare_xyz(final_data, final_transform)
        fig = plot_3d_surface(X, Y, Z, cmap, z_scale, show_grid, z_range[0], z_range[1])
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Upload file DEM (.tif, .asc, .txt) để bắt đầu.")
