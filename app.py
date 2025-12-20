import streamlit as st
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.crs import CRS
import numpy as np
import plotly.graph_objects as go
import base64

# --- Cấu hình trang (Page Config) ---
st.set_page_config(
    layout="wide", 
    page_title="GeoSpatial 3D Viewer",
    page_icon="🏔️",
    initial_sidebar_state="expanded"
)

# Custom CSS để làm đẹp giao diện
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    .stAlert {padding: 0.5rem;}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. CÁC HÀM XỬ LÝ (BACKEND)
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_and_downsample_dem(file_content, filename, max_dim=500):
    try:
        with MemoryFile(file_content, filename=filename) as memfile:
            with memfile.open() as dataset:
                profile = dataset.profile
                orig_w, orig_h = dataset.width, dataset.height
                
                # Tính scale factor
                scale = 1
                if orig_w > max_dim or orig_h > max_dim:
                    scale = max_dim / max(orig_w, orig_h)
                
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                
                data = dataset.read(
                    1,
                    out_shape=(new_h, new_w),
                    resampling=Resampling.bilinear
                )
                
                transform = dataset.transform * dataset.transform.scale(
                    (dataset.width / data.shape[1]),
                    (dataset.height / data.shape[0])
                )
                
                if dataset.nodata is not None:
                    data = data.astype('float32')
                    data[data == dataset.nodata] = np.nan
                
                return data, transform, dataset.crs, dataset.nodata, (orig_w, orig_h)
    except Exception as e:
        return None, None, None, None, None

def reproject_to_metric(data, transform, src_crs):
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
    return X, Y, data

def plot_3d_surface(X, Y, Z, colormap, z_scale, show_grid, z_min, z_max, plot_height, title_text):
    Z_plot = Z * z_scale
    
    # Clip dữ liệu hiển thị (không thay đổi dữ liệu gốc)
    if z_min is not None: Z_plot[Z_plot < z_min * z_scale] = z_min * z_scale
    if z_max is not None: Z_plot[Z_plot > z_max * z_scale] = z_max * z_scale

    fig = go.Figure(data=[go.Surface(
        z=Z_plot, x=X, y=Y, 
        colorscale=colormap,
        cmin=np.nanmin(Z_plot), cmax=np.nanmax(Z_plot),
        colorbar=dict(title='Elev (m)', len=0.7, thickness=15),
        hoverinfo='x+y+z',
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
    )])

    fig.update_layout(
        title=dict(text=title_text, x=0, font=dict(size=14, color="#555")),
        autosize=True, 
        height=plot_height, # Chiều cao tùy biến
        margin=dict(l=10, r=10, b=10, t=30),
        scene=dict(
            xaxis=dict(title='', showgrid=show_grid, visible=show_grid, showticklabels=show_grid),
            yaxis=dict(title='', showgrid=show_grid, visible=show_grid, showticklabels=show_grid),
            zaxis=dict(title='', showgrid=show_grid, visible=show_grid, showticklabels=show_grid),
            aspectmode='data', # Giữ tỷ lệ 1:1:1
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)) # Góc nhìn mặc định đẹp hơn
        ),
        paper_bgcolor='rgba(0,0,0,0)', # Nền trong suốt
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# -----------------------------------------------------------------------------
# 2. GIAO DIỆN NGƯỜI DÙNG (SIDEBAR CONTROLS)
# -----------------------------------------------------------------------------

st.sidebar.title("🛠️ Cấu hình dữ liệu")

# --- 1. Upload & Input ---
uploaded_file = st.sidebar.file_uploader("1. Chọn File (TIF/ASC/TXT)", type=['tif', 'tiff', 'asc', 'txt'])

if uploaded_file:
    # --- 2. Xử lý CRS (Ngay trong Sidebar) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Hệ tọa độ (CRS)")
    
    # Load data tạm thời để check CRS
    with st.spinner("Đang đọc dữ liệu..."):
        data, transform, raw_crs, nodata, orig_dims = load_and_downsample_dem(
            uploaded_file.getvalue(), uploaded_file.name, max_dim=2000 # Load full check trước
        )

    if data is None:
        st.sidebar.error("Lỗi format file!")
        st.stop()

    # Logic CRS
    use_crs = raw_crs
    crs_status = "✅ Hợp lệ"
    
    if raw_crs is None:
        st.sidebar.warning("⚠️ File thiếu CRS")
        user_crs_input = st.sidebar.text_input("Nhập EPSG (VD: EPSG:4326)", value="EPSG:4326")
        try:
            use_crs = CRS.from_string(user_crs_input)
        except:
            use_crs = None
            crs_status = "❌ Không hợp lệ"
    else:
        st.sidebar.caption(f"Gốc: {raw_crs.to_string()}")
    
    if use_crs is None:
        st.sidebar.error("Vui lòng nhập CRS đúng.")
        st.stop()
        
    # Reproject Check
    is_geographic = use_crs.is_geographic
    if is_geographic:
        st.sidebar.info("ℹ️ Đang dùng Lat/Lon. Tự động chuyển sang Mét khi vẽ.")

    # --- 3. Cấu hình Hiển thị (Visuals) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Giao diện & Hiệu năng")
    
    with st.sidebar.expander("⚙️ Tùy chỉnh nâng cao", expanded=False):
        max_pixels = st.slider("Độ phân giải (Max Pixel)", 100, 1000, 400, 50, help="Số càng nhỏ chạy càng nhanh.")
        plot_height = st.slider("Chiều cao khung hình (px)", 300, 1000, 500, 50)
        show_grid = st.checkbox("Hiển thị lưới", True)

    st.sidebar.markdown("🎨 **Màu sắc & Hình khối**")
    cmap = st.sidebar.selectbox("Bảng màu", ['Earth', 'Viridis', 'Plasma', 'Turbo', 'RdBu', 'Magma'], index=0)
    z_scale = st.sidebar.slider("Độ cao (Z-Scale)", 0.5, 5.0, 1.0, 0.1)

    # Reload với resolution user chọn
    data, transform, _, _, _ = load_and_downsample_dem(
        uploaded_file.getvalue(), uploaded_file.name, max_dim=max_pixels
    )
    
    # Xử lý reproject final
    final_data = data
    final_transform = transform
    if is_geographic:
        final_data, final_transform, _ = reproject_to_metric(data, transform, use_crs)

    # --- 4. Main Display ---
    # Layout chính chia làm 2 phần: Header/Stats và Plot
    
    st.title("🏔️ 3D Terrain Viewer")
    
    # Tính toán thống kê
    min_z, max_z = float(np.nanmin(final_data)), float(np.nanmax(final_data))
    mean_z = float(np.nanmean(final_data))
    
    # Hiển thị Metrics đẹp mắt
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Min Elev", f"{min_z:.1f} m")
    col_m2.metric("Max Elev", f"{max_z:.1f} m")
    col_m3.metric("Mean Elev", f"{mean_z:.1f} m")
    col_m4.metric("Grid Size", f"{final_data.shape[1]}x{final_data.shape[0]}")

    # Slider cắt lớp (Z Range) đặt ngay trên biểu đồ để tiện thao tác
    z_range = st.slider("🔍 Cắt lớp độ cao (Filter Elevation)", min_z, max_z, (min_z, max_z))

    # Vẽ biểu đồ
    with st.spinner("Đang render 3D..."):
        X, Y, Z = prepare_xyz(final_data, final_transform)
        fig = plot_3d_surface(
            X, Y, Z, cmap, z_scale, show_grid, 
            z_range[0], z_range[1], plot_height, 
            f"Mô hình 3D - {uploaded_file.name}"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Nút Download
    # Tạo HTML string cho nút download
    buffer = io.StringIO()
    fig.write_html(buffer, include_plotlyjs='cdn')
    html_bytes = buffer.getvalue().encode()
    encoded = base64.b64encode(html_bytes).decode()
    
    st.markdown(f"""
        <a href="data:text/html;base64,{encoded}" download="3d_terrain.html">
            <button style="background-color:#4CAF50; border:none; color:white; padding:10px 24px; border-radius:4px; cursor:pointer;">
                📥 Tải xuống bản đồ 3D (HTML)
            </button>
        </a>
    """, unsafe_allow_html=True)

else:
    # Màn hình chờ (Landing State)
    st.markdown(
        """
        <div style='text-align: center; padding: 50px;'>
            <h1>👋 Chào mừng đến với Geo3D</h1>
            <p>Vui lòng chọn file dữ liệu (<b>.tif, .asc, .txt</b>) từ thanh menu bên trái để bắt đầu.</p>
            <p style='color: grey; font-size: 0.9em;'>Hỗ trợ hiển thị địa hình 3D, tự động xử lý hệ tọa độ và trực quan hóa dữ liệu GIS.</p>
        </div>
        """, unsafe_allow_html=True
    )
import io # Cần thêm thư viện này cho nút download
