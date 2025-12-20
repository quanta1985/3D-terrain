import streamlit as st
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.crs import CRS
import numpy as np
import plotly.graph_objects as go
import base64
import io

# --- Cấu hình trang (Page Config) ---
st.set_page_config(
    layout="wide", 
    page_title="GeoSpatial 3D Viewer Beta",
    page_icon="🏔️",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    .stAlert {padding: 0.5rem;}
    /* Style cho metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
        color: #00CC96;
    }
    /* Style cho footer bản quyền */
    .footer {
        font-size: 0.85em;
        color: #666;
        text-align: left;
        padding-top: 20px;
        border-top: 1px solid #ddd;
    }
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
                orig_w, orig_h = dataset.width, dataset.height
                
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

def plot_3d_surface(X, Y, Z, colormap, z_scale, show_grid, z_min, z_max, plot_height, title_text, show_contours, use_hillshade):
    
    # 1. Scale dữ liệu Z để vẽ hình khối (Trục Z hiển thị)
    Z_plot = Z * z_scale
    
    # 2. Clip dữ liệu hiển thị (theo slider)
    if z_min is not None: Z_plot[Z_plot < z_min * z_scale] = z_min * z_scale
    if z_max is not None: Z_plot[Z_plot > z_max * z_scale] = z_max * z_scale

    # 3. Hiệu ứng Hillshade
    lighting_effect = dict(ambient=0.4, diffuse=0.5, roughness=0.1, specular=0.4, fresnel=0.1)
    if use_hillshade:
        lighting_effect = dict(ambient=0.3, diffuse=0.6, roughness=0.7, specular=0.1, fresnel=0.1)

    # 4. Đường đồng mức
    contours_cfg = dict(
        z=dict(show=show_contours, usecolormap=False, project_z=False, color="white", width=2)
    )

    # 5. Tạo Surface
    # FIX: Truyền trực tiếp Z (dữ liệu gốc) vào customdata để tooltip đọc
    surface = go.Surface(
        z=Z_plot, # Dữ liệu vẽ (đã nhân scale)
        x=X, 
        y=Y,
        customdata=Z, # <--- Dữ liệu gốc để hiển thị tooltip
        colorscale=colormap,
        cmin=np.nanmin(Z_plot), cmax=np.nanmax(Z_plot),
        
        # Cấu hình Tooltip chuẩn:
        # %{z:.1f}: Giá trị trục Z hiện tại (đã nhân phóng đại)
        # %{customdata:.2f}: Giá trị Z gốc
        hovertemplate=(
            "<b>X:</b> %{x:.1f}<br>"
            "<b>Y:</b> %{y:.1f}<br>"
            "<b>Z (Mô hình):</b> %{z:.1f}<br>"
            "<b>Z (Thực tế):</b> %{customdata:.2f} m"
            "<extra></extra>" # Ẩn phần tên trace bên cạnh
        ),
        
        colorbar=dict(title='Elev (m)', len=0.7, thickness=15, x=0.9),
        lighting=lighting_effect,
        contours=contours_cfg
    )

    fig = go.Figure(data=[surface])

    fig.update_layout(
        title=dict(text=title_text, x=0, font=dict(size=14, color="#555")),
        autosize=True, 
        height=plot_height, 
        margin=dict(l=10, r=10, b=10, t=30),
        scene=dict(
            xaxis=dict(title='', showgrid=show_grid, visible=show_grid, showticklabels=show_grid),
            yaxis=dict(title='', showgrid=show_grid, visible=show_grid, showticklabels=show_grid),
            zaxis=dict(title='', showgrid=show_grid, visible=show_grid, showticklabels=show_grid),
            aspectmode='data',
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.5)) 
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# -----------------------------------------------------------------------------
# 2. GIAO DIỆN NGƯỜI DÙNG (SIDEBAR)
# -----------------------------------------------------------------------------

st.sidebar.title("🛠️ Cấu hình")

# --- 1. Upload ---
uploaded_file = st.sidebar.file_uploader("1. Chọn File (TIF/ASC/TXT)", type=['tif', 'tiff', 'asc', 'txt'])

if uploaded_file:
    # --- 2. Xử lý CRS ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Hệ tọa độ (CRS)")
    
    with st.spinner("Đang đọc dữ liệu..."):
        data, transform, raw_crs, nodata, orig_dims = load_and_downsample_dem(
            uploaded_file.getvalue(), uploaded_file.name, max_dim=1000 
        )

    if data is None:
        st.sidebar.error("Lỗi đọc file! Vui lòng kiểm tra format.")
        st.stop()
    
    # CRS Logic
    use_crs = None
    crs_mode = st.sidebar.radio(
        "Chế độ xác định tọa độ:",
        ("Tự động (Từ file)", "Thủ công (Chọn/Nhập)"),
        horizontal=True
    )
    
    if crs_mode == "Tự động (Từ file)":
        if raw_crs:
            st.sidebar.success(f"✅ Đã tìm thấy: {raw_crs.to_string()}")
            use_crs = raw_crs
        else:
            st.sidebar.warning("⚠️ Không tìm thấy CRS trong file. Vui lòng chuyển sang 'Thủ công'.")
            use_crs = None
    else:
        common_crs = {
            "WGS 84 (EPSG:4326)": "EPSG:4326",
            "Web Mercator (EPSG:3857)": "EPSG:3857",
            "VN-2000 / UTM zone 48N (EPSG:3405)": "EPSG:3405",
            "VN-2000 / UTM zone 49N (EPSG:3406)": "EPSG:3406",
            "WGS 84 / UTM zone 48N (EPSG:32648)": "EPSG:32648",
            "WGS 84 / UTM zone 49N (EPSG:32649)": "EPSG:32649",
            "Nhập mã khác...": "Custom"
        }
        selected_crs_name = st.sidebar.selectbox("Chọn hệ tọa độ:", list(common_crs.keys()))
        
        if selected_crs_name == "Nhập mã khác...":
            custom_epsg = st.sidebar.text_input("Nhập mã EPSG (VD: EPSG:32648)", "")
            if custom_epsg:
                try:
                    use_crs = CRS.from_string(custom_epsg)
                except:
                    st.sidebar.error("Mã không hợp lệ.")
        else:
            use_crs = CRS.from_string(common_crs[selected_crs_name])

    if use_crs is None:
        st.sidebar.error("⛔ Cần xác định CRS để tiếp tục.")
        st.stop()

    # --- 3. Settings Hiển thị ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Giao diện & Hiệu ứng")
    
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        plot_height = st.slider("Chiều cao khung (px)", 300, 1000, 600, 50)
    with col_s2:
        max_pixels = st.slider("Độ phân giải (Max)", 100, 1500, 500, 100)

    st.sidebar.markdown("**🎨 Style Địa hình**")
    cmap = st.sidebar.selectbox("Bảng màu", ['Earth', 'Viridis', 'Plasma', 'Turbo', 'RdBu', 'Magma'], index=0)
    
    col_e1, col_e2 = st.sidebar.columns(2)
    with col_e1:
        show_contours = st.checkbox("Show Contours", value=False)
    with col_e2:
        use_hillshade = st.checkbox("Hillshade Effect", value=True)

    z_scale = st.sidebar.slider("Vertical Exaggeration (Z-Scale)", 0.5, 10.0, 1.5, 0.1)
    show_grid = st.sidebar.checkbox("Hiển thị lưới tọa độ", value=True)

    # --- FOOTER BẢN QUYỀN (SIDEBAR) ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div class='footer'>
            <b>Thông tin ứng dụng:</b><br>
            • Phiên bản: <span style='color:orange; font-weight:bold;'>Beta 1.0</span><br>
            • Tác giả: <b>Trần Anh Quan - HUMG</b><br>
            • Copyright © 2025
        </div>
    """, unsafe_allow_html=True)

    # --- 4. Main Processing ---
    
    # Reload data with final settings
    data, transform, _, _, _ = load_and_downsample_dem(
        uploaded_file.getvalue(), uploaded_file.name, max_dim=max_pixels
    )
    
    # Reproject logic
    final_data = data
    final_transform = transform
    if use_crs.is_geographic:
        final_data, final_transform, _ = reproject_to_metric(data, transform, use_crs)

    # --- DASHBOARD INFO ---
    st.title("🏔️ 3D Terrain Analysis")
    
    min_z, max_z = float(np.nanmin(final_data)), float(np.nanmax(final_data))
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Min Elevation", f"{min_z:.1f} m")
    col_m2.metric("Max Elevation", f"{max_z:.1f} m")
    col_m3.metric("Grid Dimensions", f"{final_data.shape[1]} x {final_data.shape[0]} px")

    z_range = st.slider("🔍 Cắt lớp độ cao (Elevation Filter)", min_z, max_z, (min_z, max_z))

    # --- PLOT 3D ---
    with st.spinner("Đang dựng mô hình 3D..."):
        X, Y, Z = prepare_xyz(final_data, final_transform)
        
        fig = plot_3d_surface(
            X, Y, Z, 
            colormap=cmap, 
            z_scale=z_scale, 
            show_grid=show_grid, 
            z_min=z_range[0], 
            z_max=z_range[1], 
            plot_height=plot_height, 
            title_text=f"Mô hình: {uploaded_file.name} | CRS: {use_crs.to_string()}",
            show_contours=show_contours,
            use_hillshade=use_hillshade
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # --- DOWNLOAD BUTTON ---
    buffer = io.StringIO()
    fig.write_html(buffer, include_plotlyjs='cdn')
    html_bytes = buffer.getvalue().encode()
    encoded = base64.b64encode(html_bytes).decode()
    
    st.markdown(f"""
        <div style="text-align: right;">
        <a href="data:text/html;base64,{encoded}" download="terrain_3d.html">
            <button style="background-color:#4CAF50; border:none; color:white; padding:10px 24px; border-radius:4px; cursor:pointer;">
                📥 Xuất báo cáo HTML
            </button>
        </a>
        </div>
    """, unsafe_allow_html=True)

else:
    # Sửa lỗi hiển thị bị che: Thêm khoảng trắng lớn ở trên
    st.markdown("<br><br><br>", unsafe_allow_html=True) 
    st.info("👈 Vui lòng upload dữ liệu ở Sidebar (Bên trái) để bắt đầu.")
    st.markdown("""
    <div style="text-align:center; color: gray; margin-top: 50px;">
        <h3>GeoSpatial 3D Viewer</h3>
        <p>Hỗ trợ hiển thị TIF, ASC, TXT với khả năng tùy biến 3D mạnh mẽ.</p>
    </div>
    """, unsafe_allow_html=True)
