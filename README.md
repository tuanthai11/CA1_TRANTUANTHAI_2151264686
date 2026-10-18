# SAGE-DINOv3: Replacing Full Encoder (Backbone 1 & 2) with DINOv3

Dự án thực hiện thay thế **TOÀN BỘ ENCODER** (bao gồm cả Backbone 1 Encoder và Backbone 2 Encoder) trong sơ đồ kiến trúc **SAGE (Shape-Adapting Gated Experts)** bằng mô hình tự giám sát **DINOv3 (Vision Transformer)**.

---

## 📍 Các Vị Trí Đã Cập Nhật (Update Locations in Architecture)

Dựa trên sơ đồ kiến trúc SAGE:

### 1. Thay thế Toàn bộ Encoder bằng DINOv3 (`SAGE_Full_DINOv3_Segmentation`)
- **Vị trí trong sơ đồ:** Thay thế khối `Backbone 1 Encoder (E1...Em)` và `Backbone 2 Encoder (B1...Bn)`.
- **Vị trí trong code:** Khai báo `self.dinov3_encoder = timm.create_model('facebookresearch/dinov3-vitb16-pretrain-lvd1689m')`.
- **Chi tiết:** 
  - Các tầng đặc trưng sớm (Early Layers) của DINOv3 đóng vai trò là **Backbone 1 Encoder** ($E_1 \dots E_m$).
  - Các tầng đặc trưng sâu (Deep Layers) của DINOv3 đóng vai trò là **Backbone 2 Encoder** ($B_1 \dots B_n$).

### 2. Định tuyến Router & Cổng Shape-Adapting Hub ($S_{in}, S_{out}$)
- **Vị trí trong sơ đồ:** Khối `Router`, `S_in` (Main Block InputShape), `S_out` (Main Block OutputShape).
- **Vị trí trong code:** Lớp `HierarchicalRouter` và `ShapeAdaptingHub`.
- **Chi tiết:** Nhận đặc trưng từ Backbone 1 DINOv3 ($Z_{i-1}$), căn chỉnh kích thước qua $S_{in}$, truyền qua các `Expert Blocks e_k` ở Backbone 2, và tổng hợp lại qua $S_{out}$ vào Backbone 1 ($\hat{Z}_i^{(k)}$).

### 3. Skip Connections & Decoder ($D_m \dots D_1 
ightarrow 	ext{Projection}$)
- **Vị trí trong sơ đồ:** Khối `Decoder` màu hồng phía dưới và `Skip Connection`.
- **Vị trí trong code:** Khối `self.decoder_d1` và `self.projection`.
- **Chi tiết:** Kết nối trực tiếp đường truyền đặc trưng từ DINOv3 Backbone 1 & Backbone 2 xuống các tầng Decoder để khôi phục độ phân giải gốc $H 	imes W 	imes 3$.

---

## 🚀 Hướng dẫn chạy file Notebook
1. Mở notebook `SAGE_Full_DINOv3_Architecture.ipynb`.
2. Chọn **Run All Cells** để tiến hành khởi tạo DINOv3 Encoder và kiểm tra luồng dữ liệu.
