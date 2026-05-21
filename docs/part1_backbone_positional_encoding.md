# Phần 1: CNN Backbone & 2D Spatial Positional Encoding 🖼️

Trong phần này, chúng ta sẽ xây dựng hai thành phần nền móng đầu tiên của DETR:
1. **CNN Backbone**: Trích xuất đặc trưng không gian nâng cao (high-level features) từ ảnh đầu vào.
2. **2D Spatial Positional Encoding**: Tạo bản đồ thông tin vị trí hai chiều dạng Sinusoidal để "dạy" cho Transformer biết các điểm ảnh nằm ở đâu trong không gian 2D.

---

## 1. Kiến Trúc Tổng Quan Của Bước Sơ Chế Đặc Trưng

Trong mô hình DETR, một ảnh tự nhiên đầu vào $x_{img} \in \mathbb{R}^{3 \times H_0 \times W_0}$ sẽ đi qua hai khối chức năng này để tạo ra:
- **Feature Map ($x$)**: Có kích thước giảm đi 32 lần (với ResNet-50), kích thước cụ thể là $d \times H \times W$, trong đó $d$ là số kênh đặc trưng của Transformer (thường là 256).
- **Positional Encoding ($pos$)**: Bản đồ vị trí có kích thước tương đồng $d \times H \times W$ để cộng trực tiếp (hoặc ghép) vào các khoá (Keys) và truy vấn (Queries) trong Transformer.

```
       [Ảnh Đầu Vào] (3 x H0 x W0)
             │
             ▼
      ┌──────────────┐
      │ CNN Backbone │  (e.g., ResNet-50)
      └──────────────┘
             │
             ▼  Feature Map (C x H x W)  (e.g., 2048 x H/32 x W/32)
      ┌──────────────┐
      │  Conv 1x1    │  (Chiếu kênh xuống d_model, e.g., 256)
      └──────────────┘
             │
             ├──────────────────────────┐
             ▼                          ▼
      [Features (x)]            ┌──────────────┐
      (d x H x W)               │  2D Pos Enc  │
                                └──────────────┘
                                        │
                                        ▼
                                 [Pos Enc (pos)]
                                   (d x H x W)
```

---

## 2. Xây Dựng CNN Backbone Wrapper

Mặc dù bài báo DETR cho phép dùng bất kỳ mạng tích chập nào làm Backbone, ResNet-50 là lựa chọn phổ biến nhất. Đầu ra của ResNet-50 ở block cuối cùng có kích thước kênh là $2048$. 

Chúng ta cần chiếu (project) số kênh $2048$ này xuống số chiều mà Transformer làm việc ($d_{model} = 256$) bằng một layer Convolution $1 \times 1$.

### Cài đặt lớp `Backbone` bằng PyTorch:

Chúng ta sẽ kế thừa và bọc lại lớp ResNet từ `torchvision.models` nhưng loại bỏ các layer phân lớp (fully connected layers) và giữ lại cơ chế trích xuất đặc trưng.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):
    """
    Module Backbone sử dụng mạng ResNet để trích xuất feature map.
    Sau đó chiếu (project) số kênh đặc trưng về d_model (256 chiều).
    """
    def __init__(self, name: str = 'resnet50', train_backbone: bool = True, d_model: int = 256):
        super().__init__()
        # Load mô hình pretrained từ torchvision
        backbone = getattr(models, name)(pretrained=True)
        
        # Chỉ giữ lại các layer trích xuất đặc trưng của ResNet
        # Loại bỏ avgpool và fc
        self.body = nn.Sequential(*list(backbone.children())[:-2])
        
        # Đóng băng hoặc cho phép huấn luyện các tham số của backbone
        for name, parameter in self.body.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)
                
        # Xác định số lượng kênh đầu ra của Backbone (ResNet50 là 2048, ResNet18/34 là 512)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        
        # Lớp Convolution 1x1 để giảm số kênh từ num_channels xuống d_model (256)
        self.conv_proj = nn.Conv2d(num_channels, d_model, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor đầu vào có kích thước [Batch_size, 3, H, W]
        Trả về: Tensor đặc trưng đã chiếu có kích thước [Batch_size, d_model, H/32, W/32]
        """
        # Trích xuất đặc trưng thô qua ResNet body
        # e.g., [B, 3, 600, 800] -> [B, 2048, 19, 25] (với ResNet-50 giảm 32 lần)
        features = self.body(x)
        
        # Chiếu số kênh đặc trưng xuống d_model (256)
        # [B, 2048, 19, 25] -> [B, 256, 19, 25]
        proj_features = self.conv_proj(features)
        
        return proj_features
```

---

## 3. Bản Chất Của 2D Spatial Positional Encoding

Khác với các chuỗi văn bản (NLP) có cấu trúc 1 chiều tuần tự, ảnh kỹ thuật số là một lưới 2 chiều. Vì vậy, ta không thể dùng trực tiếp Positional Encoding 1 chiều thông thường. Chúng ta cần định nghĩa **2D Positional Encoding** để mã hóa đồng thời cả tọa độ dọc ($y$) và tọa độ ngang ($x$).

Mô tả thuật toán sinh mã hóa vị trí 2D dạng hình sin (Sine 2D Positional Encoding) trong DETR:
1. Cho một đặc trưng có kích thước $H \times W$ và số chiều mong muốn là $d_{model}$ (phải là số chẵn, thường $d_{model}=256$).
2. Ta chia $d_{model}$ thành 2 phần bằng nhau: $d_{model}/2$ (128) chiều để mã hóa tọa độ trục $X$ và $d_{model}/2$ (128) chiều để mã hóa tọa độ trục $Y$.
3. Với mỗi vị trí lưới $(y, x)$ (trong đó $y \in [0, H-1]$ và $x \in [0, W-1]$):
   * Chuẩn hóa tọa độ thành số thực. Trong DETR gốc, họ xử lý cả các vùng đệm (mask) bằng cách cộng dồn tích lũy, nhưng ở dạng triển khai cơ bản từ đầu, ta có thể chuẩn hóa tọa độ tuyến tính trong khoảng $[0, 2\pi]$.
   * Tính toán các bước sóng hình sin với tần số khác nhau:
     $$\omega_i = \frac{1}{10000^{\frac{2i}{d/2}}} \quad \text{với } i \in [0, \frac{d}{4} - 1]$$
   * Tạo vector mã hóa cho trục $X$ (gồm các hàm $\sin$ và $\cos$):
     $$PE_X(x, 2i) = \sin(x \cdot \omega_i)$$
     $$PE_X(x, 2i+1) = \cos(x \cdot \omega_i)$$
   * Tạo vector mã hóa cho trục $Y$ tương tự:
     $$PE_Y(y, 2i) = \sin(y \cdot \omega_i)$$
     $$PE_Y(y, 2i+1) = \cos(y \cdot \omega_i)$$
4. Ghép (concatenate) hai vector mã hóa vị trí $PE_X$ và $PE_Y$ dọc theo trục kênh đặc trưng để nhận được vector mã hóa vị trí 2D hoàn chỉnh có chiều dài $d_{model}$.

```
    [Tọa độ Y] (128 chiều) ────► Sine/Cosine Enc ──┐
                                                  ├──► Concatenate (256 chiều)
    [Tọa độ X] (128 chiều) ────► Sine/Cosine Enc ──┘
```

### Cài đặt lớp `PositionEmbeddingSine2D` bằng PyTorch:

```python
import math

class PositionEmbeddingSine2D(nn.Module):
    """
    Tạo mã hóa vị trí 2 chiều (2D Spatial Positional Encoding) dạng Sine/Cosine
    cho đặc trưng ảnh đầu vào.
    """
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats # d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor đặc trưng [Batch_size, C, H, W] (Trong đó C == d_model)
        Trả về: Tensor mã hóa vị trí [Batch_size, C, H, W] tương ứng
        """
        # Lấy kích thước batch, chiều cao, chiều rộng
        b, c, h, w = x.shape
        
        # 1. Tạo ma trận tọa độ tích lũy cho trục Y và X
        # y_embed có kích thước [H, W] chứa chỉ số hàng [0, 1, 2, ..., H-1] lặp lại trên các cột
        y_embed = torch.arange(1, h + 1, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(1, w)
        # x_embed có kích thước [H, W] chứa chỉ số cột [0, 1, 2, ..., W-1] lặp lại trên các hàng
        x_embed = torch.arange(1, w + 1, dtype=torch.float32, device=x.device).unsqueeze(0).repeat(h, 1)
        
        # 2. Chuẩn hóa tọa độ về khoảng [0, 2 * pi]
        if self.normalize:
            eps = 1e-6
            # Chuẩn hóa tuyến tính chia cho giá trị lớn nhất cộng eps để tránh chia cho 0
            y_embed = y_embed / (y_embed[-1, :, None] + eps) * 2 * math.pi
            x_embed = x_embed / (x_embed[:, -1, None] + eps) * 2 * math.pi
            
        # 3. Tạo vector tần số (tần số góc omega)
        # dim_t = [0, 1, 2, ..., num_pos_feats-1]
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # Áp dụng công thức giảm tần số theo hàm lũy thừa
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        # 4. Tính toán mã hóa Sine và Cosine
        # y_embed: [H, W] -> [H, W, 1] / [128] -> [H, W, 128]
        pos_y = y_embed.unsqueeze(-1) / dim_t
        # x_embed: [H, W] -> [H, W, 1] / [128] -> [H, W, 128]
        pos_x = x_embed.unsqueeze(-1) / dim_t
        
        # Phân tách sin cho các chỉ số chẵn, cos cho các chỉ số lẻ
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        
        # 5. Ghép kết quả trục X và Y lại với nhau
        # pos_y: [H, W, 128], pos_x: [H, W, 128] -> [H, W, 256]
        pos = torch.cat((pos_y, pos_x), dim=2)
        
        # 6. Biến đổi kích thước đầu ra phù hợp định dạng [B, C, H, W]
        # [H, W, C] -> [C, H, W] -> [1, C, H, W] -> [B, C, H, W]
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)
        
        return pos
```

---

## 4. Chạy Thử Nghiệm Tích Hợp (Sanity Check)

Hãy viết một script ngắn để kiểm tra kích thước tensor đầu ra của cả Backbone và Module mã hóa vị trí 2D này. Bạn có thể tạo file `test_part1.py` trong thư mục chính và chạy thử.

```python
if __name__ == '__main__':
    # Giả lập một batch ảnh đầu vào: Batch_size=2, 3 kênh màu, kích thước 600x800
    dummy_images = torch.randn(2, 3, 600, 800)
    print(f"Kích thước ảnh đầu vào: {dummy_images.shape}")
    
    # Khởi tạo backbone (sử dụng ResNet-50 mặc định)
    backbone = Backbone(name='resnet50', train_backbone=False, d_model=256)
    
    # Khởi tạo bộ sinh Positional Encoding
    pos_embedder = PositionEmbeddingSine2D(num_pos_feats=128, normalize=True)
    
    # Trích xuất đặc trưng
    features = backbone(dummy_images)
    print(f"Kích thước Feature Map sau Backbone (đã giảm 32 lần): {features.shape}")
    
    # Tạo mã hóa vị trí
    pos = pos_embedder(features)
    print(f"Kích thước Positional Encoding: {pos.shape}")
    
    # Đảm bảo kích thước khớp hoàn hảo
    assert features.shape == pos.shape, "Lỗi: Kích thước của đặc trưng và mã hóa vị trí không khớp nhau!"
    print("🎉 Thành công! Cả hai module hoạt động chính xác về mặt kích thước tensor!")
```

### Kết quả mong đợi khi chạy:
```text
Kích thước ảnh đầu vào: torch.Size([2, 3, 600, 800])
Kích thước Feature Map sau Backbone (đã giảm 32 lần): torch.Size([2, 256, 19, 25])
Kích thước Positional Encoding: torch.Size([2, 256, 19, 25])
🎉 Thành công! Cả hai module hoạt động chính xác về mặt kích thước tensor!
```

---

> [!NOTE]
> **Điểm mấu chốt**: Trong DETR, mã hóa vị trí 2D được cộng trực tiếp vào các vector **Keys** và **Queries** ở tất cả các lớp self-attention và cross-attention của Transformer, chứ không chỉ cộng vào đầu vào như các mô hình NLP truyền thống. Điều này giúp Transformer liên tục ghi nhớ được cấu trúc không gian ảnh ở mọi mức độ biểu diễn sâu.

Hãy cùng bước sang **[Phần 2: Custom Transformer Từ Đầu](part2_transformer.md)** để tìm hiểu cách xây dựng lõi xử lý chuỗi thông tin này!
