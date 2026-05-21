# Phần 5: Lắp Ráp Mô Hình DETR & Pipeline Huấn Luyện 🚀

Chào mừng bạn đến với chương cuối cùng! Chúng ta đã có đầy đủ các mảnh ghép quan trọng:
1. `Backbone` & `PositionEmbeddingSine2D` (Phần 1)
2. `CustomTransformer` (Phần 2)
3. `HungarianMatcher` (Phần 3)
4. `SetCriterion` (Phần 4)

Bây giờ, chúng ta sẽ **lắp ráp tất cả thành mô hình DETR hoàn chỉnh**, đồng thời thiết lập **Pipeline Huấn Luyện (Training Pipeline)** và viết code **Dự Đoán (Inference/Visualization)**.

---

## 1. Bản Đồ Dữ Liệu Chạy Trong Mô Hình DETR

Trước khi đi vào viết mã nguồn, hãy cùng xem luồng dữ liệu di chuyển từ đầu vào cho đến đầu ra:

```
[Ảnh Đầu Vào: B x 3 x H x W]
         │
         ▼
     [Backbone] ────────────────────────► Conv 1x1 ──► [Feature Map: B x 256 x H/32 x W/32]
         │                                                      │
         ▼                                                      ▼
  [Positional Sine 2D] ──► [Pos Enc: B x 256 x H/32 x W/32]   [Duỗi phẳng & Hoán vị]
         │                                                      │
         ▼                                                      ▼
   [Flatten & Permute] ─────────────────────────────────► [Memory: HW x B x 256]
         │                                                      │
         └─────────────────────────┐                            │
                                   ▼                            ▼
  [Object Queries: 100 x 256] ──► [Transformer Decoder] ◄───────┘
                                   │
                                   ▼
                   [Output States: 6 x B x 100 x 256]
                                   │
                ┌──────────────────┴──────────────────┐
                ▼                                     ▼
        [Class Head (Linear)]                 [Box Head (MLP)]
      [B x 100 x Num_Classes + 1]               [B x 100 x 4]
```

---

## 2. Viết Code Lớp MLP (Multi-Layer Perceptron) Cho Bounding Box

Đầu ra của Transformer Decoder là các vector đặc trưng của các Object Queries. Để dự đoán tọa độ hộp bao ($cx, cy, w, h$), DETR sử dụng một mạng nơ-ron truyền thẳng 3 lớp (3-layer MLP) có hàm kích hoạt ReLU.

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Mạng Multi-Layer Perceptron (đầu dự đoán tọa độ hộp bao).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # Tạo chuỗi các tầng tuyến tính liên tiếp
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor đầu ra của Decoder
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
```

---

## 3. Lắp Ráp Mô Hình DETR Hoàn Chỉnh

Bây giờ ta gộp Backbone, Transformer, Class Head, và Bounding Box Head vào module chính `DETR`.

```python
import torch.nn.functional as F

# Import các thành phần từ các phần trước
from part1_backbone_positional_encoding import Backbone, PositionEmbeddingSine2D
from part2_transformer import CustomTransformer

class DETR(nn.Module):
    """
    Kiến trúc mạng DETR hoàn chỉnh.
    """
    def __init__(self, num_classes: int = 80, num_queries: int = 100, 
                 d_model: int = 256, nhead: int = 8, 
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, 
                 aux_loss: bool = True):
        super().__init__()
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        
        # 1. Khởi tạo Backbone CNN và mã hóa vị trí 2D
        self.backbone = Backbone(name='resnet50', train_backbone=True, d_model=d_model)
        self.pos_embed = PositionEmbeddingSine2D(num_pos_feats=d_model // 2, normalize=True)
        
        # 2. Khởi tạo khối Transformer giải mã từ đầu
        self.transformer = CustomTransformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # 3. Khởi tạo Object Queries (tham số học được của Decoder)
        # Tương đương nn.Embedding(num_queries, d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # 4. Thiết lập các đầu dự đoán song song (Prediction Heads)
        # Phân lớp: thêm 1 lớp đại diện cho class rỗng (no-object)
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        # Định vị: MLP 3 lớp xuất ra [center_x, center_y, width, height]
        self.bbox_embed = MLP(d_model, d_model, output_dim=4, num_layers=3)
        
    def forward(self, samples: torch.Tensor) -> dict:
        """
        samples: Batch ảnh đầu vào [Batch_size, 3, H, W]
        """
        # Bước 1: Trích xuất đặc trưng ảnh từ Backbone
        features = self.backbone(samples)
        
        # Bước 2: Tạo mã hóa vị trí 2D tương ứng
        pos = self.pos_embed(features)
        
        # Bước 3: Truyền qua khối Custom Transformer
        # hs shape: [num_decoder_layers, Batch_size, Num_Queries, d_model]
        hs = self.transformer(features, self.query_embed.weight, pos)
        
        # Bước 4: Lấy đầu ra của layer Decoder cuối cùng để dự đoán
        # shape: [Batch_size, Num_Queries, d_model]
        last_decoder_out = hs[-1]
        
        # Dự đoán nhãn phân lớp và tọa độ hộp
        outputs_class = self.class_embed(last_decoder_out)
        outputs_coord = self.bbox_embed(last_decoder_out).sigmoid() # Chuẩn hóa tọa độ hộp về [0, 1]
        
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        
        # Bước 5: Nếu cấu hình sử dụng Auxiliary Loss, lấy đầu ra của toàn bộ các tầng Decoder trung gian
        if self.aux_loss:
            # Dự đoán nhãn và box cho các tầng Decoder 0, 1, 2, 3, 4
            out['aux_outputs'] = self._set_aux_metadata(hs[:-1])
            
        return out
        
    @torch.jit.unused
    def _set_aux_metadata(self, hs: torch.Tensor) -> list:
        # Hàm bổ trợ chuyển đổi tensor các tầng trung gian thành danh sách các dự đoán riêng lẻ
        return [{'pred_logits': self.class_embed(as_tensor), 
                 'pred_boxes': self.bbox_embed(as_tensor).sigmoid()} 
                for as_tensor in hs]
```

---

## 4. Xây Dựng Pipeline Huấn Luyện (Training Loop)

Để mô phỏng quá trình huấn luyện thực tế, chúng ta sẽ tạo một **Dataset Giả Lập** sinh ngẫu nhiên các ảnh và các nhãn vật thể dạng hình học, sau đó viết một vòng lặp huấn luyện hoàn chỉnh sử dụng optimizer `AdamW` với cơ chế giảm tốc độ học (learning rate scheduler).

```python
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class DummyDetectionDataset(Dataset):
    """
    Dataset giả lập để sinh ngẫu nhiên dữ liệu ảnh và nhãn đối tượng.
    """
    def __init__(self, size: int = 100, num_classes: int = 80):
        self.size = size
        self.num_classes = num_classes
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Sinh ngẫu nhiên 1 ảnh kích thước 3 x 256 x 256
        image = torch.rand(3, 256, 256)
        
        # Sinh ngẫu nhiên số lượng đối tượng (từ 1 đến 5 vật thể)
        num_objs = torch.randint(1, 6, (1,)).item()
        
        # Sinh nhãn và tọa độ cx, cy, w, h ngẫu nhiên
        labels = torch.randint(0, self.num_classes, (num_objs,), dtype=torch.long)
        boxes = torch.rand(num_objs, 4)
        
        # Đảm bảo box hợp lệ bằng cách ép rộng và cao nhỏ hơn tâm
        boxes[:, 2:] = boxes[:, 2:] * 0.4  # Giới hạn kích thước hộp
        
        return image, {"labels": labels, "boxes": boxes}

def collate_fn(batch):
    # Hàm tùy biến ghép mẫu (collate) để xử lý kích thước nhãn động (vì mỗi ảnh có số lượng vật thể khác nhau)
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    criterion.train()
    
    epoch_loss = 0.0
    for images, targets in dataloader:
        images = images.to(device)
        # Chuyển target lên GPU/CPU tương ứng
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        outputs = model(images)
        
        # Tính toán loss đa nhiệm qua SetCriterion
        loss_dict = criterion(outputs, targets)
        
        # Tính tổng loss có trọng số
        losses = sum(loss_dict[k] * criterion.weight_dict[k[:-2] if any(k.endswith(s) for s in ['_0','_1','_2','_3','_4','_5']) else k] 
                     for k in loss_dict.keys())
        
        # Backward & Optimize
        optimizer.zero_grad()
        losses.backward()
        # Clip gradient để tránh bùng nổ gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        
        epoch_loss += losses.item()
        
    return epoch_loss / len(dataloader)
```

---

## 5. Script Huấn Luyện & Kiểm Thử Tích Hợp

Chúng ta sẽ chạy một đoạn mã hoàn chỉnh để huấn luyện mô hình DETR tự xây dựng từ đầu trên Dataset giả lập này:

```python
if __name__ == '__main__':
    from part3_hungarian_matcher import HungarianMatcher
    from part4_loss_criterion import SetCriterion
    
    # 1. Cấu hình phần cứng
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang sử dụng thiết bị: {device}")
    
    # 2. Khởi tạo Mô hình DETR
    num_classes = 80
    model = DETR(
        num_classes=num_classes, 
        num_queries=20, # Sử dụng 20 queries cho mô phỏng nhanh
        num_encoder_layers=2, # Giảm số layer xuống 2 để huấn luyện nhanh trên CPU/GPU cá nhân
        num_decoder_layers=2,
        aux_loss=True
    ).to(device)
    
    # 3. Khởi tạo Matcher và Hàm Loss
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
    
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
        losses=['labels', 'boxes']
    ).to(device)
    
    # 4. Chuẩn bị Dữ liệu
    dataset = DummyDetectionDataset(size=40, num_classes=num_classes)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # 5. Khởi tạo Optimizer & LR Scheduler
    # Trong DETR, ta thường đặt tốc độ học của backbone nhỏ hơn (ví dụ 1e-5) so với transformer (1e-4)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-5,
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
    
    # 6. Vòng lặp huấn luyện thử nghiệm (3 Epochs)
    print("--- BẮT ĐẦU HUẤN LUYỆN THỬ NGHIỆM ---")
    for epoch in range(1, 4):
        loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch}/3 | Loss trung bình: {loss:.4f}")
        
    print("🎉 Huấn luyện thành công!")
```

---

## 6. Pipeline Dự Đoán & Trực Quan Hóa (Inference & Visualization)

Khi chạy dự đoán (inference), chúng ta không cần dùng đến Hungarian Matcher hay SetCriterion. Chúng ta chỉ cần chạy mô hình ở chế độ `eval` để lấy các dự đoán, lọc các kết quả có độ tự tin cao bằng một ngưỡng tự thiết lập (ví dụ $> 0.7$), và chuyển tọa độ chuẩn hóa về tọa độ ảnh thực tế để vẽ đồ họa.

### Triển khai code Inference:

```python
import matplotlib.pyplot as plt
import cv2

@torch.no_grad()
def detect_and_visualize(model, image_path: str, device, threshold: float = 0.7):
    """
    Nhận diện vật thể trên ảnh thực tế và hiển thị kết quả.
    """
    model.eval()
    
    # 1. Đọc và tiền xử lý ảnh
    orig_img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    h, w, _ = orig_img.shape
    
    # Biến đổi ảnh thành tensor và chuẩn hóa
    img_tensor = torch.as_tensor(orig_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    # Thêm chiều Batch: [3, H, W] -> [1, 3, H, W]
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 2. Dự đoán qua mô hình DETR
    outputs = model(img_tensor)
    
    # Lấy phân phối xác suất và bounding boxes
    # out_logits shape: [1, Num_Queries, Num_Classes + 1] -> [Num_Queries, Num_Classes + 1]
    # out_boxes shape: [1, Num_Queries, 4] -> [Num_Queries, 4]
    out_logits = outputs['pred_logits'][0]
    out_boxes = outputs['pred_boxes'][0]
    
    # Tính xác suất phân lớp dùng Softmax trên tất cả các lớp (bỏ lớp rỗng cuối cùng)
    prob = out_logits.softmax(-1)[:, :-1]
    # Lấy lớp có điểm cao nhất và giá trị điểm của nó
    scores, labels = prob.max(-1)
    
    # 3. Lọc các dự đoán dựa trên ngưỡng threshold
    keep = scores > threshold
    keep_scores = scores[keep].cpu().numpy()
    keep_labels = labels[keep].cpu().numpy()
    keep_boxes = out_boxes[keep].cpu().numpy()
    
    # 4. Trực quan hóa kết quả bằng Matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(orig_img)
    ax = plt.gca()
    
    for score, label, box in zip(keep_scores, keep_labels, keep_boxes):
        # Chuyển đổi hộp từ tọa độ [cx, cy, w, h] tương đối về [x_min, y_min, x_max, y_max] tuyệt đối
        cx, cy, box_w, box_h = box
        xmin = (cx - 0.5 * box_w) * w
        ymin = (cy - 0.5 * box_h) * h
        xmax = (cx + 0.5 * box_w) * w
        ymax = (cy + 0.5 * box_h) * h
        
        # Vẽ hình chữ nhật
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                             fill=False, color='red', linewidth=3)
        ax.add_patch(rect)
        
        # Thêm nhãn chữ hiển thị lớp và độ tự tin
        text = f'Class {label}: {score:.2f}'
        ax.text(xmin, ymin - 5, text, bbox=dict(facecolor='yellow', alpha=0.5), 
                fontsize=12, color='black')
                
    plt.axis('off')
    plt.savefig('inference_result.png', bbox_inches='tight')
    plt.show()
    print("🎉 Đã thực hiện dự đoán và lưu kết quả vẽ hộp vào file 'inference_result.png'!")
```

---

## 7. Tổng Kết Kiến Thức Về Kiến Trúc DETR 🌟

Chúc mừng bạn đã hoàn thành loạt bài hướng dẫn xây dựng **DETR từ đầu không dùng thư viện cao cấp**! Chúng ta hãy cùng điểm lại các cuộc cách mạng thiết kế mà DETR mang lại:

| Thành phần | Đặc điểm trong DETR | Lợi ích mang lại |
| :--- | :--- | :--- |
| **Backbone CNN** | Trích xuất bản đồ đặc trưng 2D. | Giữ lại sức mạnh học không gian mạnh mẽ của CNN. |
| **2D Positional Encoding** | Mã hóa vị trí Sine 2D tách biệt theo trục X và Y. | Dạy cho Transformer biết cấu trúc lưới hình học của ảnh. |
| **Object Queries** | Tập hợp $N$ vector tham số học được hoạt động như các "câu hỏi". | Loại bỏ hoàn toàn sự cần thiết của các hộp neo (Anchor boxes) cố định thủ công. |
| **Transformer Decoder** | Cơ chế Cross-Attention tương tác đa-đối-đa. | Các đối tượng tự thương lượng vị trí để không dự đoán trùng nhau. |
| **Hungarian Matcher** | Gán nhãn song phương tối ưu theo tỉ lệ 1-đối-1. | **Loại bỏ hoàn toàn bước lọc trùng NMS** phức tạp ở cuối pipeline. |
| **Multi-task Loss (Set Criterion)** | Kết hợp Cross Entropy, $L_1$ Loss và GIoU Loss. | Tối ưu hóa đồng thời cả độ chính xác nhãn lẫn hình học hộp bao. |

---

> [!TIP]
> **Hướng phát triển tiếp theo**: Bạn có thể lưu toàn bộ các code của 5 phần này vào các file `.py` riêng lẻ, tải bộ dữ liệu **COCO** hoặc một tập dữ liệu nhận diện biển báo giao thông nhỏ, áp dụng kỹ thuật Transfer Learning bằng cách load trọng số ResNet-50 có sẵn, và tiến hành huấn luyện thực tế để kiểm chứng sức mạnh của mô hình DETR này nhé!
