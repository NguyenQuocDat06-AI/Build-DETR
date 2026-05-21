# Phần 3: Thuật Toán Khớp Cặp Hungarian Matcher 🤝

Trong các mô hình Object Detection truyền thống (như YOLO, Faster R-CNN), hàng vạn ô bao neo (anchor boxes) hoặc vùng đề xuất (proposals) được tạo ra, dẫn tới hiện tượng nhiều ô bao cùng dự đoán một đối tượng. Hệ quả là ta bắt buộc phải dùng thuật toán lọc trùng **NMS (Non-Maximum Suppression)** ở bước hậu xử lý.

**DETR loại bỏ hoàn toàn NMS!** Làm thế nào? Câu trả lời nằm ở cơ chế **Bipartite Matching (Khớp cặp song phương)** thông qua **Thuật Toán Hungary (Hungarian Algorithm)**. Trong phần này, chúng ta sẽ đi sâu vào toán học và tự tay code module cực kỳ quan trọng này.

---

## 1. Bản Chất Toán Học Của Khớp Cặp Song Phương

Giả sử mạng DETR dự đoán ra đúng $N$ kết quả (thường $N = 100$ tương ứng với 100 Object Queries), ký hiệu tập dự đoán là $\hat{y} = \{\hat{y}_i\}_{i=1}^N$. 

Tập nhãn gốc (ground truth) có số lượng đối tượng thực tế là $M$ (với $M < N$), ký hiệu là $y$. Để kích thước hai tập bằng nhau, ta đệm thêm các phần tử rỗng $\varnothing$ ("no object" - không có đối tượng) vào tập $y$ cho đủ $N$ phần tử.

Bài toán đặt ra: **Tìm một phép hoán vị (matching) $\sigma \in \mathfrak{S}_N$ sao cho tổng chi phí khớp cặp giữa nhãn gốc $y_i$ và dự đoán $\hat{y}_{\sigma(i)}$ là NHỎ NHẤT.**

$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{C}_{match}(y_i, \hat{y}_{\sigma(i)})$$

Trong đó, ma trận chi phí khớp cặp $\mathcal{C}_{match}$ (Matching Cost Matrix) giữa đối tượng thực tế $y_i = (c_i, b_i)$ và đối tượng dự đoán thứ $j$ là $\hat{y}_j = (\hat{p}_j, \hat{b}_j)$ được định nghĩa như sau:

$$\mathcal{C}_{match}(y_i, \hat{y}_j) = - \mathbb{1}_{\{c_i \neq \varnothing\}} \hat{p}_j(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}} \mathcal{L}_{box}(b_i, \hat{b}_j)$$

> [!IMPORTANT]
> **Lưu ý**: Khác với hàm mất mát (Loss) dùng để cập nhật lan truyền ngược, chi phí khớp cặp chỉ dùng để liên kết nhãn. Do đó, phần phân lớp sử dụng trực tiếp xác suất dự đoán $\hat{p}_j(c_i)$ thay vì dùng $-\log \hat{p}_j(c_i)$ để chi phí phân lớp và chi phí tọa độ hộp có cùng thang đo ổn định.

Chi phí hộp $\mathcal{L}_{box}$ bao gồm cả khoảng cách $L_1$ và Generalized IoU (GIoU):

$$\mathcal{L}_{box}(b_i, \hat{b}_j) = \lambda_{L1} \| b_i - \hat{b}_j \|_1 + \lambda_{giou} \mathcal{L}_{giou}(b_i, \hat{b}_j)$$

---

## 2. Giải Thích Thuật Toán Generalized IoU (GIoU)

Chỉ số IoU thông thường bằng $0$ nếu hai bounding box không hề giao nhau, dẫn tới đạo hàm bằng $0$ (gradient biến mất). **GIoU (Generalized Intersection over Union)** giải quyết triệt để vấn đề này bằng cách tính thêm diện tích của hộp bao nhỏ nhất chứa cả hai hộp bao (ký hiệu là $C$).

Công thức tính GIoU giữa hai hộp $A$ và $B$:

$$GIoU = IoU - \frac{|C \setminus (A \cup B)|}{|C|}$$

Trong đó:
* $|A \cup B| = \text{Diện tích vùng hợp của } A \text{ và } B$.
* $C$ là bounding box nhỏ nhất bao quanh cả $A$ và $B$.
* $|C \setminus (A \cup B)| = \text{Diện tích của } C \text{ sau khi loại bỏ phần hợp } A \cup B$.

Hàm mất mát GIoU sẽ là:

$$\mathcal{L}_{giou} = 1 - GIoU$$

```
   ┌──────────────────────────┐  ◄── Hộp Bao Nhỏ Nhất C
   │  ┌──────────┐            │
   │  │  Hộp A   │            │
   │  └──────────┘            │
   │            ┌──────────┐  │
   │            │  Hộp B   │  │
   │            └──────────┘  │
   └──────────────────────────┘
```

---

## 3. Viết Hàm Tính Bounding Box IoU & GIoU Bằng PyTorch

Chúng ta cần viết hàm tính toán GIoU dạng song song (vectorized) trên GPU để đảm bảo hiệu năng tối ưu.

```python
import torch

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Chuyển đổi bounding box từ định dạng [center_x, center_y, width, height]
    sang [x_min, y_min, x_max, y_max].
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Tính IoU giữa hai nhóm hộp bao.
    boxes1: [N, 4] (định dạng xyxy)
    boxes2: [M, 4] (định dạng xyxy)
    
    Trả về:
      iou: Ma trận [N, M] chứa chỉ số IoU giữa các cặp.
      union: Ma trận [N, M] chứa diện tích vùng hợp.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Tìm tọa độ vùng giao nhau
    # [N, 1, 2] vs [1, M, 2] -> [N, M, 2]
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union

def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Tính toán GIoU giữa hai nhóm hộp bao.
    boxes1: [N, 4] (định dạng xyxy)
    boxes2: [M, 4] (định dạng xyxy)
    
    Trả về: Ma trận GIoU kích thước [N, M]
    """
    # Đảm bảo đầu vào ở định dạng xyxy
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    iou, union = box_iou(boxes1, boxes2)
    
    # Tìm tọa độ của hộp bao nhỏ nhất C bao quanh cả hai hộp
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    area_c = wh[:, :, 0] * wh[:, :, 1]  # Diện tích của C

    return iou - (area_c - union) / (area_c + 1e-6)
```

---

## 4. Xây Dựng Lớp HungarianMatcher

Chúng ta sử dụng hàm `linear_sum_assignment` từ thư viện `scipy.optimize` để tìm nghiệm tối ưu cho bài toán gán nhãn tuyến tính này trên CPU. Lớp `HungarianMatcher` sẽ chạy forward theo các bước:
1. Nhận thông tin dự đoán (`pred_logits`, `pred_boxes`) và danh sách nhãn gốc (`targets`).
2. Duỗi phẳng batch để tính toán ma trận chi phí khớp cặp gộp chung cho hiệu năng cao.
3. Tính toán chi phí phân loại, chi phí khoảng cách $L_1$ và chi phí $GIoU$.
4. Tách kết quả theo từng ảnh trong batch và chạy thuật toán Hungary trên CPU.
5. Trả về ánh xạ chỉ số khớp cặp tương ứng giữa dự đoán và nhãn thực tế.

```python
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    """
    Module Hungarian Matcher thực hiện gán nhãn tối ưu 1-đối-1 giữa các
    đối tượng thực tế và các dự đoán của mô hình (Bipartite Matching).
    """
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        
    @torch.no_grad()
    def forward(self, outputs: dict, targets: list) -> list:
        """
        outputs: Dict chứa:
           - "pred_logits": Tensor [Batch_size, Num_Queries, Num_Classes + 1] (class logits)
           - "pred_boxes": Tensor [Batch_size, Num_Queries, 4] (hộp dạng [cx, cy, w, h])
        targets: List có độ dài Batch_size, mỗi phần tử là một Dict chứa:
           - "labels": Tensor [M] (nhãn lớp thực tế)
           - "boxes": Tensor [M, 4] (tọa độ thực tế dạng [cx, cy, w, h])
           
        Trả về:
           Một list có độ dài Batch_size. Mỗi phần tử là một tuple (index_i, index_j) trong đó:
             - index_i: Tensor chứa chỉ số của dự đoán (predictions) được khớp cặp.
             - index_j: Tensor chứa chỉ số của nhãn gốc (ground truth) tương ứng.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # 1. Gom tất cả dự đoán trong toàn bộ batch lại
        # [B * N, C]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        # [B * N, 4]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        
        # 2. Gom tất cả nhãn thực tế trong batch lại
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # 3. Tính toán chi phí phân lớp (Classification Cost)
        # Tại mỗi nhãn thực tế c, chi phí bằng -xác suất dự đoán tại nhãn đó
        cost_class = -out_prob[:, tgt_ids]
        
        # 4. Tính toán chi phí khoảng cách L1 của tọa độ hộp
        # [B * N, M]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # 5. Tính toán chi phí GIoU
        # Chuyển đổi sang xyxy trước khi tính GIoU
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), 
                                         box_cxcywh_to_xyxy(tgt_bbox))
        
        # 6. Tổng hợp ma trận chi phí hoàn chỉnh
        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        # Định hình lại ma trận chi phí về [Batch_size, Num_Queries, Total_Targets_in_Batch]
        C = C.view(bs, num_queries, -1).cpu()
        
        # Lấy kích thước số lượng đối tượng thực tế trong mỗi ảnh
        sizes = [len(v["labels"]) for v in targets]
        
        # 7. Duyệt qua từng ảnh trong batch để giải thuật toán gán nhãn tối ưu
        indices = []
        for i, (c_slice, size) in enumerate(zip(C.split(sizes, -1), sizes)):
            if size == 0:
                # Nếu ảnh không chứa đối tượng nào, khớp cặp rỗng
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue
                
            # Trích xuất ma trận chi phí của ảnh thứ i: [Num_Queries, size]
            cost_matrix = c_slice[i].numpy()
            
            # Chạy thuật toán Hungary trên CPU
            # out_ind: chỉ số dự đoán được chọn, tgt_ind: chỉ số nhãn gốc tương ứng
            out_ind, tgt_ind = linear_sum_assignment(cost_matrix)
            
            indices.append((
                torch.as_tensor(out_ind, dtype=torch.int64),
                torch.as_tensor(tgt_ind, dtype=torch.int64)
            ))
            
        return indices
```

---

## 5. Chạy Thử Nghiệm Hungarian Matcher (Sanity Check)

Hãy viết một script ngắn kiểm tra tính đúng đắn của việc tìm chỉ số khớp cặp:

```python
if __name__ == '__main__':
    # Giả định tham số
    num_classes = 80 # Ví dụ bộ dữ liệu COCO có 80 lớp
    
    # 1. Giả lập đầu ra mô hình: Batch_size=1, 100 queries
    pred_logits = torch.randn(1, 100, num_classes + 1) # +1 kênh cho class rỗng
    pred_boxes = torch.rand(1, 100, 4) # Tọa độ [cx, cy, w, h] ngẫu nhiên trong [0, 1]
    
    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
    
    # 2. Giả lập nhãn thực tế: Có 3 đối tượng trong ảnh
    targets = [{
        "labels": torch.tensor([3, 17, 52], dtype=torch.long), # Nhãn của 3 vật thể
        "boxes": torch.tensor([
            [0.5, 0.5, 0.2, 0.2],
            [0.3, 0.4, 0.1, 0.3],
            [0.8, 0.2, 0.4, 0.1]
        ], dtype=torch.float32)
    }]
    
    # 3. Khởi tạo Hungarian Matcher
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    
    # Tìm khớp cặp song phương
    indices = matcher(outputs, targets)
    
    # Lấy kết quả ảnh đầu tiên
    pred_idx, tgt_idx = indices[0]
    
    print(f"Chỉ số các dự đoán được chọn từ 100 Queries: {pred_idx.tolist()}")
    print(f"Chỉ số nhãn gốc tương ứng khớp cặp: {tgt_idx.tolist()}")
    
    assert len(pred_idx) == 3, "Lỗi: Số lượng khớp cặp phải bằng số đối tượng thực tế (3)!"
    print("🎉 Thành công! Hungarian Matcher khớp cặp chính xác tỉ lệ 1-đối-1 hoàn hảo!")
```

### Kết quả đầu ra mong đợi:
```text
Chỉ số các dự đoán được chọn từ 100 Queries: [14, 48, 92] (ví dụ ngẫu nhiên)
Chỉ số nhãn gốc tương ứng khớp cặp: [0, 1, 2]
🎉 Thành công! Hungarian Matcher khớp cặp chính xác tỉ lệ 1-đối-1 hoàn hảo!
```

---

> [!TIP]
> Việc chạy thuật toán Hungary trên CPU bằng `scipy` vô cùng nhanh vì số lượng `queries` nhỏ ($N = 100$) và số đối tượng thực tế thường chỉ dưới vài chục vật thể trên mỗi ảnh. Do đó, bước tính toán trung gian này không hề gây nghẽn cổ chai (bottleneck) hiệu năng huấn luyện.

Có được kết quả khớp cặp song phương, làm thế nào để định nghĩa hàm mất mát lan truyền ngược để cập nhật trọng số cho mô hình? 

Hãy cùng bước sang **[Phần 4: Thiết Kế Hàm Mất Mát Set Criterion](part4_loss_criterion.md)**!
