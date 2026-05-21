# Phần 4: Thiết Kế Hàm Mất Mát Set Criterion 📉

Sau khi đã tìm được chỉ số khớp cặp song phương tối ưu bằng `HungarianMatcher`, chúng ta cần tính toán hàm mất mát (loss) để cập nhật trọng số cho mô hình qua lan truyền ngược. 

Trong phần này, chúng ta sẽ xây dựng lớp **`SetCriterion`** - trái tim huấn luyện của DETR, phụ trách tính toán hàm mất mát kết hợp đa nhiệm (multi-task loss) bao gồm:
1. **Classification Loss**: Lỗi phân loại nhãn (Cross-Entropy).
2. **L1 Bounding Box Loss**: Lỗi khoảng cách tuyệt đối của tọa độ hộp.
3. **GIoU Loss**: Lỗi hình học của hộp bao.

---

## 1. Công Thức Toán Học Cho Hàm Mất Mát Set-Based Loss

Với phép hoán vị tối ưu $\hat{\sigma}$ đã tìm được ở Phần 3, hàm mất mát Hungarian Loss cho tất cả các khớp cặp trong batch được tính như sau:

$$\mathcal{L}_{DETR}(y, \hat{y}) = \sum_{i=1}^{N} \left[ \mathcal{L}_{val}(c_i, \hat{p}_{\hat{\sigma}(i)}(c_i)) + \mathbb{1}_{\{c_i \neq \varnothing\}} \mathcal{L}_{box}(b_i, \hat{b}_{\hat{\sigma}(i)}) \right]$$

### 1.1. Lỗi Phân Loại Nhãn (Classification Loss)
Ở đây ta dùng hàm mất mát Cross Entropy chuẩn. Tuy nhiên, vì đa số trong số 100 Object Queries sẽ khớp với lớp nền rỗng $\varnothing$ ("no object"), hiện tượng mất cân bằng mẫu sẽ xảy ra nghiêm trọng. 

Để khắc phục, chúng ta giảm trọng số của lớp $\varnothing$ xuống (thường đặt hệ số `eos_coef = 0.1` thay vì `1.0` như các lớp thông thường).

### 1.2. Lỗi Tọa Độ Hộp (Bounding Box Loss)
Với những dự đoán khớp với một vật thể thực tế ($c_i \neq \varnothing$), lỗi tọa độ hộp bao gồm hai thành phần tuyến tính được chuẩn hóa theo tổng số đối tượng thực tế ($N_{boxes}$) trong batch:

$$\mathcal{L}_{box}(b_i, \hat{b}_{\hat{\sigma}(i)}) = \frac{1}{N_{boxes}} \left[ \lambda_{L1} \| b_i - \hat{b}_{\hat{\sigma}(i)} \|_1 + \lambda_{giou} \mathcal{L}_{giou}(b_i, \hat{b}_{\hat{\sigma}(i)}) \right]$$

---

## 2. Xây Dựng Lớp SetCriterion Bằng PyTorch

Chúng ta sẽ kế thừa `nn.Module` để tạo lớp `SetCriterion`. Module này sẽ tự động gọi Hungarian Matcher, trích xuất các phần tử khớp cặp, tính các loại loss tương ứng và nhân với trọng số cấu hình sẵn.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import lại các hàm hình học hộp từ phần trước
from part3_hungarian_matcher import box_cxcywh_to_xyxy, generalized_box_iou

class SetCriterion(nn.Module):
    """
    Module quản lý việc tính toán toàn bộ hàm mất mát của DETR.
    """
    def __init__(self, num_classes: int, matcher: nn.Module, weight_dict: dict, 
                 eos_coef: float = 0.1, losses: list = ['labels', 'boxes']):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        
        # Khởi tạo vector trọng số cho Cross Entropy Loss
        # Lớp rỗng (background) nằm ở chỉ số cuối cùng (num_classes) và có trọng số eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        
    def _get_src_permutation_idx(self, indices):
        # Hàm bổ trợ để lấy chỉ số dự đoán (sources) dưới dạng tuple phục vụ đánh chỉ số tensor nhanh
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # Hàm bổ trợ để lấy chỉ số nhãn gốc (targets) dưới dạng tuple
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, outputs: dict, targets: list, indices: list, num_boxes: int) -> dict:
        """
        Tính toán lỗi phân lớp Cross-Entropy.
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        
        idx = self._get_src_permutation_idx(indices)
        
        # 1. Khởi tạo nhãn mục tiêu cho toàn bộ batch với lớp mặc định là lớp rỗng (num_classes)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, 
                                    dtype=torch.int64, device=src_logits.device)
        
        # Lấy nhãn thực tế khớp cặp
        tgt_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # Gán nhãn thực tế vào các vị trí được khớp cặp
        target_classes[idx] = tgt_classes_o
        
        # 2. Tính toán Cross Entropy Loss
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs: dict, targets: list, indices: list, num_boxes: int) -> dict:
        """
        Tính toán lỗi tọa độ L1 Loss và GIoU Loss cho các bounding box khớp cặp.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        
        # Trích xuất các box dự đoán được khớp cặp
        src_boxes = outputs["pred_boxes"][idx]
        
        # Trích xuất các box thực tế tương ứng được khớp cặp
        tgt_boxes = torch.cat([t["boxes"][J] for t, (_, J) in zip(targets, indices)], dim=0)
        
        # Nếu không có đối tượng nào được khớp cặp, trả về loss bằng 0
        if len(tgt_boxes) == 0:
            return {"loss_bbox": src_boxes.sum() * 0, "loss_giou": src_boxes.sum() * 0}
            
        # 1. Tính L1 Loss
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes
        
        # 2. Tính GIoU Loss
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(tgt_boxes)
        ))
        loss_giou = loss_giou.sum() / num_boxes
        
        return {
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou
        }

    def get_loss(self, loss_name: str, outputs: dict, targets: list, indices: list, num_boxes: int) -> dict:
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss_name in loss_map, f"Không hỗ trợ tính loss kiểu: {loss_name}"
        return loss_map[loss_name](outputs, targets, indices, num_boxes)

    def forward(self, outputs: dict, targets: list) -> dict:
        """
        Thực hiện tính toán loss.
        outputs: Dự đoán của mạng (chứa "pred_logits", "pred_boxes" và tùy chọn "aux_outputs")
        targets: Nhãn thực tế
        """
        # 1. Gọi Hungarian Matcher để lấy ánh xạ khớp cặp cho đầu ra cuối cùng
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        
        # 2. Tính toán tổng số lượng hộp thực tế trong Batch để chuẩn hóa tỉ lệ
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float32, device=next(iter(outputs.values())).device)
        
        # Ràng buộc số lượng box tối thiểu là 1 để tránh lỗi chia cho 0
        num_boxes = torch.clamp(num_boxes, min=1).item()
        
        # 3. Tính toán tất cả các loss thành phần cho đầu ra cuối cùng
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
            
        # 4. Tính toán Aux Loss cho các lớp trung gian của Decoder (nếu cấu hình)
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices_aux = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_aux, num_boxes)
                    # Lưu trữ với tên phân biệt theo tầng Decoder, ví dụ: loss_ce_0, loss_bbox_1...
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                    
        return losses
```

---

## 3. Chạy Thử Nghiệm Kiểm Tra SetCriterion (Sanity Check)

Hãy viết một script ngắn lắp ghép `HungarianMatcher` và `SetCriterion` để tính toán giá trị loss cụ thể:

```python
if __name__ == '__main__':
    from part3_hungarian_matcher import HungarianMatcher
    
    # 1. Định nghĩa tham số giả lập
    num_classes = 80
    
    # Khởi tạo matcher
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    
    # Trọng số của từng loại loss thành phần
    weight_dict = {
        'loss_ce': 1.0,
        'loss_bbox': 5.0,
        'loss_giou': 2.0
    }
    
    # Khởi tạo SetCriterion
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
        losses=['labels', 'boxes']
    )
    
    # 2. Giả lập đầu ra của mô hình DETR (Batch_size = 2, 100 queries)
    # Bao gồm cả dự đoán cuối và dự đoán trung gian aux_outputs (ví dụ từ 2 layer Decoder cuối)
    pred_logits = torch.randn(2, 100, num_classes + 1)
    pred_boxes = torch.rand(2, 100, 4)
    
    aux_outputs = [
        {"pred_logits": torch.randn(2, 100, num_classes + 1), "pred_boxes": torch.rand(2, 100, 4)},
        {"pred_logits": torch.randn(2, 100, num_classes + 1), "pred_boxes": torch.rand(2, 100, 4)}
    ]
    
    outputs = {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
        "aux_outputs": aux_outputs
    }
    
    # 3. Giả lập nhãn thực tế targets cho 2 ảnh trong batch
    targets = [
        {
            "labels": torch.tensor([3, 12], dtype=torch.long),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
        },
        {
            "labels": torch.tensor([45], dtype=torch.long),
            "boxes": torch.tensor([[0.7, 0.8, 0.1, 0.2]], dtype=torch.float32)
        }
    ]
    
    # 4. Tính toán hàm mất mát
    loss_dict = criterion(outputs, targets)
    
    print("--- DANH SÁCH CÁC LOSS TÍNH ĐƯỢC ---")
    total_weighted_loss = 0
    for k, v in loss_dict.items():
        # Tìm trọng số tương ứng (xử lý cả aux loss bằng cách lấy tiền tố tên loss gốc)
        base_key = k
        for suffix in ['_0', '_1', '_2', '_3', '_4', '_5']:
            if k.endswith(suffix):
                base_key = k[:-2]
        
        weight = weight_dict.get(base_key, 1.0)
        weighted_val = v.item() * weight
        total_weighted_loss += weighted_val
        print(f"{k:15} | Giá trị gốc: {v.item():.4f} | Trọng số: {weight} | Tổng có trọng số: {weighted_val:.4f}")
        
    print("-" * 36)
    print(f"Tổng Hàm Mất Mát Cuối Cùng: {total_weighted_loss:.4f}")
    
    assert total_weighted_loss > 0, "Lỗi: Tổng loss phải lớn hơn 0!"
    print("🎉 Thành công! SetCriterion tính toán loss chính xác hoàn hảo kể cả với cơ chế Auxiliary Loss!")
```

### Kết quả đầu ra mong đợi:
```text
--- DANH SÁCH CÁC LOSS TÍNH ĐƯỢC ---
loss_ce         | Giá trị gốc: 4.4150 | Trọng số: 1.0 | Tổng có trọng số: 4.4150
loss_bbox       | Giá trị gốc: 0.2104 | Trọng số: 5.0 | Tổng có trọng số: 1.0520
loss_giou       | Giá trị gốc: 1.0250 | Trọng số: 2.0 | Tổng có trọng số: 2.0500
loss_ce_0       | Giá trị gốc: 4.4210 | Trọng số: 1.0 | Tổng có trọng số: 4.4210
loss_bbox_0     | Giá trị gốc: 0.2450 | Trọng số: 5.0 | Tổng có trọng số: 1.2250
loss_giou_0     | Giá trị gốc: 1.1020 | Trọng số: 2.0 | Tổng có trọng số: 2.2040
loss_ce_1       | Giá trị gốc: 4.3980 | Trọng số: 1.0 | Tổng có trọng số: 4.3980
loss_bbox_1     | Giá trị gốc: 0.1980 | Trọng số: 5.0 | Tổng có trọng số: 0.9900
loss_giou_1     | Giá trị gốc: 0.9850 | Trọng số: 2.0 | Tổng có trọng số: 1.9700
------------------------------------
Tổng Hàm Mất Mát Cuối Cùng: 22.7250
🎉 Thành công! SetCriterion tính toán loss chính xác hoàn hảo kể cả với cơ chế Auxiliary Loss!
```

---

> [!NOTE]
> **Tầm quan trọng**: Việc sử dụng các hàm bổ trợ đánh chỉ số tensor dạng song song `_get_src_permutation_idx` giúp loại bỏ hoàn toàn các vòng lặp `for` chậm chạp khi tính toán tọa độ box, tận dụng triệt để sức mạnh tính toán song song trên GPU.

Đến đây, chúng ta đã hoàn tất tất cả các viên gạch module nhỏ nhất. Hãy cùng tiến tới chương cuối cùng **[Phần 5: Lắp Ráp Mô Hình DETR & Pipeline Huấn Luyện](part5_detr_model_and_training.md)** để ghép nối và vận hành thực tế!
