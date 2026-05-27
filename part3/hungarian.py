import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Change bounding box từ fomat [center_x, center_y, width, height]
    to [x_min, y_min, x_max, y_max].
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Calculate IoU between two groups of bounding boxes.
    boxes1: [N, 4] (format xyxy)
    boxes2: [M, 4] (format xyxy)
    
    Returns:
      iou: Matrix [N, M] contains the IoU index between pairs.
      union: Matrix [N, M] contains the area of union.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Find the coordinates of the intersection region
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
    Calculate GIoU between two groups of bounding boxes.
    boxes1: [N, 4] (format xyxy)
    boxes2: [M, 4] (format xyxy)
    
    Returns: GIoU Matrix [N, M]
    """
    # Ensure the input is in xyxy format
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    iou, union = box_iou(boxes1, boxes2)
    
    # Find the coordinates of the smallest enclosing box C that contains both boxes
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    area_c = wh[:, :, 0] * wh[:, :, 1]  # Area of C

    return iou - (area_c - union) / (area_c + 1e-6)

class HungarianMatcher(nn.Module):
    """
    The Hungarian Matcher module performs optimal 1-to-1 assignment between
    ground truth objects and model predictions (Bipartite Matching).
    """
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        
    @torch.no_grad()
    def forward(self, outputs: dict, targets: list) -> list:
        """
        outputs: Dict contains:
           - "pred_logits": Tensor [Batch_size, Num_Queries, Num_Classes + 1] (class logits)
           - "pred_boxes": Tensor [Batch_size, Num_Queries, 4] (hộp dạng [cx, cy, w, h])
        targets: List with length Batch_size, each element is a Dict contains:
           - "labels": Tensor [M] (ground truth class labels)
           - "boxes": Tensor [M, 4] (ground truth coordinates in [cx, cy, w, h] format)
           
        Returns:
           A list with length Batch_size. Each element is a tuple (index_i, index_j) where:
             - index_i: Tensor contains the indices of the predictions matched.
             - index_j: Tensor contains the indices of the corresponding ground truth labels.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # 1. Flatten all predictions in the batch
        # [B * N, C]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        # [B * N, 4]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        
        # 2. Flatten all ground truth labels in the batch
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # 3. Calculate Classification Cost
        # At each ground truth label c, the cost is -the probability of predicting that label
        cost_class = -out_prob[:, tgt_ids]
        
        # 4. Calculate L1 distance cost for box coordinates
        # [B * N, M]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # 5. Calculate GIoU cost
        # Convert to xyxy before calculating GIoU
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), 
                                         box_cxcywh_to_xyxy(tgt_bbox))
        
        # 6. Combine the complete cost matrix
        C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        # Reshape the cost matrix to [Batch_size, Num_Queries, Total_Targets_in_Batch]
        C = C.view(bs, num_queries, -1).cpu()
        
        # Get the number of actual objects in each image
        sizes = [len(v["labels"]) for v in targets]
        
        # 7. Iterate through each image in the batch to solve the optimal assignment algorithm
        indices = []
        for i, (c_slice, size) in enumerate(zip(C.split(sizes, -1), sizes)):
            if size == 0:
                # If the image contains no objects, the matching pair is empty
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue
                
            # Extract the cost matrix of the i-th image: [Num_Queries, size]
            cost_matrix = c_slice[i].numpy()
            
            # Run the Hungarian algorithm on the CPU
            # out_ind: indices of predictions selected, tgt_ind: indices of ground truth labels corresponding
            out_ind, tgt_ind = linear_sum_assignment(cost_matrix)
            
            indices.append((
                torch.as_tensor(out_ind, dtype=torch.int64),
                torch.as_tensor(tgt_ind, dtype=torch.int64)
            ))
            
        return indices

if __name__ == '__main__':
    # Assume parameters
    num_classes = 80 # For example, COCO dataset has 80 classes
    
    # 1. Assume model output: Batch_size=1, 100 queries
    pred_logits = torch.randn(1, 100, num_classes + 1) # +1 channel for empty class
    pred_boxes = torch.rand(1, 100, 4) # Random coordinates [cx, cy, w, h] in [0, 1]
    
    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
    
    # 2. Assume ground truth labels: 3 objects in the image
    targets = [{
        "labels": torch.tensor([3, 17, 52], dtype=torch.long), # Nhãn của 3 vật thể
        "boxes": torch.tensor([
            [0.5, 0.5, 0.2, 0.2],
            [0.3, 0.4, 0.1, 0.3],
            [0.8, 0.2, 0.4, 0.1]
        ], dtype=torch.float32)
    }]
    
    # 3. Initialize Hungarian Matcher
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    
    # Find bipartite matching
    indices = matcher(outputs, targets)
    
    # Get the first image result
    pred_idx, tgt_idx = indices[0]
    
    print(f"Indices of predictions selected from 100 Queries: {pred_idx.tolist()}")
    print(f"Indices of ground truth labels corresponding matched: {tgt_idx.tolist()}")
    
    assert len(pred_idx) == 3, "Error: The number of matches must be equal to the number of actual objects (3)!"
    print("Success! Hungarian Matcher performs perfect 1-to-1 matching!")
