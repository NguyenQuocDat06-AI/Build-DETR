import torch

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Change fomat bouding box [center_x, center_y, width, height]
    to [x_min, y_min, x_max, y_max].
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]

    return torch.stack(b, dim=-1)

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Cal IoU between two set of bounding boxes
    boxes1: [N, 4]
    boxes2: [M, 4]
    return:
    IOU: [N, M]
    UNION: [N, M]
    """
    # Cal area
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Find coord of intersection
    # [N,1,2] vs [1,M,2] -> [N,M,2]
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # Left-top
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # Right-bottom

    # Size of intersection
    wh = (rb - lt).clamp(min=0) # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1] # [N,M]

    # Union
    union = area1[:, None] + area2 - inter # [N,M]
    iou = inter / (union + 1e-6) # [N,M]
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Cal GIoU between two set of bounding boxes
    boxes1: [N, 4]
    boxes2: [M, 4]
    return:
    GIoU: [N, M]
    """
    # Cal IoU and Union
    iou, union = box_iou(boxes1, boxes2)

    # Find coord of smallest enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2]) # Left-top
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:]) # Right-bottom

    # Size of enclosing box
    wh = (rb - lt).clamp(min=0) # [N,M,2]
    area = wh[:,:,0] * wh[:,:,1] # [N,M]

    giou = iou - (area - union) / (area + 1e-6) # [N,M]
    return giou 
    