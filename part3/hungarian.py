import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    """
    Performs bipartite matching between predictions and ground truth using the Hungarian algorithm.
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        """
        Initialize the HungarianMatcher.
        
        Args:
            cost_class: Weight for classification cost.
            cost_bbox: Weight for L1 (bbox) cost.
            cost_giou: Weight for GIoU cost.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class + cost_bbox + cost_giou > 0

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Calculate classification loss (Cross-Entropy).
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # Classification probabilities
        src_prob = F.softmax(src_logits, -1)
        src_prob = src_prob.view(-1, src_prob.shape[-1])
        
        # Loss
        loss_ce = F.cross_entropy(src_logits[idx], target_classes_o, weight=None)
        losses = {'loss_ce': loss_ce}

        if log:
            # Calculate accuracy
            prob = src_prob[idx].max(-1)[1]
            acc = (prob == target_classes_o).float().mean()
            losses['class_error'] = 1 - acc
        
        return losses, src_prob, target_classes_o

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Calculate bounding box loss (L1 + GIoU).
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][J] for t, (_, J) in zip(targets, indices)], dim=0)

        # L1 Loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}  # We'll add this later
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # GIoU Loss
        loss_giou = 1 - generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), 
                                            box_cxcywh_to_xyxy(target_boxes))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        return losses, src_boxes, target_boxes

    def _get_src_permutation_idx(self, indices):
        """
        Get the flat indices for the source predictions that were matched.
        """
        batch_idx = torch.cat([torch.full_like(J, i) for i, (_, J) in enumerate(indices)])
        src_idx = torch.cat([J for (_, J) in indices])
        return batch_idx, src_idx

    def __call__(self, outputs, targets):
        """
        Perform Hungarian matching.
        
        Args:
            outputs: Dictionary from the DETR model containing 'pred_logits' and 'pred_boxes'.
            targets: List of ground truth dictionaries.
            
        Returns:
            A tuple containing:
            - losses: Dictionary of losses.
            - indices: Tuple of matched indices ((batch_idx, src_idx), (batch_idx, tgt_idx)).
            - matched_obj_probs: Probabilities of matched objects.
            - matched_tgt_classes: Ground truth classes of matched objects.
        """
        with torch.no_grad():
            # 1. Calculate Cost Matrix
            C = self.calculate_cost_matrix(outputs, targets)

            # 2. Apply Hungarian Algorithm
            # C shape is [Batch_size, num_queries, num_gt]
            # We apply it to each image in the batch independently
            C_bsz = C.shape[0]
            C_ = C.cpu() # linear_sum_assignment only works on CPU
            
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_)]
            # indices is a list of tuples: [((row_idx, col_idx), (row_idx, col_idx)), ...]
            # We need to convert this to the format used by loss functions
            
            # Format: ((batch_idx, src_idx), (batch_idx, tgt_idx))
            # src_idx corresponds to queries, tgt_idx corresponds to ground truth
            bs_idx = torch.arange(C_bsz, device=C.device)
            
            # Extract source and target indices
            # src_idx_matched: Flat indices of queries that were matched
            src_idx_matched = torch.cat([ind[0] for ind in indices])
            # tgt_idx_matched: Flat indices of ground truths that were matched
            tgt_idx_matched = torch.cat([ind[1] for ind in indices])
            
            # Batch indices for source and target
            bs_idx_src = torch.cat([bs_idx.unsqueeze(1).expand_as(ind[0]) for ind in indices], dim=0)
            bs_idx_tgt = torch.cat([bs_idx.unsqueeze(1).expand_as(ind[1]) for ind in indices], dim=0)
            
            # Final indices tuple
            indices_out = ((bs_idx_src, src_idx_matched), (bs_idx_tgt, tgt_idx_matched))

            # 3. Calculate Losses
            # Get predicted object probabilities for the matched queries
            pred_logits = outputs['pred_logits']
            idx = self._get_src_permutation_idx(indices_out)
            matched_obj_probs = F.softmax(pred_logits, dim=-1)[idx]
            
            # Get ground truth classes for the matched targets
            matched_tgt_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices_out[1])])
            
            # Loss calculations
            losses, _, _ = self.loss_labels(outputs, targets, indices_out, None, log=False)
            losses.update(self.loss_boxes(outputs, targets, indices_out, None))
            
            # 4. Compute Total Loss
            # Cost is negative log probability of the correct class + weighted costs for bbox and GIoU
            # Note: loss_ce is positive, so we subtract it to convert from log-likelihood to cost
            # This is because we want to minimize cost, and maximizing likelihood means minimizing negative log-likelihood
            cls_cost = -matched_obj_probs[:, matched_tgt_classes]
            
            total_cost = (self.cost_bbox * losses['loss_bbox'] + 
                          self.cost_giou * losses['loss_giou'] + 
                          self.cost_class * cls_cost)
            
            # Reshape total_cost to match the format of C for consistency (optional, but good for debugging)
            # Actually, let's keep it as a flat tensor for the loss function
            
            return losses, indices_out, matched_obj_probs, matched_tgt_classes, total_cost

    def calculate_cost_matrix(self, outputs, targets):
        """
        Calculate the cost matrix between predictions and ground truth.
        
        Cost = cost_class * (-log(p(correct_class))) + 
               cost_bbox * L1_loss + 
               cost_giou * (1 - GIoU)
        
        Args:
            outputs: Model predictions
