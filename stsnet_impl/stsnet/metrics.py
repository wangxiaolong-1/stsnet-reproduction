"""Segmentation metrics (paper Eq. 13-18): IoU, F1, OA, mIoU."""
import torch


@torch.no_grad()
def confusion_matrix(pred, target, num_classes):
    """pred/target: (B, H, W) integer class maps -> (K, K) matrix."""
    mask = (target >= 0) & (target < num_classes)
    hist = torch.bincount(
        num_classes * target[mask].flatten() + pred[mask].flatten(),
        minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


class SegMetrics:
    """Accumulates a confusion matrix and reports OA / mIoU / per-class IoU & F1."""

    def __init__(self, num_classes, class_names=None):
        self.K = num_classes
        self.names = class_names
        self.reset()

    def reset(self):
        self.hist = torch.zeros(self.K, self.K, dtype=torch.long)

    @torch.no_grad()
    def update(self, logits, target):
        pred = logits.argmax(1)
        self.hist += confusion_matrix(pred, target, self.K).to(self.hist.device)

    def iou(self):
        tp = self.hist.diag().float()
        return tp / (self.hist.sum(1) + self.hist.sum(0) - tp).clamp(min=1)

    def f1(self):
        tp = self.hist.diag().float()
        prec = tp / self.hist.sum(0).clamp(min=1)
        rec = tp / self.hist.sum(1).clamp(min=1)
        return 2 * prec * rec / (prec + rec).clamp(min=1e-8)

    def summary(self):
        iou, f1 = self.iou(), self.f1()
        oa = self.hist.diag().sum().float() / self.hist.sum().clamp(min=1)
        miou = iou.mean()
        rows = []
        for i in range(self.K):
            name = self.names[i] if self.names else f"class_{i}"
            rows.append(f"  {name:<12s} IoU {iou[i] * 100:6.2f}  F1 {f1[i] * 100:6.2f}")
        return "\n".join([
            *rows,
            f"  OA   {oa.item() * 100:6.2f}   mIoU {miou.item() * 100:6.2f}",
        ]), {"oa": oa.item(), "miou": miou.item(),
             "iou": iou.tolist(), "f1": f1.tolist()}
