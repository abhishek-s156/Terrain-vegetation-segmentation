import torch


def pixel_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / max(total, 1)


def mean_iou(logits, targets, num_classes=7):
    preds = torch.argmax(logits, dim=1)
    ious = []

    for cls in range(num_classes):
        pred_c = preds == cls
        target_c = targets == cls

        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()

        if union > 0:
            ious.append(intersection / union)

    return sum(ious) / max(len(ious), 1)
