import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def grad_norm(model) -> float:
    total = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None)
    return total**0.5


def per_class_metrics(targets, predictions, classes) -> dict[str, dict[str, float]]:
    result = {}
    for i, class_name in enumerate(classes):
        y_true, y_pred = targets[:, i], predictions[:, i]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        result[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }
    return result


def macro_average(per_class) -> dict[str, float]:
    keys = ("precision", "recall", "f1", "accuracy")
    return {key: float(np.mean([item[key] for item in per_class.values()])) for key in keys}


@torch.inference_mode()
def evaluate(model, loader, classes, device, loss_fn, threshold=0.5):
    model.eval()
    logits_batches, target_batches, running_loss = [], [], 0.0
    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        running_loss += loss_fn(logits, targets).item() * images.size(0)
        logits_batches.append(logits.cpu())
        target_batches.append(targets.cpu())

    logits = torch.cat(logits_batches)
    targets = torch.cat(target_batches).numpy()
    predictions = (torch.sigmoid(logits) > threshold).float().numpy()

    per_class = per_class_metrics(targets, predictions, classes)
    return running_loss / len(loader.dataset), macro_average(per_class), per_class
