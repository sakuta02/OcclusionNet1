import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from .data import build_loaders, image_paths
from .inference import BundleManifest, OcclusionPredictor, save_bundle
from .metrics import evaluate, grad_norm
from .models import build_classifier, focal_bce_loss
from .tracking import make_tracker
from .visualization import plot_prediction


@dataclass
class TrainingResult:
    best_epoch: int
    best_val_loss: float
    macro_metrics: dict
    per_class_metrics: dict


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_optimizer(model, encoder_lr, head_lr):
    encoder_params = list(model.encoder.parameters())
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    return torch.optim.AdamW(
        [{"params": encoder_params, "lr": encoder_lr}, {"params": head_params, "lr": head_lr}]
    )


def make_scheduler(optimizer, steps_per_epoch, epochs):
    """Эпоха линейного warmup, дальше косинус до нуля."""
    warmup_steps = steps_per_epoch
    total_steps = epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device, freeze_encoder):
    model.train()
    if freeze_encoder:
        model.encoder.eval()  # держим BatchNorm энкодера в eval, статистики не двигаем

    total_loss, norms = 0.0, []
    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), targets)
        loss.backward()
        norms.append(grad_norm(model))
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset), float(np.mean(norms))


def log_inference_samples(predictor, paths, tracker, epoch, threshold):
    for path in paths:
        probabilities, attention = predictor.predict(path, return_attn=True)
        figure = plot_prediction(path, probabilities, attention, threshold)
        tracker.figure("Inference", path.name, figure, epoch)
        figure.clf()


def run_training(config) -> TrainingResult:
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_loader, test_loader = build_loaders(config)
    model = build_classifier(config, device)
    optimizer = make_optimizer(model, config.encoder_lr, config.head_lr)
    scheduler = make_scheduler(optimizer, len(train_loader), config.epochs)

    def loss_fn(logits, targets):
        return focal_bce_loss(logits, targets, gamma=config.focal_gamma)

    tracker = make_tracker(config.project_name, config.task_name)
    tracker.connect({"epochs": config.epochs, "head_lr": config.head_lr, "classes": config.classes})
    inference_paths = image_paths(config.inference_dir) if config.inference_dir else []

    best_val_loss, best_epoch = float("inf"), 0
    macro, per_class = {}, {}
    for epoch in range(1, config.epochs + 1):
        train_loss, mean_grad_norm = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device, config.freeze_encoder
        )
        current_lr = optimizer.param_groups[0]["lr"]
        tracker.scalar("Loss", "train", train_loss, epoch)
        tracker.scalar("Grad norm", "mean", mean_grad_norm, epoch)
        tracker.scalar("LR", "value", current_lr, epoch)

        val_loss, macro, per_class = evaluate(
            model, test_loader, config.classes, device, loss_fn, config.threshold
        )
        tracker.scalar("Loss", "val", val_loss, epoch)
        for name, value in macro.items():
            tracker.scalar(f"{name} macro", "val", value, epoch)
        for class_name, metrics in per_class.items():
            for name, value in metrics.items():
                tracker.scalar(f"{name} per-class", class_name, value, epoch)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss, best_epoch = val_loss, epoch
            manifest = BundleManifest(
                classes=config.classes,
                arch=config.arch,
                encoder_name=config.encoder_name,
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                threshold=config.threshold,
                metrics={"macro": macro, "per_class": per_class, "val_loss": val_loss, "epoch": epoch},
            )
            save_bundle(config.output_dir, manifest, model.state_dict())

        print(
            f"Epoch {epoch}/{config.epochs} | train_loss={train_loss:.4f} | grad_norm={mean_grad_norm:.4f}"
            f" | lr={current_lr:.2e} | val_loss={val_loss:.4f} | val_f1_macro={macro['f1']:.4f}"
            + (" | ★ best" if is_best else "")
        )

        if inference_paths and epoch % config.inference_every_n_epochs == 0:
            predictor = OcclusionPredictor(model, config.classes, config.threshold, device)
            log_inference_samples(predictor, inference_paths, tracker, epoch, config.threshold)
            model.train()

    tracker.close()
    return TrainingResult(best_epoch, best_val_loss, macro, per_class)
