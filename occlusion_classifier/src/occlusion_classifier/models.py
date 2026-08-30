import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn

from .exceptions import CheckpointLoadError


class QueryAttentionBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, queries, tokens, return_attn=False):
        attention, weights = self.cross_attn(
            queries, tokens, tokens, need_weights=return_attn, average_attn_weights=True
        )
        queries = self.norm1(queries + attention)
        attention, _ = self.self_attn(queries, queries, queries)
        queries = self.norm2(queries + attention)
        queries = self.norm3(queries + self.ffn(queries))
        return (queries, weights) if return_attn else queries


class OcclusionClassifier(nn.Module):
    """Энкодер сегментационной модели + обучаемый query на класс.
    Каждый query собирает признаки кросс-аттеншеном по карте энкодера,
    общая линейная голова превращает его в логит — вектор длины num_classes."""

    def __init__(self, encoder, num_classes=7, embed_dim=128, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.input_proj = nn.Conv2d(encoder.out_channels[-1], embed_dim, kernel_size=1)
        self.class_queries = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.layers = nn.ModuleList(
            [QueryAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x, return_attn=False):
        feature_map = self.input_proj(self.encoder(x)[-1])
        batch, _, height, width = feature_map.shape
        tokens = feature_map.flatten(2).permute(0, 2, 1)
        queries = self.class_queries.unsqueeze(0).expand(batch, -1, -1)

        last_attn = None
        for layer in self.layers:
            if return_attn:
                queries, last_attn = layer(queries, tokens, return_attn=True)
            else:
                queries = layer(queries, tokens)

        logits = self.classifier(queries).squeeze(-1)
        if return_attn:
            return logits, last_attn.reshape(batch, -1, height, width)
        return logits


def build_encoder(arch, encoder_name):
    """Пустой граф энкодера — под него потом грузится state_dict."""
    return getattr(smp, arch)(encoder_name=encoder_name, encoder_weights=None).encoder


def load_segmentation_encoder(arch, encoder_name, checkpoint_path):
    """Достаёт энкодер из чекпойнта Lightning-обёртки сегментационной модели
    (ключи в нём лежат с префиксом `model.`)."""
    seg_model = getattr(smp, arch)(encoder_name=encoder_name, encoder_weights=None)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("state_dict", checkpoint)
    state_dict = {key[len("model.") :]: value for key, value in raw.items() if key.startswith("model.")}
    if not state_dict:
        raise CheckpointLoadError(f"В {checkpoint_path} нет весов с префиксом 'model.'")

    missing, _ = seg_model.load_state_dict(state_dict, strict=False)
    if [key for key in missing if key.startswith("encoder.")]:
        raise CheckpointLoadError(f"В {checkpoint_path} не хватает весов энкодера")
    return seg_model.encoder


def build_classifier(config, device="cpu") -> OcclusionClassifier:
    encoder = load_segmentation_encoder(config.arch, config.encoder_name, config.encoder_checkpoint)
    if config.freeze_encoder:
        for parameter in encoder.parameters():
            parameter.requires_grad = False

    model = OcclusionClassifier(
        encoder,
        num_classes=len(config.classes),
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    return model.to(device)


def focal_bce_loss(logits, targets, gamma=2.0, alpha=None, reduction="mean"):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)  # эквивалент p при target=1 и (1-p) при target=0
    loss = (1 - p_t) ** gamma * bce
    if alpha is not None:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
