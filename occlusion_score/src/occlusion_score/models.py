import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tvm


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

    def forward(self, q, tokens):
        attention, _ = self.cross_attn(q, tokens, tokens)
        q = self.norm1(q + attention)
        attention, _ = self.self_attn(q, q, q)
        q = self.norm2(q + attention)
        return self.norm3(q + self.ffn(q))


class ExactClassifier(nn.Module):
    def __init__(self, features, n_classes):
        super().__init__()
        self.encoder_holder = nn.Module()
        self.encoder_holder.features = features
        self.input_proj = nn.Conv2d(1536, 128, 1)
        self.class_queries = nn.Parameter(torch.randn(n_classes, 128) * 0.02)
        self.layers = nn.ModuleList([QueryAttentionBlock(), QueryAttentionBlock()])
        self.classifier = nn.Linear(128, 1)

    def forward_features(self, x):
        feature_map = self.input_proj(self.encoder_holder.features(x))
        batch = feature_map.shape[0]
        tokens = feature_map.flatten(2).permute(0, 2, 1)
        queries = self.class_queries.unsqueeze(0).expand(batch, -1, -1)
        for layer in self.layers:
            queries = layer(queries, tokens)
        return self.classifier(queries).squeeze(-1), queries.mean(dim=1)

    def forward(self, x):
        return self.forward_features(x)[0]


class MobileNetStudent(nn.Module):
    def __init__(self, out_dim=64, weights=None):
        super().__init__()
        backbone = tvm.mobilenet_v3_small(weights=weights)
        self.features, self.avgpool = backbone.features, backbone.avgpool
        self.proj = nn.Sequential(
            nn.Linear(backbone.classifier[0].in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.proj(self.avgpool(self.features(x)).flatten(1))


class TinyHead(nn.Module):
    def __init__(self, in_dim=209, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pairwise_ranking_loss(pred, target, margin=0.05):
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    mask = target_diff.abs() > margin
    if not mask.any():
        return pred.new_tensor(0.0)
    return F.relu(margin - (pred.unsqueeze(1) - pred.unsqueeze(0)) * target_diff.sign())[mask].mean()
