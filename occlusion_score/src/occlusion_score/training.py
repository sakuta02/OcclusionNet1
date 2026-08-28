from dataclasses import dataclass
import numpy as np
import torch
import torchvision.models as tvm
from PIL import Image
from scipy.stats import kendalltau, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

from .data import image_paths, recover_labels
from .features import FEATURE_DIM, build_feature_matrix, cheap_features, extract_classifier_features, extract_dino_features
from .inference import BundleManifest, save_bundle
from .models import ExactClassifier, MobileNetStudent, TinyHead, pairwise_ranking_loss


@dataclass
class TrainingResult:
    status: str
    teacher_metrics: dict
    student_metrics: dict


def acceptance_status(teacher_metrics, student_metrics, tolerance=0.01):
    close_enough_rank = student_metrics["Spearman"] >= teacher_metrics["Spearman"] - tolerance
    close_enough_error = student_metrics["MAE"] <= teacher_metrics["MAE"] + tolerance
    return "ACCEPT" if close_enough_rank and close_enough_error else "REJECT"


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Spearman": spearmanr(y_true, y_pred).statistic,
        "Kendall": kendalltau(y_true, y_pred).statistic,
        "R2": r2_score(y_true, y_pred),
    }


def make_ridge():
    return Pipeline([("scaler", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-3, 3, 30)))])


def run_teacher(X, y, weights, groups, seed=42, n_seeds=5, n_splits=5):
    oof, metrics = np.zeros(len(y)), []
    for offset in range(n_seeds):
        permutation = np.random.RandomState(seed + offset).permutation(len(y))
        shuffled_X, shuffled_y, shuffled_groups = X[permutation], y[permutation], groups[permutation]
        seed_oof = np.zeros(len(y))
        for train, valid in GroupKFold(n_splits=n_splits).split(shuffled_X, shuffled_y, shuffled_groups):
            model = make_ridge()
            model.fit(shuffled_X[train], shuffled_y[train], model__sample_weight=weights[permutation][train])
            seed_oof[valid] = model.predict(shuffled_X[valid])
        seed_oof = seed_oof[np.argsort(permutation)]
        oof += seed_oof
        metrics.append(evaluate(y, seed_oof))
    mean_metrics = {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]}
    return oof / n_seeds, mean_metrics


def train_one_head(X_train, y_train, weights, X_valid, y_valid, device="cpu", epochs=40):
    model = TinyHead(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_X, train_y, train_w = [torch.from_numpy(value).to(device) for value in (X_train, y_train, weights)]
    valid_X = torch.from_numpy(X_valid).to(device)

    best_state, best_mae = None, float("inf")
    for _ in range(epochs):
        model.train()
        for batch in np.array_split(np.random.permutation(len(y_train)), max(1, len(y_train) // 32)):
            index = torch.from_numpy(batch).to(device)
            prediction = model(train_X[index])
            mse = (train_w[index] * (prediction - train_y[index]).square()).mean()
            loss = mse + 0.3 * pairwise_ranking_loss(prediction, train_y[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            prediction = model(valid_X).cpu().numpy()
        score = mean_absolute_error(y_valid, prediction)
        if score < best_mae:
            best_mae = score
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model.cpu()


def run_training(config):
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    paths = image_paths(config.data_root)
    labels = recover_labels(config.labels_root, config.data_root)
    indices = {str(path): i for i, path in enumerate(paths)}
    labeled = np.asarray([indices[path] for path in labels.image_path])
    y = labels.human_score.to_numpy(np.float32)
    weights = labels.weight.to_numpy(np.float32)
    groups = labels.group.to_numpy()

    checkpoint = torch.load(config.weights_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    classes = list(checkpoint.get("classes", sorted({path.parent.name for path in paths if path.parent.name.lower() != "clean"})))
    classifier = ExactClassifier(tvm.efficientnet_b3(weights=None).features, len(classes))
    classifier.load_state_dict(state)
    classifier.to(device).eval()

    clf_probs, clf_semantic = extract_classifier_features(classifier, paths, device)
    cheap = np.stack([cheap_features(path) for path in paths])
    dino, pca = extract_dino_features(paths, device)
    teacher_X = build_feature_matrix(dino, clf_probs, clf_semantic, cheap)

    _, teacher_metrics = run_teacher(teacher_X[labeled], y, weights, groups, config.seed, config.n_seeds, config.n_splits)

    student = MobileNetStudent(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT).to(device)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    distill_data = [(transform(Image.open(path).convert("RGB")), torch.from_numpy(target)) for path, target in zip(paths, dino)]
    loader = torch.utils.data.DataLoader(distill_data, batch_size=64, shuffle=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=1e-4)
    for _ in range(8):
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            prediction = student(images)
            mse = torch.nn.functional.mse_loss(prediction, target)
            cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=-1).mean()
            loss = mse + 0.5 * (1 - cosine)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    student.eval()
    student_X = []
    with torch.inference_mode():
        for start in range(0, len(paths), 64):
            batch_paths = paths[start : start + 64]
            images = torch.stack([transform(Image.open(path).convert("RGB")) for path in batch_paths]).to(device)
            student_X.append(student(images).cpu().numpy())

    production_X = build_feature_matrix(np.concatenate(student_X), clf_probs, clf_semantic, cheap)
    labeled_X = production_X[labeled]
    oof, scalers, head_states = np.zeros(len(y)), [], []
    for seed in range(config.n_seeds):
        permutation = np.random.RandomState(config.seed + 100 + seed).permutation(len(y))
        seed_oof = np.zeros(len(y))
        for train, valid in GroupKFold(n_splits=config.n_splits).split(labeled_X[permutation], y[permutation], groups[permutation]):
            scaler = StandardScaler().fit(labeled_X[permutation][train])
            head = train_one_head(
                scaler.transform(labeled_X[permutation][train]).astype(np.float32),
                y[permutation][train],
                weights[permutation][train],
                scaler.transform(labeled_X[permutation][valid]).astype(np.float32),
                y[permutation][valid],
                device,
            )
            with torch.inference_mode():
                valid_features = torch.from_numpy(scaler.transform(labeled_X[permutation][valid]).astype(np.float32))
                seed_oof[valid] = head(valid_features).numpy()
            scalers.append(scaler)
            head_states.append(head.state_dict())
        oof += seed_oof[np.argsort(permutation)]
    oof /= config.n_seeds

    isotonic = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(oof, y)
    student_metrics = evaluate(y, isotonic.predict(oof))
    status = acceptance_status(teacher_metrics, student_metrics)

    manifest = BundleManifest(
        status=status,
        feature_dim=FEATURE_DIM,
        metrics={"teacher": teacher_metrics, "student": student_metrics},
        classifier_classes=classes,
    )
    sklearn_artifacts = {"scalers": scalers, "isotonic": isotonic, "pca": pca}
    torch_artifacts = {
        "classifier_state": classifier.state_dict(),
        "student_state": student.state_dict(),
        "head_states": head_states,
    }
    save_bundle(config.output_dir, manifest, sklearn_artifacts, torch_artifacts)
    return TrainingResult(status, teacher_metrics, student_metrics)
