import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.gridspec import GridSpec

DETECTED_COLOR = "#d84b30"
MUTED_COLOR = "#888780"


def plot_prediction(image_path, probabilities, attention, threshold=0.5, max_heatmaps=3):
    """Кадр, столбики вероятностей по классам и тепловые карты внимания
    для сработавших классов (или для топ-1, если не сработал никто)."""
    image = Image.open(image_path).convert("RGB")
    classes = list(probabilities)
    values = [probabilities[name] for name in classes]

    detected = [name for name in classes if probabilities[name] > threshold]
    ranked = sorted(classes, key=lambda name: -probabilities[name])
    heat_classes = (detected or ranked[:1])[:max_heatmaps]

    columns = max(2, len(heat_classes))
    figure = plt.figure(figsize=(4 * columns, 7))
    grid = GridSpec(2, columns, figure=figure)

    axis = figure.add_subplot(grid[0, 0])
    axis.imshow(image)
    axis.axis("off")
    axis.set_title(", ".join(detected) if detected else "Clean", fontsize=10)

    axis = figure.add_subplot(grid[0, 1:])
    axis.barh(classes, values, color=[DETECTED_COLOR if v > threshold else MUTED_COLOR for v in values])
    axis.axvline(threshold, color="gray", linestyle="--", linewidth=1)
    axis.set_xlim(0, 1)
    axis.invert_yaxis()

    image_array = np.array(image)
    for column, class_name in enumerate(heat_classes):
        heat = attention[classes.index(class_name)]
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        axis = figure.add_subplot(grid[1, column])
        axis.imshow(image_array)
        axis.imshow(heat, cmap="jet", alpha=0.45)
        axis.axis("off")
        axis.set_title(f"{class_name} ({probabilities[class_name]:.2f})", fontsize=9)

    figure.tight_layout()
    return figure
