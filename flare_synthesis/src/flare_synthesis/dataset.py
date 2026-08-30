import glob
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.distributions import Normal

from .exceptions import FlareSynthesisError

IMAGE_EXTENSIONS = ("png", "jpeg", "jpg", "bmp", "tif")
EPSILON = 1e-7


class RandomGammaCorrection:
    def __init__(self, gamma=None):
        self.gamma = gamma

    def __call__(self, image):
        if self.gamma is None:
            return TF.adjust_gamma(image, random.choice([0.5, 1, 2]), gain=1)
        if isinstance(self.gamma, tuple):
            return TF.adjust_gamma(image, random.uniform(*self.gamma), gain=1)
        if self.gamma == 0:
            return image
        return TF.adjust_gamma(image, self.gamma, gain=1)


def remove_background(image):
    """Растягивает канал в полный диапазон, чтобы от снимка блика остался
    только сам блик. Чистый torch — работает и на CPU, и на GPU-тензорах."""
    rgb_max = image.amax(dim=(0, 1), keepdim=True)
    rgb_min = image.amin(dim=(0, 1), keepdim=True)
    return (image - rgb_min) * rgb_max / (rgb_max - rgb_min + EPSILON)


def list_images(directory) -> list[str]:
    return sorted(sum((glob.glob(f"{directory}/*.{ext}") for ext in IMAGE_EXTENSIONS), []))


class FlareCompositeDataset(data.Dataset):
    """Урезанная версия `Paired_Flare_Image_Loader` из FlareX: без depth-aware
    веток и без пар lq/gt из внешнего датасета — только случайный синтез блика
    поверх фонового кадра, ровно то, что нужно для генерации обучающих данных.

    Каждый элемент — словарь с `lq` (кадр с бликом), `gt` (кадр только с
    источником света) и `flare` (сам блик). В датасет классификатора идёт `lq`.
    """

    def __init__(self, background_dir, flare_dir, light_dir, img_size=720, device="cpu"):
        self.data_list = list_images(background_dir)
        if not self.data_list:
            raise FlareSynthesisError(f"Нет фоновых кадров в {background_dir}")

        self.flare_list = list_images(flare_dir)
        self.light_list = list_images(light_dir)
        if not self.flare_list:
            raise FlareSynthesisError(f"Нет изображений бликов в {flare_dir}")
        if len(self.flare_list) != len(self.light_list):
            raise FlareSynthesisError("Число flare- и light-изображений не совпадает")

        self.img_size, self.device = img_size, device
        self.to_tensor = transforms.ToTensor()
        self.transform_base = transforms.Compose(
            [
                transforms.RandomCrop((img_size, img_size), pad_if_needed=True, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                # без vertical flip: кадр с дороги перевернулся бы вверх ногами
            ]
        )
        self.transform_flare = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=(0, 0), scale=(0.8, 1.5), translate=(300 / 1440, 300 / 1440), shear=(-20, 20)
                ),
                transforms.CenterCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
        self.blur_transform = transforms.GaussianBlur(21, sigma=(0.1, 3.0))

    def __len__(self):
        return len(self.data_list)

    def _load(self, path, adjust_gamma):
        image = self.to_tensor(Image.open(path).convert("RGB")).to(self.device, non_blocking=True)
        return adjust_gamma(image)

    @torch.no_grad()
    def __getitem__(self, index):
        gamma = np.random.uniform(1.8, 2.2)
        adjust_gamma = RandomGammaCorrection(gamma)
        adjust_gamma_reverse = RandomGammaCorrection(1 / gamma)

        # декодирование остаётся на CPU (PIL), дальнейшая математика идёт на device
        base = self.transform_base(self._load(self.data_list[index], adjust_gamma))
        base = Normal(base, 0.01 * np.random.chisquare(df=1)).sample()
        base = torch.clamp(np.random.uniform(0.5, 1.2) * base, min=0, max=1)

        choice = random.randrange(len(self.flare_list))
        flare = remove_background(self._load(self.flare_list[choice], adjust_gamma))
        light = self._load(self.light_list[choice], adjust_gamma)

        # блик и источник света крутим одним и тем же преобразованием — иначе разъедутся
        merged = self.transform_flare(torch.cat((flare, light), dim=0))
        flare, light = torch.split(merged, 3, dim=0)
        flare = torch.clamp(self.blur_transform(flare), min=0, max=1)
        light = torch.clamp(self.blur_transform(light), min=0, max=1)

        return {
            "lq": adjust_gamma_reverse(torch.clamp(base + flare, min=0, max=1)),
            "gt": adjust_gamma_reverse(torch.clamp(base + light, min=0, max=1)),
            "flare": adjust_gamma_reverse(torch.clamp(flare - light, min=0, max=1)),
        }
