# OcclusionNet

Выявление и классификация окклюзий на кадрах с автомобильных камер плюс оценка
того, насколько сильно кадр испорчен.

Задача multilabel: на кадре может быть несколько окклюзий сразу, у чистого кадра
вектор меток нулевой. Классы: `DaytimeFlare`, `Fog`, `MotionBlur`,
`NighttimeFlare`, `Raindrops`, `Reflections`, `Soil`. Отдельно `Clean`,
отсутствие всех семи.

## Структура

| Папка | Что внутри |
|---|---|
| `training/` | `occlusionnet.ipynb`, обучение основной модели |
| `validation/` | `testclassifier.ipynb`, прогон весов на отложенной выборке |
| `occlusion_classifier/` | классификатор поверх энкодера сегментационной модели |
| `occlusion_score/` | оценка степени загрязнённости кадра |
| `flare_synthesis/` | генерация синтетических кадров с бликами |
| `data/` | описание датасетов, заливка Google Drive в Kaggle |
| `infra/` | ClearML-сервер в Yandex Cloud, см. [infra/README.md](infra/README.md) |

Три пакета устроены одинаково: код в `src/<пакет>/`, CLI в `scripts/`,
зависимости в `pyproject.toml`, исходные ноутбуки в `notebooks/`.

## Данные

* Kaggle: https://www.kaggle.com/datasets/mishasavinov/occlusion-dataset
* Google Drive: [папка с датасетами](https://drive.google.com/drive/folders/1P81qiLkSbpOT2rNEV-64RXUGqdlRCUep)

Раскладка: `Datasets/<Класс>/<Источник>/{train,test}/`, кадры 512×512. Папка
`Clean/` подключается отдельным датасетом. Подробности в
[data/occlusions_datasets.md](data/occlusions_datasets.md).

## Запуск

Всё считается на Kaggle, Colab или DataSphere, в облаке живёт только ClearML.
Ключи нигде в репозитории не хранятся, они берутся из переменных окружения
`CLEARML_API_HOST`, `CLEARML_API_ACCESS_KEY`, `CLEARML_API_SECRET_KEY`.

**Основная модель.** EfficientNet-B3 и семь независимых MLP-голов,
`BCEWithLogitsLoss`, 20 эпох. Обучение в `training/occlusionnet.ipynb`, оно же
подбирает пороги по классам и складывает веса в артефакты задачи. Проверка на
отложенной выборке в `validation/testclassifier.ipynb`, нужно указать
`TRAIN_TASK_ID`.

**Классификатор на сегментационном энкодере.** Вторая ветка эксперимента: энкодер
обученной FPN с ResNet-18, поверх по обучаемому query на класс с кросс-аттеншеном
к карте признаков, лосс focal BCE. В примере конфига шесть классов, `Reflections`
отключён из-за нехватки данных.

```bash
cp occlusion_classifier/configs/train.example.yaml my.yaml   # правим пути
python occlusion_classifier/scripts/train.py --config my.yaml
python occlusion_classifier/scripts/predict.py <bundle_dir> <папка_с_кадрами>
```

**Генерация бликов.** Блик из FlareX накладывается на фоновый кадр вождения со
случайной гаммой, шумом, аффинным преобразованием и блюром.

```bash
python flare_synthesis/scripts/generate.py \
  --background-dir <кадры_вождения> \
  --flare-dir <FlareX/Flare2D/input> --light-dir <FlareX/Flare2D/gt> \
  --output-dir ./flare_generated -n 1000 --crop-size 512 --archive
```

**Степень загрязнённости.** Основной вариант обучения не требует: кадр
сегментируется тяжёлой (`segformer-b5`) и лёгкой (`segformer-b0`) моделями,
разница их уверенности усредняется по пикселям. Ноутбук и результаты лежат
в `occlusion_score/diff_model_conf/`. Обучаемый вариант дистиллирует признаки
DINOv2 в MobileNetV3 и предсказывает разметку людей:

```bash
python occlusion_score/scripts/train.py --config <config.yaml>
python occlusion_score/scripts/predict.py <bundle_dir> <папка_с_кадрами>
```

## Зависимости

`torch`, `torchvision`, `clearml`, `scikit-learn`, `pandas`, `numpy`, `pillow`,
`tqdm`, `matplotlib`, `seaborn`, плюс `transformers` для скора
и `segmentation_models_pytorch` для классификатора. На Kaggle и Colab стоит всё,
кроме `clearml` и `segmentation_models_pytorch`. Версии в `pyproject.toml`
соответствующего пакета.
