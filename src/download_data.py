from pathlib import Path
import shutil
import kagglehub

DATASET = "pranavraikokte/covid19-image-dataset"
TARGET = Path("data/Covid19-dataset")

path = Path(kagglehub.dataset_download(DATASET))
print(f"Kaggle cache path: {path}")

# kagglehub may return a directory containing Covid19-dataset.
source = path / "Covid19-dataset"
if not source.exists():
    source = path

TARGET.parent.mkdir(parents=True, exist_ok=True)
if TARGET.exists():
    shutil.rmtree(TARGET)
shutil.copytree(source, TARGET)

print(f"Dataset copied to: {TARGET.resolve()}")
