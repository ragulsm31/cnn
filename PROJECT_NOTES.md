# Project notes

Source notebook: uploaded `covid-19-prediction-using-cnn.ipynb`.

The cleaned project preserves the original baseline CNN idea and three-class task while making the project portable outside Kaggle.

Changes made for reproducibility:
- Removed Kaggle `/input/...` paths.
- Added KaggleHub dataset download script.
- Added consistent train/test preprocessing.
- Added model saving.
- Added reusable CLI inference script.
- Added README, requirements, and .gitignore.
- Kept dataset and trained model out of Git history because they are generated/large artifacts.
