# AI-Powered Medical Image Analysis System

An educational computer-vision project that demonstrates binary classification of chest X-ray images into **NORMAL** and **PNEUMONIA** classes. It includes a TensorFlow/Keras training pipeline, evaluation entry point, prediction utilities, and a Streamlit web interface.

> **Responsible-use notice:** This is a student/research prototype, not a clinically validated medical device. Its output must not be used to diagnose, rule out disease, or make treatment decisions. Model scores are not necessarily calibrated probabilities. Clinical use would require independent validation, governance, security controls, and qualified clinical oversight.

## Features

- Upload a chest X-ray image through a Streamlit interface.
- Run an educational model prediction and display the class and score.
- Train a TensorFlow/Keras image classifier using local dataset folders.
- Evaluate the model using the project's evaluation entry point.
- Save training artifacts and evaluation outputs locally.
- Keep the large dataset and trained weight files out of Git history.

## Project structure

```text
AI-Medical-Image-Analysis/
├── app.py                    # Streamlit web application
├── main.py                   # Command-line entry point (train/evaluate)
├── requirements.txt          # Python dependencies
├── data/
│   └── chest_xray/            # Local dataset; not committed to Git
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
├── models/                    # Locally generated model artifacts
├── notebook/                  # Exploratory analysis notebook(s)
├── outputs/                   # Local plots and evaluation reports
├── docs/
│   └── limitations.md         # Limitations and responsible use
└── src/
    ├── __init__.py
    ├── data_utils.py          # Data utilities
    ├── model.py               # Model utilities/architecture
    ├── train.py               # Training pipeline
    ├── evaluate.py            # Evaluation utilities
    └── predict.py             # Prediction utilities
```

Some files and artifact names can vary as the project evolves. The commands below follow the CLI shown by `main.py` (`train` and `evaluate`).

## Requirements

- Python 3.10–3.12, subject to compatibility with the versions pinned in `requirements.txt`
- pip
- Enough disk space for the dataset, trained model, and output files
- CPU is sufficient to experiment, but training can take a long time; a compatible GPU can speed it up

## 1. Clone the repository

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI-Medical-Image-Analysis
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your repository URL. If you already have the project locally, open PowerShell in the folder containing `main.py` and `app.py` instead.

## 2. Create and activate a virtual environment (Windows PowerShell)

Run these commands from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell says that the activation script does not exist, first verify you are in the folder containing `main.py`, then run:

```powershell
Test-Path .\.venv\Scripts\Activate.ps1
```

If it returns `False`, recreate the environment using the first command above. If PowerShell blocks script activation due to the execution policy, use a permitted environment activation method for your machine, or invoke the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Prepare the dataset

This repository intentionally does **not** include the dataset. Download the public dataset from the source you are permitted to use, review its license/terms, and arrange the files as follows:

```text
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Place the actual image files inside the corresponding class folders. The training loader in the supplied version recognizes the class names without relying on uppercase/lowercase spelling, but the class meanings and label order must remain consistent across training, evaluation, and prediction.

Before training, check the directory structure in PowerShell:

```powershell
Get-ChildItem .\data\chest_xray -Directory
Get-ChildItem .\data\chest_xray\train -Directory
Get-ChildItem .\data\chest_xray\val -Directory
Get-ChildItem .\data\chest_xray\test -Directory
```

Do not upload the multi-gigabyte dataset to GitHub. The `.gitignore` file excludes the contents of `data/chest_xray/` by design.

## 4. Train the model

Run this command from the project root (the same directory that contains `main.py`):

```powershell
python main.py train
```

The current CLI accepts the action `train`; it does not accept `--epochs` or `--batch-size` unless those options are added to `main.py`. If your `src/train.py` version reads the `MEDIMG_EPOCHS` and `MEDIMG_BATCH_SIZE` environment variables, you can configure them in PowerShell before training:

```powershell
$env:MEDIMG_EPOCHS = "10"
$env:MEDIMG_BATCH_SIZE = "16"
python main.py train
```

Use those environment variables only if your current training script supports them. TensorFlow may print oneDNN informational messages; those are not by themselves training errors. Training is successful only after the process completes and the expected model artifact is saved.

## 5. Evaluate the model

```powershell
python main.py evaluate
```

Check the terminal output and the `outputs/` directory for evaluation results. Consult `main.py` and `src/evaluate.py` if your local version writes reports to different paths or requires an additional dataset argument.

## 6. Launch the Streamlit application

From the project root:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

Upload a sample X-ray and select the model's prediction button. If the interface reports that the model was not found, first train the model successfully, then compare the path used to save the model in `src/train.py` with the path loaded by `app.py`/`src/predict.py`. Model filenames must match.

## Common terminal issues

### `invalid choice: 'train\\'`

There is an accidental trailing backslash after `train`. Use:

```powershell
python main.py train
```

### `unrecognized arguments: --epochs ... --batch-size ...`

The current `main.py` command-line interface accepts `train` and `evaluate` actions but may not define those options. Run the supported command without them, or add argument definitions to `main.py` deliberately.

### `ValueError: class_names ... did not match`

The class names passed to Keras do not exactly match the class directory names. Inspect the actual folder names and update the loader to use the discovered directory names consistently. Do not relabel or rename the dataset blindly.

### `ModuleNotFoundError: No module named 'src'`

Do not execute `python src/train.py` directly when the file imports `src.*`. Run it through the project entry point instead:

```powershell
python main.py train
```

### `.venv\Scripts\Activate.ps1` not found

The environment may not exist at the current project root. Check the current directory with `Get-Location`, confirm `main.py` is present with `Get-ChildItem`, and create the environment using `python -m venv .venv` if necessary.

## GitHub: recommended files to commit

Commit source code, dependency definitions, documentation, notebooks, and reasonably sized evaluation plots/reports. Do not commit:

- The downloaded chest X-ray dataset
- Virtual environments or Python cache files
- Large trained model weights unless you intentionally distribute them using an appropriate model/data hosting method
- Secrets, API keys, or private medical information

Example commands:

```powershell
git status
git add .gitignore README.md app.py main.py requirements.txt src docs notebook outputs
git commit -m "Document AI medical image analysis project"
git push
```

If Git reports that your local branch has no upstream, use the upstream command for your current branch, for example `git push -u origin main` or `git push -u origin master`. Use the branch name shown by `git branch --show-current`.

## Limitations

- Dataset quality, label quality, class balance, and potential data leakage affect results.
- Performance on one dataset may not generalize to other hospitals, age groups, scanners, or imaging protocols.
- A model score is not automatically a calibrated clinical probability.
- The application is for education and experimentation only and is not a substitute for qualified medical assessment.

## Acknowledgements and data provenance

Add the exact name, link, author, license, and citation for the dataset you downloaded from Kaggle or another source before publishing the repository. Do not imply that this project created or owns the dataset.
