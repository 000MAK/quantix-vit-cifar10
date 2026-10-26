# Vision Transformer | CIFAR-10

A compact patch-token Vision Transformer implemented in PyTorch for CIFAR-10 image classification. The script downloads CIFAR-10 on first use, takes a reproducible subset of the training set, and keeps a separate test subset for final evaluation.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --epochs 1 --limit 2000 --batch-size 64
```

Run `python main.py --help` for options. Data is stored in `data/`; weights and metrics are written to `outputs/`. Those paths are excluded from Git.

## Model and interpretation

Images are divided into 8x8 patches, projected into tokens, processed by a small Transformer encoder, then classified from a learned class token. The default subset and short runs are learning baselines; compare against a CNN and report the data size and number of epochs with any metric.

Dataset source: [CIFAR-10, University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html).
