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

## Results

No reference run is published yet. When you run it, report the training-set size, epochs and seed alongside any metric.

## Repository contents

| File | Purpose |
|---|---|
| `.gitignore` | Keeps data, weights and outputs out of Git |
| `LICENSE` | MIT licence for the code |
| `main.py` | Training and evaluation script |
| `requirements.txt` | Python dependencies |

## License

Code is released under the MIT License (see `LICENSE`). Datasets keep their own licences and are downloaded at run time, not redistributed here.

## Related

Part of a set of compact, reproducible ML/DL baselines by [000MAK](https://github.com/000MAK). Flagship research project: [QMASE martial-arts analytics engine](https://github.com/000MAK/quantix-martial-arts-engine).
