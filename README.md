# Vision Transformer CIFAR-10

A compact patch-token Vision Transformer trained on CIFAR-10. Downloads the dataset on first run; configurable epochs and subset size.

## Run

Use Python 3.10+ in a virtual environment; install `pip install -r requirements.txt`, then run `python main.py --help` (or see project-specific command below). Dataset files, model weights and outputs are kept locally and ignored by Git.

Training code uses a fixed random seed/split and writes held-out metrics and model artifacts to `outputs/`.
