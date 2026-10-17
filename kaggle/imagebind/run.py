# Kaggle run for E5: pretrained ImageBind on ESC-50 and Flickr30k test images.
import os
import shutil
import subprocess
import sys

INPUT = "/kaggle/input"
print("started", flush=True)


def find(name):
    # follow Kaggle's symlinked mounts, but visit each real directory once
    seen = set()
    for dirpath, dirnames, filenames in os.walk(INPUT, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        if name in filenames:
            return os.path.join(dirpath, name)
    raise FileNotFoundError(name)


os.chdir("/kaggle/working")
src_dir = os.path.dirname(find("heads.py"))
print("src at", src_dir, flush=True)
shutil.copytree(src_dir, "src", dirs_exist_ok=True)


def run(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


# ImageBind's setup.py pins torch 1.13, so install it without dependencies.
run("git", "clone", "--depth", "1", "https://github.com/facebookresearch/ImageBind", "/tmp/ImageBind")
run(sys.executable, "-u", "-m", "pip", "install", "-q", "--no-deps", "/tmp/ImageBind",
    "git+https://github.com/facebookresearch/pytorchvideo.git")
run(sys.executable, "-u", "-m", "pip", "install", "-q", "timm", "ftfy", "regex", "einops", "iopath", "fvcore")
run(sys.executable, "-c", "import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__)")

karpathy = find("dataset_flickr30k.json")
# ImageBind loads its BPE vocab and checkpoint from paths relative to the working dir
shutil.copytree("/tmp/ImageBind/bpe", "bpe", dirs_exist_ok=True)
run(sys.executable, "-u", "-m", "src.imagebind_eval", "--esc50-root", INPUT, "--karpathy-json", karpathy,
    "--flickr-root", INPUT, "--out-dir", "results/e5")
shutil.rmtree("src")
shutil.rmtree("bpe")
shutil.rmtree(".checkpoints", ignore_errors=True)
