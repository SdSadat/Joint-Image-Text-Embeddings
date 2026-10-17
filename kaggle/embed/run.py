# Kaggle run 1: cache all frozen-encoder embeddings, then E1 and the OpenCLIP ceiling.
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


run(sys.executable, "-u", "-m", "pip", "install", "-q", "open_clip_torch", "umap-learn")
run("nvidia-smi", "--query-gpu=name", "--format=csv")
karpathy = find("dataset_flickr30k.json")

run(sys.executable, "-u", "-m", "src.embed", "--karpathy-json", karpathy, "--flickr-root", INPUT, "--out", "emb")
run(sys.executable, "-u", "-m", "src.unimodal", "--emb-dir", "emb", "--out-dir", "results/e1")
run(sys.executable, "-u", "-m", "src.evaluate", "--emb-dir", "emb", "--enc", "clip", "--out", "results/e2/clip_ceiling.csv")
shutil.rmtree("src")
