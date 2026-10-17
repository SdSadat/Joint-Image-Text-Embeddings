# Kaggle run 1: cache all frozen-encoder embeddings, then E1 and the OpenCLIP ceiling.
import glob
import os
import shutil
import subprocess
import sys

INPUT = "/kaggle/input"
os.chdir("/kaggle/working")
src_dir = os.path.dirname(glob.glob(f"{INPUT}/**/heads.py", recursive=True)[0])
shutil.copytree(src_dir, "src", dirs_exist_ok=True)


def run(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


run(sys.executable, "-m", "pip", "install", "-q", "open_clip_torch", "umap-learn")
run("nvidia-smi", "--query-gpu=name", "--format=csv")
karpathy = glob.glob(f"{INPUT}/**/dataset_flickr30k.json", recursive=True)[0]

run(sys.executable, "-m", "src.embed", "--karpathy-json", karpathy, "--flickr-root", INPUT, "--out", "emb")
run(sys.executable, "-m", "src.unimodal", "--emb-dir", "emb", "--out-dir", "results/e1")
run(sys.executable, "-m", "src.evaluate", "--emb-dir", "emb", "--enc", "clip", "--out", "results/e2/clip_ceiling.csv")
shutil.rmtree("src")
