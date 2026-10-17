# Kaggle runs for E2-E4: train heads on the embeddings cached by jite-embed.
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
emb_dir = os.path.dirname(find("flickr_meta.json"))

SEEDS = [0, 1, 2]
MAIN = ["--img-enc", "vit_b_16", "--txt-enc", "mpnet"]


def run(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def train(out, *flags):
    run(sys.executable, "-u", "-m", "src.train", "--emb-dir", emb_dir, "--out", out, *flags)


run(sys.executable, "-u", "-m", "pip", "install", "-q", "umap-learn")

for seed in SEEDS:
    s = ["--seed", str(seed)]
    # E2: every encoder pair, plus untrained heads as the floor
    for img_enc in ["resnet50", "vit_b_16"]:
        for txt_enc in ["minilm", "mpnet"]:
            pair = ["--img-enc", img_enc, "--txt-enc", txt_enc]
            train("results/e2/runs.csv", *pair, *s)
            train("results/e2/runs.csv", *pair, *s, "--random-heads")

    # E3: loss x batch size
    for loss in ["infonce", "sigmoid"]:
        for batch_size in ["64", "256"]:
            train("results/e3/runs.csv", *MAIN, *s, "--loss", loss, "--batch-size", batch_size)

    # E4: fixed temperatures; keep seed-0 projections for the UMAP plots
    for temp in ["0.01", "0.07", "0.5"]:
        save = ["--save-test-emb", f"proj/fixed_t{temp}"] if seed == 0 else []
        train("results/e4/runs.csv", *MAIN, *s, "--temp", temp, "--fixed-temp", *save)

train("results/e4/extra.csv", *MAIN, "--save-test-emb", "proj/learned_temp")
train("results/e4/extra.csv", *MAIN, "--random-heads", "--save-test-emb", "proj/before_training")
run(sys.executable, "-u", "-m", "src.plots", "--results", "results", "--proj", "proj")
shutil.rmtree("src")
