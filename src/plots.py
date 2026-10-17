import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def mean_r1(row):
    return (float(row["i2t_r1"]) + float(row["t2i_r1"])) / 2


def group(rows, keys, value_fn):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(value_fn(row))
    return {k: (np.mean(v), np.std(v), len(v)) for k, v in groups.items()}


def plot_losses(e3_csv, path):
    stats = group(read_rows(e3_csv), ["loss", "batch_size"], mean_r1)
    batch_sizes = sorted({int(b) for _, b in stats}, key=int)
    width = 0.35
    plt.figure(figsize=(5, 4))
    for offset, loss in [(-width / 2, "infonce"), (width / 2, "sigmoid")]:
        means = [stats[(loss, str(b))][0] for b in batch_sizes]
        stds = [stats[(loss, str(b))][1] for b in batch_sizes]
        plt.bar(np.arange(len(batch_sizes)) + offset, means, width, yerr=stds, capsize=4, label=loss)
    plt.xticks(range(len(batch_sizes)), [f"batch {b}" for b in batch_sizes])
    plt.ylabel("mean Recall@1 (i2t, t2i), %")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_gap(e4_csv, path):
    rows = read_rows(e4_csv)
    gap = group(rows, ["temp_init"], lambda r: float(r["modality_gap"]))
    r1 = group(rows, ["temp_init"], mean_r1)
    temps = sorted(gap, key=lambda k: float(k[0]))
    x = [float(t[0]) for t in temps]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(x, [gap[t][0] for t in temps], yerr=[gap[t][1] for t in temps], marker="o", capsize=4)
    ax.set_xscale("log")
    ax.set_xlabel("temperature (fixed)")
    ax.set_ylabel("modality gap ||mean(img) - mean(txt)||")
    ax2 = ax.twinx()
    ax2.errorbar(x, [r1[t][0] for t in temps], yerr=[r1[t][1] for t in temps], marker="s",
                 color="tab:orange", capsize=4)
    ax2.set_ylabel("mean Recall@1, %", color="tab:orange")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_joint_umap(proj_dir, title, path):
    import umap
    img = np.load(os.path.join(proj_dir, "img.npy"))
    txt = np.load(os.path.join(proj_dir, "txt.npy"))
    caption_image = np.load(os.path.join(proj_dir, "caption_image.npy"))
    first_caption = np.unique(caption_image, return_index=True)[1]
    txt = txt[first_caption]
    gap = np.linalg.norm(img.mean(0) - txt.mean(0))
    points = umap.UMAP(metric="cosine", random_state=0).fit_transform(np.concatenate([img, txt]))
    plt.figure(figsize=(5, 4.5))
    plt.scatter(*points[:len(img)].T, s=3, label="image")
    plt.scatter(*points[len(img):].T, s=3, label="text")
    plt.legend(markerscale=4)
    plt.title(f"{title} (gap {gap:.3f})")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--proj", default=None, help="folder of saved test projections for UMAP plots")
    args = parser.parse_args()

    plot_losses(os.path.join(args.results, "e3/runs.csv"), os.path.join(args.results, "e3/losses.png"))
    plot_gap(os.path.join(args.results, "e4/runs.csv"), os.path.join(args.results, "e4/gap_vs_temperature.png"))
    if args.proj:
        for name in sorted(os.listdir(args.proj)):
            plot_joint_umap(os.path.join(args.proj, name), name,
                            os.path.join(args.results, f"e4/umap_{name}.png"))


if __name__ == "__main__":
    main()
