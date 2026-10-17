import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F


def recall_at_k(img, txt, caption_image, ks=(1, 5, 10)):
    """img: (N_img, D), txt: (N_cap, D), both normalised.
    caption_image[j] is the index of the image that caption j describes."""
    caption_image = torch.as_tensor(caption_image, device=img.device)
    sims = img @ txt.t()
    max_k = max(ks)

    # image -> text: hit if any of the image's captions is in the top k
    top_caps = sims.topk(max_k, dim=1).indices
    i2t_hits = caption_image[top_caps] == torch.arange(len(img), device=img.device)[:, None]

    # text -> image: hit if the caption's own image is in the top k
    top_imgs = sims.t().topk(max_k, dim=1).indices
    t2i_hits = top_imgs == caption_image[:, None]

    out = {}
    for k in ks:
        out[f"i2t_r{k}"] = i2t_hits[:, :k].any(dim=1).float().mean().item() * 100
        out[f"t2i_r{k}"] = t2i_hits[:, :k].any(dim=1).float().mean().item() * 100
    return out


def zero_shot_accuracy(img, class_txt, labels):
    preds = (img @ class_txt.t()).argmax(dim=1).cpu()
    return (preds == torch.as_tensor(labels)).float().mean().item() * 100


def modality_gap(img, txt):
    # Liang et al. 2022: distance between the two centroids of normalised embeddings
    return (img.mean(dim=0) - txt.mean(dim=0)).norm().item()


def knn_accuracy(bank, bank_labels, queries, query_labels, k=10, chunk=1024):
    bank = F.normalize(bank, dim=-1)
    queries = F.normalize(queries, dim=-1)
    bank_labels = torch.as_tensor(bank_labels, device=bank.device)
    query_labels = torch.as_tensor(query_labels, device=bank.device)
    n_classes = int(bank_labels.max()) + 1
    correct = 0
    for start in range(0, len(queries), chunk):
        q = queries[start:start + chunk]
        top = (q @ bank.t()).topk(k, dim=1).indices
        votes = F.one_hot(bank_labels[top], n_classes).sum(dim=1)
        correct += (votes.argmax(dim=1) == query_labels[start:start + chunk]).sum().item()
    return correct / len(queries) * 100


def caption_nn_accuracy(txt, caption_image):
    """Text-only k-NN check: is a caption's nearest other caption about the same image?"""
    txt = F.normalize(txt, dim=-1)
    caption_image = torch.as_tensor(caption_image, device=txt.device)
    sims = txt @ txt.t()
    sims.fill_diagonal_(-2)
    nearest = sims.argmax(dim=1)
    return (caption_image[nearest] == caption_image).float().mean().item() * 100


def load_split(emb_dir, img_enc, txt_enc, split):
    """Return normalised image and caption embeddings for one Karpathy split,
    and the caption -> image index mapping within that split."""
    with open(os.path.join(emb_dir, "flickr_meta.json")) as f:
        meta = json.load(f)
    img_all = np.load(os.path.join(emb_dir, f"flickr_img_{img_enc}.npy"))
    txt_all = np.load(os.path.join(emb_dir, f"flickr_txt_{txt_enc}.npy"))

    img_ids = [i for i, s in enumerate(meta["split"]) if s == split]
    new_index = {old: new for new, old in enumerate(img_ids)}
    cap_ids = [j for j, i in enumerate(meta["caption_image"]) if i in new_index]
    caption_image = [new_index[meta["caption_image"][j]] for j in cap_ids]

    img = torch.from_numpy(img_all[img_ids]).float()
    txt = torch.from_numpy(txt_all[cap_ids]).float()
    return img, txt, caption_image


def append_row(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})


def load_cifar(emb_dir, img_enc, txt_enc):
    img = torch.from_numpy(np.load(os.path.join(emb_dir, f"cifar_test_img_{img_enc}.npy"))).float()
    prompts = torch.from_numpy(np.load(os.path.join(emb_dir, f"cifar_prompts_txt_{txt_enc}.npy"))).float()
    labels = np.load(os.path.join(emb_dir, "cifar_test_labels.npy"))
    return img, prompts, labels


def main():
    # Evaluates raw cached embeddings that already share a space (the OpenCLIP ceiling).
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", required=True)
    parser.add_argument("--enc", default="clip")
    parser.add_argument("--out", default="results/clip_ceiling.csv")
    args = parser.parse_args()

    img, txt, caption_image = load_split(args.emb_dir, args.enc, args.enc, "test")
    img, txt = F.normalize(img, dim=-1), F.normalize(txt, dim=-1)
    row = {"model": f"openclip_{args.enc}"}
    row.update(recall_at_k(img, txt, caption_image))
    row["modality_gap"] = modality_gap(img, txt)

    cifar, prompts, labels = load_cifar(args.emb_dir, args.enc, args.enc)
    row["cifar10_zero_shot"] = zero_shot_accuracy(F.normalize(cifar, dim=-1), F.normalize(prompts, dim=-1), labels)
    append_row(args.out, row)
    print(row)


if __name__ == "__main__":
    main()
