import argparse
import json
import os
import time

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from src.data import ImageFiles, ImagesOnly, cifar10_prompts, find_dir_with, load_karpathy

IMAGE_ENCODERS = ["resnet50", "vit_b_16", "clip"]
TEXT_ENCODERS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}
CLIP_ARCH, CLIP_WEIGHTS = "ViT-B-32", "laion2b_s34b_b79k"


def load_image_encoder(name, device):
    """Returns (function from image batch to features, preprocessing transform)."""
    if name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
    elif name == "vit_b_16":
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        model = torchvision.models.vit_b_16(weights=weights)
        model.heads = torch.nn.Identity()
    elif name == "clip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=CLIP_WEIGHTS)
        model = model.to(device).eval()
        return model.encode_image, preprocess
    else:
        raise ValueError(name)
    model = model.to(device).eval()
    return model, weights.transforms()


@torch.no_grad()
def encode_images(encoder, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    out = []
    for batch in loader:
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            out.append(encoder(batch.to(device)).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def encode_texts(name, texts, device, batch_size=256):
    if name == "clip":
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=CLIP_WEIGHTS)
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(CLIP_ARCH)
        out = []
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(texts[start:start + batch_size]).to(device)
            out.append(model.encode_text(tokens).float().cpu())
        return torch.cat(out).numpy()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(TEXT_ENCODERS[name], device=device)
    # keep raw outputs; the heads and metrics normalise where needed
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=False, show_progress_bar=False)


def done(out_dir, name):
    return os.path.exists(os.path.join(out_dir, name + ".npy"))


def save(out_dir, name, array, start):
    np.save(os.path.join(out_dir, name + ".npy"), array.astype(np.float16))
    print(f"saved {name} {array.shape} in {time.time() - start:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--karpathy-json", required=True)
    parser.add_argument("--flickr-root", required=True, help="folder that contains the Flickr30k images somewhere below it")
    parser.add_argument("--cifar-root", default="data/cifar")
    parser.add_argument("--out", default="emb")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    filenames, splits, captions, caption_image = load_karpathy(args.karpathy_json)
    image_dir = find_dir_with(args.flickr_root, filenames[0])
    with open(os.path.join(args.out, "flickr_meta.json"), "w") as f:
        json.dump({"images": filenames, "split": splits, "captions": captions,
                   "caption_image": caption_image}, f)

    cifar_test = torchvision.datasets.CIFAR10(args.cifar_root, train=False, download=True)
    cifar_train = torchvision.datasets.CIFAR10(args.cifar_root, train=True, download=True)
    np.save(os.path.join(args.out, "cifar_test_labels.npy"), np.array(cifar_test.targets))
    np.save(os.path.join(args.out, "cifar_train_labels.npy"), np.array(cifar_train.targets))

    for name in IMAGE_ENCODERS:
        encoder, preprocess = load_image_encoder(name, device)
        jobs = [(f"flickr_img_{name}", ImageFiles(image_dir, filenames, preprocess))]
        cifar_test.transform = preprocess
        jobs.append((f"cifar_test_img_{name}", ImagesOnly(cifar_test)))
        if name != "clip":  # train set is only the k-NN bank for E1
            cifar_train.transform = preprocess
            jobs.append((f"cifar_train_img_{name}", ImagesOnly(cifar_train)))
        for out_name, dataset in jobs:
            if not done(args.out, out_name):
                start = time.time()
                save(args.out, out_name, encode_images(encoder, dataset, device), start)
        del encoder
        torch.cuda.empty_cache()

    for name in list(TEXT_ENCODERS) + ["clip"]:
        for out_name, texts in [(f"flickr_txt_{name}", captions), (f"cifar_prompts_txt_{name}", cifar10_prompts())]:
            if not done(args.out, out_name):
                start = time.time()
                save(args.out, out_name, encode_texts(name, texts, device), start)

if __name__ == "__main__":
    main()
