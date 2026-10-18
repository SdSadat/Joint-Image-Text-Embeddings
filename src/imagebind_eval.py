import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import find_dir_with, load_esc50, load_karpathy
from src.evaluate import append_row, zero_shot_accuracy

# ESC-50 classes that have an obvious word in Flickr30k captions, used to score
# sound -> image search without manual labels.
CAPTION_WORDS = {
    "dog": ["dog", "dogs", "puppy"],
    "cat": ["cat", "cats", "kitten"],
    "rooster": ["rooster", "chicken", "chickens"],
    "cow": ["cow", "cows", "cattle"],
    "rain": ["rain", "raining", "umbrella"],
    "sea waves": ["ocean", "wave", "waves", "surf", "surfer", "beach"],
    "crying baby": ["baby", "infant"],
    "church bells": ["church"],
    "train": ["train", "subway"],
    "airplane": ["plane", "airplane", "jet"],
    "car horn": ["car", "cars", "traffic"],
    "keyboard typing": ["computer", "laptop", "keyboard"],
    "engine": ["car", "motorcycle", "truck"],
    "laughing": ["laughing", "laughs", "smiling"],
}


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def load_imagebind(device):
    # Meta's download server is very slow from Kaggle; this Hugging Face copy has
    # exactly the same tensor names and shapes as the official checkpoint.
    from huggingface_hub import hf_hub_download
    from imagebind.models import imagebind_model
    from safetensors.torch import load_file

    model = imagebind_model.imagebind_huge(pretrained=False)
    log("downloading weights")
    path = hf_hub_download("nielsr/imagebind-huge", "model.safetensors")
    log("loading weights")
    model.load_state_dict(load_file(path))
    return model.to(device).eval()


def load_wav(path):
    # newer torchaudio needs torchcodec + ffmpeg just to read a file; plain wavs don't
    import soundfile
    waveform, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(waveform.T), sample_rate


@torch.no_grad()
def embed(model, modality, items, load_fn, device, batch_size=32):
    out = []
    for start in range(0, len(items), batch_size):
        inputs = {modality: load_fn(items[start:start + batch_size], device)}
        out.append(model(inputs)[modality].float().cpu())
    return F.normalize(torch.cat(out), dim=-1)


def caption_words(caption):
    return set(caption.lower().replace(".", " ").replace(",", " ").split())


def save_grid(queries, top_paths, path):
    fig, axes = plt.subplots(len(queries), 5, figsize=(10, 2 * len(queries)), squeeze=False)
    for row, (query, paths) in enumerate(zip(queries, top_paths)):
        for col, image_path in enumerate(paths):
            ax = axes[row, col]
            ax.imshow(Image.open(image_path).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(query, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc50-root", required=True)
    parser.add_argument("--karpathy-json", required=True)
    parser.add_argument("--flickr-root", required=True)
    parser.add_argument("--out-dir", default="results/e5")
    args = parser.parse_args()

    from imagebind import data
    from imagebind.models.imagebind_model import ModalityType

    data.torchaudio.load = load_wav
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    model = load_imagebind(device)
    log("model ready")

    # Zero-shot ESC-50 over all 2000 clips
    audio_paths, labels, class_names = load_esc50(args.esc50_root)
    log(f"embedding {len(audio_paths)} audio clips")
    audio = embed(model, ModalityType.AUDIO, audio_paths, data.load_and_transform_audio_data, device)
    prompts = [f"a sound of a {name}." for name in class_names]
    text = embed(model, ModalityType.TEXT, prompts, data.load_and_transform_text, device)
    acc = zero_shot_accuracy(audio, text, labels)
    append_row(os.path.join(args.out_dir, "esc50_zero_shot.csv"), {"model": "imagebind_huge", "top1": acc})
    print("ESC-50 zero-shot top-1:", acc)

    # Sound -> image search over the Flickr30k test images
    filenames, splits, captions, caption_image = load_karpathy(args.karpathy_json)
    image_dir = find_dir_with(args.flickr_root, filenames[0])
    test_ids = [i for i, s in enumerate(splits) if s == "test"]
    image_paths = [os.path.join(image_dir, filenames[i]) for i in test_ids]
    words_of = {i: set() for i in test_ids}
    for caption, i in zip(captions, caption_image):
        if i in words_of:
            words_of[i] |= caption_words(caption)
    log(f"embedding {len(image_paths)} images")
    images = embed(model, ModalityType.VISION, image_paths, data.load_and_transform_vision_data, device)

    top5 = (audio @ images.t()).topk(min(5, len(images)), dim=1).indices
    labels = np.array(labels)
    grid_queries, grid_paths = [], []
    for class_name, words in CAPTION_WORDS.items():
        if class_name not in class_names:
            continue
        clips = np.where(labels == class_names.index(class_name))[0]
        n_with_word = sum(bool(w & set(words)) for w in words_of.values())
        hits = [[bool(words_of[test_ids[k]] & set(words)) for k in top5[c].tolist()] for c in clips]
        append_row(os.path.join(args.out_dir, "sound_to_image.csv"),
                   {"class": class_name, "clips": len(clips), "precision_at_5": float(np.mean(hits)) * 100,
                    "chance": n_with_word / len(test_ids) * 100})
        grid_queries.append(class_name)
        grid_paths.append([image_paths[k] for k in top5[clips[0]].tolist()])
    save_grid(grid_queries, grid_paths, os.path.join(args.out_dir, "sound_to_image_examples.png"))


if __name__ == "__main__":
    main()
