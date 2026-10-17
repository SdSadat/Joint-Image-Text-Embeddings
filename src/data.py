import csv
import json
import os

from PIL import Image
from torch.utils.data import Dataset

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def find_file(root, name):
    """Kaggle datasets nest files differently; return the first match under root.
    Kaggle mounts inputs through symlinks, so follow them but visit each real dir once."""
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        if name in filenames:
            return os.path.join(dirpath, name)
    raise FileNotFoundError(f"{name} not found under {root}")


def find_dir_with(root, filename):
    return os.path.dirname(find_file(root, filename))


def load_karpathy(json_path):
    """Flickr30k Karpathy split. Returns image filenames, their split, the captions
    and, for each caption, the index of the image it belongs to."""
    with open(json_path) as f:
        data = json.load(f)
    filenames, splits, captions, caption_image = [], [], [], []
    for i, item in enumerate(data["images"]):
        filenames.append(item["filename"])
        splits.append(item["split"])
        for sentence in item["sentences"]:
            captions.append(sentence["raw"].strip())
            caption_image.append(i)
    return filenames, splits, captions, caption_image


class ImageFiles(Dataset):
    def __init__(self, image_dir, filenames, transform):
        self.paths = [os.path.join(image_dir, name) for name in filenames]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


class ImagesOnly(Dataset):
    """Wraps a labelled dataset (e.g. torchvision CIFAR10) and drops the label."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i][0]


def cifar10_prompts():
    return [f"a photo of a {name}." for name in CIFAR10_CLASSES]


def load_esc50(root):
    """Returns (wav paths, integer labels, class names) for ESC-50."""
    meta_path = find_file(root, "esc50.csv")
    audio_dir = find_dir_with(root, "1-100032-A-0.wav")
    paths, labels, names = [], [], {}
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            paths.append(os.path.join(audio_dir, row["filename"]))
            labels.append(int(row["target"]))
            names[int(row["target"])] = row["category"]
    class_names = [names[i].replace("_", " ") for i in range(len(names))]
    return paths, labels, class_names
