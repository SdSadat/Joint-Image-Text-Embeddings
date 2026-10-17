# Joint Image–Text Embeddings

AI course term project. We freeze pretrained image and text encoders, cache their
outputs once, and train only two small MLP projection heads that map both modalities
into a shared 512-d L2-normalised space. Then we study how that space behaves.

## Research questions

- **RQ1** How much of CLIP's retrieval quality do small heads on frozen encoders recover?
- **RQ2** Softmax InfoNCE (CLIP) vs pairwise sigmoid (SigLIP) at small batch sizes.
- **RQ3** Size of the modality gap (distance between the mean image and mean text
  embedding) and how temperature changes it.
- **RQ4** Audio extension, evaluation only: pretrained ImageBind, no training.

## Experiments

| ID | What | Metric |
|----|------|--------|
| E1 | Image encoders ResNet-50, ViT-B/16; text encoders all-MiniLM-L6-v2, all-mpnet-base-v2 | k-NN accuracy, UMAP |
| E2 | Train heads on Flickr30k; random heads (floor) and OpenCLIP ViT-B-32 (ceiling) | Recall@1/5/10 both directions on the 1k test split; CIFAR-10 zero-shot |
| E3 | InfoNCE vs sigmoid at batch size 64 and 256, 3 seeds | Recall@k |
| E4 | Temperature 0.01, 0.07, 0.5 | modality gap, UMAP before/after |
| E5 | ImageBind | ESC-50 zero-shot top-1, sound→image search on Flickr30k test |

E1 k-NN: image encoders are scored on CIFAR-10 (train set as the bank, test set as
queries). Text encoders are scored on the Flickr30k test captions: a caption counts as
correct if its nearest other caption describes the same image.

## Layout

```
src/data.py            dataset loading, Karpathy split, captions
src/embed.py           run frozen encoders once, save embeddings
src/heads.py           projection MLP
src/losses.py          infonce(), sigmoid_loss()
src/train.py           train heads on cached embeddings
src/evaluate.py        recall@k, zero-shot, modality gap, k-NN
src/imagebind_eval.py  E5
kaggle/                one kernel folder per run
results/               small CSVs and plots
```

## Running

All GPU work runs on Kaggle. Embeddings are computed once by `src/embed.py` and stored
as a private Kaggle dataset; training runs only read those files.

```
python -m src.embed --karpathy-json <path> --flickr-images <dir> --out emb/
python -m src.train --emb-dir emb/ --loss infonce --batch-size 256 --seed 0
```

## Head and training

`Linear(in, 1024) -> GELU -> Dropout(0.1) -> Linear(1024, 512)`, then L2 normalisation.
AdamW, lr 1e-3, up to 20 epochs, learnable temperature starting at 0.07, early stopping
on validation Recall@5.
