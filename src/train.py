import argparse
import copy
import math
import os
import random

import numpy as np
import torch

from src.evaluate import (append_row, load_cifar, load_split, modality_gap,
                          recall_at_k, zero_shot_accuracy)
from src.heads import ProjectionHead
from src.losses import infonce, sigmoid_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", required=True)
    parser.add_argument("--img-enc", default="vit_b_16")
    parser.add_argument("--txt-enc", default="mpnet")
    parser.add_argument("--loss", choices=["infonce", "sigmoid"], default="infonce")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temp", type=float, default=0.07, help="initial temperature")
    parser.add_argument("--fixed-temp", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--random-heads", action="store_true", help="skip training (floor baseline)")
    parser.add_argument("--save-test-emb", default=None, help="folder for projected test embeddings")
    parser.add_argument("--out", default="results/runs.csv")
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def project(head, x, chunk=4096):
    head.eval()
    return torch.cat([head(x[i:i + chunk]) for i in range(0, len(x), chunk)])


def val_score(img_head, txt_head, img, txt, caption_image):
    r = recall_at_k(project(img_head, img), project(txt_head, txt), caption_image)
    return (r["i2t_r5"] + r["t2i_r5"]) / 2


def train(args, device):
    img, txt, caption_image = load_split(args.emb_dir, args.img_enc, args.txt_enc, "train")
    val_img, val_txt, val_caption_image = load_split(args.emb_dir, args.img_enc, args.txt_enc, "val")
    img, txt = img.to(device), txt.to(device)
    val_img, val_txt = val_img.to(device), val_txt.to(device)

    captions_of = [[] for _ in range(len(img))]
    for j, i in enumerate(caption_image):
        captions_of[i].append(j)

    img_head = ProjectionHead(img.shape[1]).to(device)
    txt_head = ProjectionHead(txt.shape[1]).to(device)
    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / args.temp), device=device),
                                     requires_grad=not args.fixed_temp)
    bias = torch.nn.Parameter(torch.tensor(-10.0, device=device))

    if args.random_heads:
        return img_head, txt_head, logit_scale, 0

    params = list(img_head.parameters()) + list(txt_head.parameters())
    if not args.fixed_temp:
        params.append(logit_scale)
    if args.loss == "sigmoid":
        params.append(bias)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_score, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, args.epochs + 1):
        img_head.train()
        txt_head.train()
        # one caption per image per epoch, so a batch never holds the same image twice
        order = torch.randperm(len(img)).tolist()
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            if len(ids) < 2:
                continue
            cap_ids = [random.choice(captions_of[i]) for i in ids]
            u = img_head(img[ids])
            v = txt_head(txt[cap_ids])
            if args.loss == "infonce":
                loss = infonce(u, v, logit_scale)
            else:
                loss = sigmoid_loss(u, v, logit_scale, bias)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                logit_scale.clamp_(0, math.log(100))
            total_loss += loss.item()
            n_batches += 1

        score = val_score(img_head, txt_head, val_img, val_txt, val_caption_image)
        print(f"epoch {epoch:2d}  loss {total_loss / n_batches:.4f}  val R@5 {score:.2f}  "
              f"temp {1 / logit_scale.exp().item():.4f}")

        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = (copy.deepcopy(img_head.state_dict()), copy.deepcopy(txt_head.state_dict()),
                          logit_scale.detach().clone())
        elif epoch - best_epoch >= args.patience:
            print("early stopping")
            break

    img_head.load_state_dict(best_state[0])
    txt_head.load_state_dict(best_state[1])
    with torch.no_grad():
        logit_scale.copy_(best_state[2])
    return img_head, txt_head, logit_scale, best_epoch


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_head, txt_head, logit_scale, best_epoch = train(args, device)

    test_img, test_txt, test_caption_image = load_split(args.emb_dir, args.img_enc, args.txt_enc, "test")
    u = project(img_head, test_img.to(device))
    v = project(txt_head, test_txt.to(device))

    row = {
        "tag": args.tag,
        "img_enc": args.img_enc,
        "txt_enc": args.txt_enc,
        "loss": "random" if args.random_heads else args.loss,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "temp_init": args.temp,
        "fixed_temp": args.fixed_temp,
        "temp_final": 1 / logit_scale.exp().item(),
        "best_epoch": best_epoch,
    }
    row.update(recall_at_k(u, v, test_caption_image))
    row["modality_gap"] = modality_gap(u, v)

    cifar, prompts, labels = load_cifar(args.emb_dir, args.img_enc, args.txt_enc)
    row["cifar10_zero_shot"] = zero_shot_accuracy(project(img_head, cifar.to(device)),
                                                  project(txt_head, prompts.to(device)), labels)
    append_row(args.out, row)
    print(row)

    if args.save_test_emb:
        os.makedirs(args.save_test_emb, exist_ok=True)
        np.save(os.path.join(args.save_test_emb, "img.npy"), u.cpu().numpy())
        np.save(os.path.join(args.save_test_emb, "txt.npy"), v.cpu().numpy())
        np.save(os.path.join(args.save_test_emb, "caption_image.npy"), np.array(test_caption_image))


if __name__ == "__main__":
    main()
