import torch
import torch.nn.functional as F


def infonce(img, txt, logit_scale):
    # img, txt: (N, D) L2-normalised, row i of each is a matched pair
    logits = logit_scale.exp() * img @ txt.t()
    targets = torch.arange(len(img), device=img.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) / 2


def sigmoid_loss(img, txt, logit_scale, bias):
    # SigLIP: every (i, j) pair is a binary decision, +1 on the diagonal, -1 elsewhere
    logits = logit_scale.exp() * img @ txt.t() + bias
    labels = 2 * torch.eye(len(img), device=img.device) - 1
    return -F.logsigmoid(labels * logits).sum() / len(img)
