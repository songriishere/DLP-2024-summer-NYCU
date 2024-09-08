import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.optim import lr_scheduler
from oxford_pet import *
from utils import *

def evaluate(net , data, device):
    # implement the evaluation function here
    net.eval()
    with torch.no_grad():
        dice_total = 0
        for ex in tqdm(data):
            image = ex["image"].to(device)
            mask = ex["mask"].to(device)
            mask_pred = net(image)
            mask_pred = mask_pred > 0.5
            mask_pred = mask_pred.float().flatten(start_dim=1)
            dice_total += dice_score(pred_mask = mask_pred, mask= mask)
        dice_sc = dice_total / len(data)
    return dice_sc