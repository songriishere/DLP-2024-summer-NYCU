# def dice_score(pred_mask, gt_mask):
#     # implement the Dice score here
    
#     assert False, "Not implemented yet!"

import torch
from tqdm import tqdm
import os
import numpy as np
from PIL import Image


def dice_score(pred_mask, mask):
    """
    計算 Dice Score。

    :param pred_mask: 模型預測的概率或二值分割圖 (tensor, shape: [batch_size, 1, height, width])
    :param mask: 真實標籤 (tensor, shape: [batch_size, 1, height, width])
    :param threshold: 用於將概率轉換為二值分割圖的閾值
    :return: Dice Score
    """
    with torch.no_grad():
        result = 0
        for idx in range(pred_mask.shape[0]):
            intersection = torch.sum(pred_mask[idx] * mask[idx])
            result += 2. * intersection / (torch.sum(mask[idx]) + torch.sum(pred_mask[idx]))
        return ( result / pred_mask.shape[0] ).item()
    
    #Dice score = 2 * (number of common pixels) / (predicted img size + groud truth img size)
    # 將預測概率轉換為二值分割圖

    # # 計算交集和並集
    # intersection = torch.sum(pred_mask * mask)
    # union = torch.sum(pred_mask) + torch.sum(mask)

    # # 避免除以零的情況
    # if union == 0:
    #     return torch.tensor(1.0)  # 完全沒有前景區域的情況下，Dice Score 設置為 1

    # return 2. * intersection / union


def output_img(model, test_path, device, data_path, model_name):
    # 讀取文件名列表
    with open(test_path, 'r') as file:
        filenames = [line.split(' ')[0].strip() for line in file.readlines()]

    # 準備輸出目錄
    output_dir = os.path.join('outputs_imgs', model_name)
    os.makedirs(output_dir, exist_ok=True)

    # 處理每個文件
    for filename in tqdm(filenames):
        img_path = os.path.join(data_path, 'images', f'{filename}.jpg')
        
        image = Image.open(img_path).convert("RGB")
        image = image.resize((256, 256), Image.BILINEAR)
        image_np = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
        
        mask = model(image_tensor).cpu().detach().numpy().reshape(256, 256)
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        image_np = (image_np * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        mask_pil = Image.fromarray(np.stack([mask]*3, axis=-1))
 
        blended_image = Image.blend(image_pil, mask_pil, alpha=0.5)
        
        output_path = os.path.join(output_dir, f'{filename}_mask.png')
        blended_image.save(output_path)