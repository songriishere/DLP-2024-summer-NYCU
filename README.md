# Deep Learning and Practice (NYCU) - Lab1 to Lab5

---

## Lab1 - Back Propagation
**目標**:  
實作兩層 Hidden Layer 的神經網路，包含 Forward Propagation 和 Back Propagation，並在 Linear Data 和 XOR Data 進行訓練與測試。  

**實作內容**:
- 自訂神經網路架構
- `Sigmoid` / `ReLU` Activation Functions
- `SGD` / `AdaGrad` Optimizer
- 測試不同的 `learning rate` 和 `hidden layer size`
- 移除 Activation Function 進行比較

---

## Lab2 - EEG Motor Imagery Classification
**目標**:  
使用 SCCNet 模型對 EEG 訊號進行分類，並測試不同的訓練方法。  

**實作內容**:
- `SCCNet` 模型實作
- 訓練方法：
  - Subject Dependent (SD)
  - Leave-One-Subject-Out (LOSO)
  - LOSO + Fine-tuning
- `Adam` Optimizer，`CrossEntropyLoss`
- Hyperparameter 調整 (`batch_size`, `Nu`)

---

## Lab3 - Binary Semantic Segmentation
**目標**:  
使用 U-Net 和 ResNet34-Unet 進行 Oxford-IIIT Pet Dataset 的語意分割，並計算 `Dice Score` 進行評估。  

**實作內容**:
- `U-Net` 和 `ResNet34-Unet` 架構實作
- `Dice Loss` 計算
- `Adam` Optimizer
- Data Augmentation (Rotation, Flip)
- 訓練、驗證與測試流程實作

---

## Lab4 - Conditional VAE for Video Prediction
**目標**:  
訓練 Conditional VAE 進行未來影像預測，並使用 KL annealing 與 Teacher Forcing 優化訓練過程。  

**實作內容**:
- `Conditional VAE` 模型實作
- `Reparameterization Trick`
- `Teacher Forcing Strategy`
- `KL Annealing` 方法：
  - Without KL annealing
  - Monotonic KL
  - Cyclical KL
- 訓練、驗證與 PSNR 計算

---

## Lab5 - MaskGIT for Image Inpainting
**目標**:  
使用 MaskGIT 進行影像修補 (Image Inpainting)，探討不同 Mask Scheduling 的影響。  

**實作內容**:
- `Multi-Head Self-Attention` 模型實作
- `Bidirectional Transformer` 訓練
- `Masked Visual Token Modeling (MVTM)`
- `Iterative Decoding`
- `Mask Scheduling`:
  - Cosine
  - Linear
  - Square
- `FID Score` 計算
