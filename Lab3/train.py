import argparse
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from oxford_pet import *
from models.resnet34_unet import *
from models.unet import *
from utils import *

def train(args,model):
    # implement the training function here
    
    save_path = f"saved_models/{args.model}"
    os.makedirs(save_path, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device) #cuda
    train_data = DataLoader(load_dataset(args.data_path, "train"), batch_size=args.batch_size, shuffle=True)
    validation_data = DataLoader(load_dataset(args.data_path , "valid") , batch_size=args.batch_size , shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99) #lr′ = lr * gamma^epoch
    loss_function = nn.CrossEntropyLoss()

    loss_total = []
    dice_score_total = []
    epoch_out = 0
    
    if args.load_epoch != 0:
        checkpoint = torch.load((f"saved_models/{args.model}/{args.model}_epoch_{args.load_epoch}.pth"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = checkpoint['learning_rate']
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for ex in tqdm(train_data):
            image = ex["image"].to(device)
            mask = ex["mask"].to(device)
            #print(mask.shape)
            optimizer.zero_grad()
            mask_pred = model(image)
            #print(mask_pred.shape)
            mask_pred = mask_pred.flatten(start_dim=1)
            loss = loss_function(mask_pred , mask)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss = train_loss / len(train_data)
        loss_total.append(train_loss)
        #透過model對驗證集測試dice_score

        model.eval()
        with torch.no_grad():
            dice_score_val = 0
            for ex in tqdm(validation_data):
                image = ex["image"].to(device)
                mask = ex["mask"].to(device)
                #optimizer.zero_grad()
                mask_pred = model(image)
                mask_pred = mask_pred > 0.5
                mask_pred = mask_pred.float().flatten(start_dim=1)
                dice_score_val += dice_score(pred_mask= mask_pred , mask = mask)
            dice_scores = dice_score_val / len(validation_data)
            if args.load_epoch != 0:
                print(f"Epoch {epoch + args.load_epoch + 1}, Train Loss: {train_loss:.6f}, Dice Score: {dice_scores:.6f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}") #, Learning Rate: {scheduler.get_last_lr()[0]:.6f}
            else:
                print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, Dice Score: {dice_scores:.6f} , Learning Rate: {scheduler.get_last_lr()[0]:.6f}") #, Learning Rate: {scheduler.get_last_lr()[0]:.6f}
            #更新Learning rate
            scheduler.step()
        dice_score_total.append(dice_scores)
        if (epoch+1) % 5 == 0:
            np.save(f"{save_path}/{args.model}_loss_{args.load_epoch + epoch+1}.npy", loss_total)
            np.save(f"{save_path}/{args.model}_dice_score_{args.load_epoch + epoch+1}.npy", dice_score_total)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'learning_rate': scheduler.get_last_lr()[0] 
            }, f"{save_path}/{args.model}_epoch_{args.load_epoch + epoch + 1}.pth")
        epoch_out = epoch

    np.save(f"{save_path}/{args.model}_loss_{args.load_epoch + epoch_out+1}.npy", loss_total)
    np.save(f"{save_path}/{args.model}_dice_score_{args.load_epoch + epoch_out+1}.npy", dice_score_total)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate': scheduler.get_last_lr()[0] 
    }, f"{save_path}/{args.model}_epoch_{args.load_epoch + epoch_out + 1}.pth")
    #torch.save(model.state_dict(), f"{save_path}/{args.model}_final_model.pth")
    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument("--model", type=str, default="U", help="U: UNet / R: ResNet_UNet")
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--load_epoch', '-le' , type=int, default = 0, help='load epoch with training')
    parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Now using : {device}")
    args = get_args()
    if args.model == "U" :
        model = UNet()
    else :
        model = ResNet34_UNet()
    train(args, model)
