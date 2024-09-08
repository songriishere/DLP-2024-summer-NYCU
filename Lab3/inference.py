import argparse
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from oxford_pet import *
from models.resnet34_unet import *
from models.unet import *
from utils import *
from evaluate import *

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--load_epoch', '-le' , type=int, default = 0, help='load model with training')
    parser.add_argument('--cp', '-c' , type=int, default = 0, help='checkpoint type')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(f'outputs_imgs/{args.model}', exist_ok=True)

    # read model
    model = UNet() if args.model == 'U' else ResNet34_UNet()
    
    model = model.to(device)
    print("Evaluate model")
    if(args.cp == 0): #checkpoint type中有存其他東西的時候
        if args.model == 'U':
            checkpoint = torch.load((f"saved_models/DL_Lab3_UNet_313551159_洪子奇.pth"))
            model.load_state_dict(checkpoint['model_state_dict'])
        else : 
            checkpoint = torch.load((f"saved_models/DL_Lab3_ ResNet34_UNet _313551159_洪子奇.pth"))
            model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = checkpoint['learning_rate']
        #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    else:
        checkpoint = torch.load((f"saved_models/{args.model}/{args.model}_epoch_{args.load_epoch}.pth"))
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loader = DataLoader(load_dataset(args.data_path, "test"), batch_size=args.batch_size, shuffle=False)
    dice = evaluate(model, test_loader , device)
    print(f"Dice Score: {dice:.6f}")


    test_path = os.path.join(args.data_path, 'annotations', 'test.txt')
    print("Output images now")
    output_img(model , test_path ,device ,args.data_path , args.model)
     