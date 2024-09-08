# implement your training script here
import torch
import torch.nn as nn
from model.SCCNet import *
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from Dataloader import *
from torch.optim import lr_scheduler
from tester import *
import os
import matplotlib.pyplot as plt
from utils import *


def train(model , method ,train_dataset , test_dataset , optimizer = "Adam" , batch = 16 , device = "cuda:0" , lr = 1e-2 , epochs = 150 , show_epoch = 10 , Nu = 6):
    os.makedirs('save_model', exist_ok=True)
    

    model = model.to(device) #cuda
    loss_function = nn.CrossEntropyLoss()
    if optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr , weight_decay = 0.0001)

    train_accuracy_plot = []
    test_accuracy_plot = []
    loss_all = []
    loss_all_test = []
    epoch_list = []
    
    for epoch in range(epochs):
        model.train()
        train_loss=0
        train_accuracy=0.0
        for _, (train_data, train_label) in enumerate(train_dataset):
            train_data = np.expand_dims(train_data, axis=0)
            train_data = np.transpose(train_data, (1, 0, 2, 3))
            train_data = torch.from_numpy(train_data).float().to(device)
            train_label = torch.tensor(train_label).long().to(device)
            optimizer.zero_grad()
            output=model(train_data).float()
            loss=loss_function(output,train_label)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            _,preds = torch.max(output,dim=1) #preds為索引值
            train_accuracy+=int(torch.sum(preds==train_label))
        loss_all.append(train_loss / batch)
        test_accuracy ,test_loss = test(model=model , test_dataset = test_dataset , device = device , batch = batch)
        loss_all_test.append(test_loss)

        train_accuracy_plot.append(train_accuracy/len(train_dataset) / batch)
        test_accuracy_plot.append(test_accuracy)
        
        if (epoch+1) % show_epoch == 0:
            print(f"Epoch {epoch+1}: train accuracy = {train_accuracy/len(train_dataset) / batch}")   
            print(f"Epoch {epoch+1}: test accuracy = {test_accuracy}")

        save_path = "save_model"
        save_path = os.path.join(save_path,str(batch)+"_"+ str(Nu)+"_"+str(method) +"_" + str(epoch+1) + ".pt" )
        torch.save(model.state_dict(),save_path)

        epoch_list.append(epoch+1)
        
    best_test_accuracy , bestepoch = test_choose_model(batch=batch , Nu=Nu , method=method , device=device , epochs=epochs ,test_dataset=test_dataset)

    return train_accuracy_plot , test_accuracy_plot , epoch_list , best_test_accuracy ,bestepoch , loss_all , loss_all_test



if __name__ == '__main__':
    # In[2]:
    import torch
    import numpy as np
    import torch.nn as nn
    from Dataloader import *
    from model.SCCNet import *
    from torch.utils.data.dataloader import *
    from trainer import *
    import matplotlib.pyplot as plt
    from utils import *

    # SCC_LOSO_TRAIN
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_test_accuracy_final = 0
    idx_final = 0
    batch = 96
    batch_for_finetune = batch
    Nu=88
    idx = 0

    for idx in range(9):
        if idx !=0 and idx!=4 :
            training_LOSO = MIBCI2aDataset(mode='train_LOSO',idx=idx)
            test_LOSO = MIBCI2aDataset(mode='test_LOSO' ,idx=idx)
            method = "LOSO_"+str(idx)
            scc_LOSO = SCCNet(numClasses=4, timeSample=438, Nu=Nu ,C=22, Nt=1, dropoutRate=0.5)
            train_dataset, test_dataset = create_dataset(device=device, train_data=training_LOSO.features, train_label=training_LOSO.labels, test_data=test_LOSO.features, test_label=test_LOSO.labels ,batch_size = batch)
            train_accuracy_plot ,test_accuracy_plot ,epoch_list ,best_test_accuracy, bestepoch , loss_all , loss_all_test= train(model = scc_LOSO , method= method ,train_dataset = train_dataset ,test_dataset = test_dataset, optimizer = "Adam" , batch = batch, device = "cuda:0" , lr = 0.001, epochs = 300 , show_epoch = 10 , Nu = Nu)
            drawplot(loss_all ,loss_all_test , train_accuracy_plot , test_accuracy_plot ,epoch_list)
            if(best_test_accuracy > best_test_accuracy_final ):
                best_test_accuracy_final = best_test_accuracy
                idx_final = idx

    print(f"best_test_accuracy_final = {best_test_accuracy_final} , the best model is subject without {idx_final}")


   

    # In[6]:

     # SCC_LOSO_TEST ACCURACY
    Nu = 88
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch = 96
    scc_LOSO_final = SCCNet(numClasses=4, timeSample=438, Nu=Nu ,C=22, Nt=1, dropoutRate=0.5)
    scc_LOSO_final.to(device)
    checkpoint = torch.load("./LOSO_FINAL_TEST/"+str(batch)+"_"+str(Nu)+"_LOSO"+"_"+str(8)+"_75_FINAL"+".pt")
    scc_LOSO_final.load_state_dict(checkpoint)

    final_test_data = np.load('./dataset/LOSO_test/features/s9E.npy')
    final_test_label = np.load('./dataset/LOSO_test/labels/s9E.npy')
    test_dataset = create_LOSO_final_test_dataset(device=device, test_data=final_test_data, test_label=final_test_label ,batch_size = batch)
    test_accuracy ,test_loss = test(model=scc_LOSO_final , test_dataset = test_dataset , device = device , batch = batch)
    print("FINAL LOSO TEST")
    print(test_accuracy)


   

    # In[2]:
    # SCC_SD_TRAIN 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_test_accuracy_final = []

    for idx in range(10):
        if idx !=0 and idx!=4 :
            training_SD = MIBCI2aDataset(mode='train_SD',idx=idx)
            test_SD = MIBCI2aDataset(mode='test_SD' ,idx=idx)
            batch = 96
            Nu=88
            method = "SD_SUBJECT_"+str(idx)
            scc_SD = SCCNet(numClasses=4, timeSample=438, Nu=Nu ,C=22, Nt=1, dropoutRate=0.5)
            train_dataset, test_dataset = create_dataset(device=device, train_data=training_SD.features, train_label=training_SD.labels, test_data=test_SD.features, test_label=test_SD.labels ,batch_size = batch)
            train_accuracy_plot ,test_accuracy_plot ,epoch_list ,best_test_accuracy, bestepoch , loss_all , loss_all_test= train(model = scc_SD , method= method ,train_dataset = train_dataset ,test_dataset = test_dataset, optimizer = "Adam" , batch = batch, device = "cuda:0" , lr = 0.001, epochs = 300 , show_epoch = 10 , Nu = Nu)
            drawplot(loss_all ,loss_all_test , train_accuracy_plot , test_accuracy_plot ,epoch_list)
            best_test_accuracy_final.append(best_test_accuracy)
    print(best_test_accuracy_final)
    best_test_accuracy_final =np.array(best_test_accuracy_final)
    print(best_test_accuracy_final.mean())


    

    # In[4]:

    # SD_FINAL_TEST
    # 0.7065972222222222

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_test_accuracy_final = []
    best_epoch = [203 , 271 , 220 , 260, 157 , 121 , 190 , 253]
    index = 0

    for idx in range(10):
        if idx !=0 and idx!=4 :
            training_SD = MIBCI2aDataset(mode='train_SD',idx=idx)
            test_SD = MIBCI2aDataset(mode='test_SD' ,idx=idx)
            batch = 96
            Nu=88
            scc_SD_final = SCCNet(numClasses=4, timeSample=438, Nu=Nu ,C=22, Nt=1, dropoutRate=0.5)
            scc_SD_final.to(device)
            checkpoint = torch.load("./SD_FINAL_TEST/"+str(batch)+"_"+str(Nu)+"_SD_SUBJECT_"+str(idx)+"_"+str(best_epoch[index])+"_FINAL"+".pt")
            index += 1 
            scc_SD_final.load_state_dict(checkpoint)
            _, test_dataset = create_dataset(device=device, train_data=training_SD.features, train_label=training_SD.labels, test_data=test_SD.features, test_label=test_SD.labels ,batch_size = batch)
            drawplot(loss_all ,loss_all_test , train_accuracy_plot , test_accuracy_plot ,epoch_list)
            test_accuracy ,test_loss = test(model=scc_SD_final , test_dataset = test_dataset , device = device , batch = batch)
            best_test_accuracy_final.append(test_accuracy)
    print(best_test_accuracy_final)
    best_test_accuracy_final =np.array(best_test_accuracy_final)
    print(best_test_accuracy_final.mean())


    

    # In[8]:

    # LOSO_Finetune_TRAIN
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_for_finetune = 96
    batch = 96
    # print(train.features.shape)
    Nu=88
    method = "LOSOFT"
    scc_LOSOFT = SCCNet(numClasses=4, timeSample=438, Nu=Nu ,C=22, Nt=1, dropoutRate=0.2)
    scc_LOSOFT.to(device)
    checkpoint = torch.load("./LOSO_FINAL_TEST/"+str(batch_for_finetune)+"_"+str(Nu)+"_LOSO"+"_"+str(8)+"_75_FINAL"+".pt")
    scc_LOSOFT.load_state_dict(checkpoint)

    training_LOSOFT = MIBCI2aDataset(mode='finetune')
    final_LOSOFT_data = np.load('./dataset/LOSO_test/features/s9E.npy')
    final_LOSOFT_label = np.load('./dataset/LOSO_test/labels/s9E.npy')


    train_dataset, test_dataset = create_dataset(device=device, train_data=training_LOSOFT.features, train_label=training_LOSOFT.labels, test_data=final_LOSOFT_data, test_label=final_LOSOFT_label ,batch_size = batch)
    train_accuracy_plot ,test_accuracy_plot ,epoch_list ,best_test_accuracy, bestepoch , loss_all , loss_all_test= train(model = scc_LOSOFT , method= method ,train_dataset = train_dataset ,test_dataset = test_dataset, optimizer = "Adam" , batch = batch, device = "cuda:0" , lr = 0.001 , epochs = 300 , show_epoch = 10 , Nu = Nu)
    print(best_test_accuracy)
    print(bestepoch+1)

    drawplot(loss_all ,loss_all_test , train_accuracy_plot , test_accuracy_plot ,epoch_list)


    

    # In[3]:

    # LOSO_FINETUNE_TEST
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch = 96
    Nu=88

    scc_LOSOFT = SCCNet(numClasses=4, timeSample=438, Nu=Nu ,C=22, Nt=1, dropoutRate=0.2)
    scc_LOSOFT.to(device)
    checkpoint = torch.load("./LOSOFT_FINAL_TEST/"+str(batch)+"_"+str(Nu)+"_LOSOFT_279_FINAL.pt")
    scc_LOSOFT.load_state_dict(checkpoint)

    training_LOSOFT = MIBCI2aDataset(mode='finetune')
    final_LOSOFT_data = np.load('./dataset/LOSO_test/features/s9E.npy')
    final_LOSOFT_label = np.load('./dataset/LOSO_test/labels/s9E.npy')

    _, test_dataset = create_dataset(device=device, train_data=training_LOSOFT.features, train_label=training_LOSOFT.labels, test_data=final_LOSOFT_data, test_label=final_LOSOFT_label ,batch_size = batch)
    test_accuracy ,test_loss = test(model=scc_LOSOFT , test_dataset = test_dataset , device = device , batch = batch)
    print("FINAL LOSO_FT TEST")
    print(f"{test_accuracy:.4f}")