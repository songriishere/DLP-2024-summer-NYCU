# implement your testing script here
import torch
import torch.nn as nn
from model.SCCNet import *
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from Dataloader import *
from torch.optim import lr_scheduler

def test(model , test_dataset , device , batch):
    loss_function = nn.CrossEntropyLoss()
    model.eval()
    test_loss=0
    test_accuracy=0.0
    with torch.no_grad():
        for _, (test_data, test_label) in enumerate(test_dataset):
            test_data = np.expand_dims(test_data, axis=0)
            test_data = np.transpose(test_data, (1, 0, 2, 3))
            test_data = torch.from_numpy(test_data).float().to(device)
            test_label = torch.tensor(test_label).long().to(device)
            output = model(test_data).float()
            loss=loss_function(output,test_label)
            test_loss+=loss.item()
            _,preds = torch.max(output,dim = 1)
            test_accuracy+=int(torch.sum(preds==test_label))
    return test_accuracy /len(test_dataset) / batch , test_loss / batch

def test_choose_model(batch , Nu , method , device , epochs, test_dataset):
    best = 0
    test_acc_compare = []
    for epoch in range(1 , epochs+1):
        scc_test = SCCNet(numClasses=4, timeSample=438, Nu=Nu ,C=22, Nt=1, dropoutRate=0.5)
        scc_test.to(device)
        checkpoint = torch.load("./save_model/"+str(batch)+"_"+ str(Nu)+"_"+str(method) +"_" + str(epoch) + ".pt")
        scc_test.load_state_dict(checkpoint)
        test_accuracy ,_ = test(model=scc_test , test_dataset = test_dataset , device = device , batch = batch)
        test_acc_compare.append(test_accuracy)
    test_acc_compare =np.array(test_acc_compare)
    best_index = np.argmax(test_acc_compare)
    os.rename("./save_model/"+str(batch)+"_"+ str(Nu)+"_"+str(method) +"_" + str(best_index+1) + ".pt" ,"./save_model/"+str(batch)+"_"+ str(Nu)+"_"+str(method) +"_" + str(best_index+1) +"_FINAL" + ".pt")

    return test_acc_compare[best_index] , best_index+1


        
