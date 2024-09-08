import torch
import os 
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

def create_dataset(device , train_data , train_label , test_data , test_label , batch_size = 16):
    device = device
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_label = torch.tensor(train_label, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_label = torch.tensor(test_label, dtype=torch.float32)

    train_dataset = TensorDataset( train_data , train_label )
    test_dataset = TensorDataset( test_data , test_label )

    train_dataset = DataLoader(train_dataset , batch_size = batch_size , shuffle = True)
    test_dataset = DataLoader(test_dataset  , batch_size = batch_size, shuffle = False) #
    
    return train_dataset , test_dataset

def create_LOSO_final_test_dataset(device , test_data , test_label , batch_size = 16):
    device = device
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_label = torch.tensor(test_label, dtype=torch.float32)
    test_dataset = TensorDataset( test_data , test_label )
    test_dataset = DataLoader(test_dataset  , batch_size = batch_size, shuffle = False)
    return test_dataset

class MIBCI2aDataset(torch.utils.data.Dataset):
    def _getFeatures(self, filePath):
        # implement the getFeatures method
        """
        read all the preprocessed data from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        # print(f"Reading features from: {filePath}")
        # if not os.path.exists(filePath):
        #     raise FileNotFoundError(f"The directory {filePath} does not exist.")
        feature_files = [os.path.join(filePath, f) for f in os.listdir(filePath) if f.endswith('.npy')]
        # if not feature_files:
        #     raise FileNotFoundError(f"No .npy files found in directory {filePath}.")
        # print(f"Found feature files: {feature_files}")
        features = [np.load(f) for f in feature_files]
        return np.concatenate(features, axis=0)

    def _getLabels(self, filePath):
        # implement the getLabels method
        """
        read all the preprocessed labels from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        label_files = [os.path.join(filePath, f) for f in os.listdir(filePath) if f.endswith('.npy')]
        labels = [np.load(f) for f in label_files]
        return np.concatenate(labels, axis=0)
        pass

    def __init__(self, mode , idx = 1):
        # remember to change the file path according to different experiments
        assert mode in ['train_LOSO', 'test_LOSO', 'finetune','train_SD' ,'test_SD']

        if mode == 'train_LOSO':
            features = []
            labels = []
            for i in range(9):
                if i!=0 and i!=4:
                    if i != idx :
                        features1 = np.load('./dataset/LOSO_train/features/s'+str(idx)+'T.npy')
                        labels1 = np.load('./dataset/LOSO_train/labels/s'+str(idx)+'T.npy')
                        features.append(features1)
                        labels.append(labels1)
            self.features = np.concatenate(features , axis = 0)
            self.labels = np.concatenate(labels,axis = 0)

        if mode == 'test_LOSO':
            features = []
            labels = []
            for i in range(9):
                if i!=0 and i!=4:
                    if i == idx :
                        features1 = np.load('./dataset/LOSO_train/features/s'+str(idx)+'E.npy')
                        labels1 = np.load('./dataset/LOSO_train/labels/s'+str(idx)+'E.npy')
                        features.append(features1)
                        labels.append(labels1)
            self.features = np.concatenate(features , axis = 0)
            self.labels = np.concatenate(labels,axis = 0)

        if mode == 'finetune':
            # finetune: ./dataset/FT/features/ and ./dataset/FT/labels/
            self.features = np.load("./dataset/FT/features/s9T.npy")
            self.labels = np.load("./dataset/FT/labels/s9T.npy")
        
        if mode =='train_SD':
            self.features = np.load('./dataset/SD_train/features/s'+str(idx)+'T.npy')
            self.labels = np.load('./dataset/SD_train/labels/s'+str(idx)+'T.npy')

        if mode == 'test_SD':
            self.features = np.load('./dataset/SD_test/features/s'+str(idx)+'E.npy')
            self.labels = np.load('./dataset/SD_test/labels/s'+str(idx)+'E.npy')
    
    
    def __len__(self):
        # implement the len method
        return len(self.features)
        pass

    def __getitem__(self, idx):
        # implement the getitem method
        return self.features[idx] , self.labels[idx]
        pass



# if __name__ == '__main__':
#     test_dataset = MIBCI2aDataset(mode='train')
#     print(test_dataset.features.shape)
    # test_f , test_l = test_dataset.__getitem__(idx = 0)
    # print(test_f[0].shape)
    # print(test_l)
    
