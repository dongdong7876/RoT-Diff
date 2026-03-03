import torch
import os
import random
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class PSMSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, train_split, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data_val = pd.read_csv(data_path + '/train.csv')
        data_val = data_val.iloc[-(int)(len(data_val)* 0.05):, 1:].values
        data = pd.read_csv(data_path + '/PSM_test.csv')
        data_train = data.iloc[:(int)(len(data) * train_split), 1:].values
        data_test = data.iloc[(int)(len(data) * train_split):, 1:].values

        self.scaler.fit(data_train)
        self.train = self.scaler.transform(data_train)
        self.val = self.scaler.transform(data_val)
        self.test = self.scaler.transform(data_test)

        labels = pd.read_csv(data_path + '/PSM_label.csv').values[:, 1:]
        self.test_labels = labels[(int)(len(data) * train_split):]
        train_labels = labels[:(int)(len(data) * train_split)]

        print("train:", self.train.shape)
        print("train anomaly ratio:", np.mean(train_labels))
        print("val:", self.val.shape)
        print("test:", self.test.shape)
        print("test anomaly ratio:", np.mean(self.test_labels))


    def __len__(self):
        """
        Number of samples in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        else:
            return np.float32(self.val[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, train_split, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data_val = np.load(data_path + "/MSL_train.npy")
        data_val = data_val[-(int)(len(data_val)* 0.05):]
        data = pd.read_csv(data_path + '/MSL_test.csv')
        data_train = data.iloc[:(int)(len(data) * train_split), 1:].values
        data_test = data.iloc[(int)(len(data) * train_split):, 1:].values

        self.scaler.fit(data_train)
        self.train = self.scaler.transform(data_train)
        self.val = self.scaler.transform(data_val)
        self.test = self.scaler.transform(data_test)

        labels = pd.read_csv(data_path + '/MSL_label.csv').values[:, 1:]
        self.test_labels = labels[(int)(len(data) * train_split):]
        train_labels = labels[:(int)(len(data) * train_split)]

        print("train:", self.train.shape)
        print("train anomaly ratio:", np.mean(train_labels))
        print("val:", self.val.shape)
        print("test:", self.test.shape)
        print("test anomaly ratio:", np.mean(self.test_labels))

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        else:
            return np.float32(self.val[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



class SMDSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, train_split, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data_val = np.load(data_path + "/SMD_train.npy")
        data_val = data_val[-(int)(len(data_val)* 0.05):]
        data = pd.read_csv(data_path + '/SMD_test.csv')
        data_train = data.iloc[:(int)(len(data) * train_split), 1:].values
        data_test = data.iloc[(int)(len(data) * train_split):, 1:].values

        self.scaler.fit(data_train)
        self.train = self.scaler.transform(data_train)
        self.val = self.scaler.transform(data_val)
        self.test = self.scaler.transform(data_test)

        labels = pd.read_csv(data_path + '/SMD_label.csv').values[:, 1:]
        self.test_labels = labels[(int)(len(data) * train_split):]
        train_labels = labels[:(int)(len(data) * train_split)]

        print("train:", self.train.shape)
        print("train anomaly ratio:", np.mean(train_labels))
        print("val:", self.val.shape)
        print("test:", self.test.shape)
        print("test anomaly ratio:", np.mean(self.test_labels))

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        else:
            return np.float32(self.val[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class SWaTSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, train_split, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data_val = np.load(data_path + "/SWaT_train.npy", allow_pickle=True)
        data_val = data_val[-(int)(len(data_val)* 0.05):]
        data = pd.read_csv(data_path + '/SWaT_test.csv')
        data_train = data.iloc[:(int)(len(data) * train_split), 1:].values
        data_test = data.iloc[(int)(len(data) * train_split):, 1:].values

        self.scaler.fit(data_train)
        self.train = self.scaler.transform(data_train)
        self.val = self.scaler.transform(data_val)
        self.test = self.scaler.transform(data_test)

        labels = pd.read_csv(data_path + '/SWaT_label.csv').values[:, 1:]
        self.test_labels = labels[(int)(len(data) * train_split):]
        train_labels = labels[:(int)(len(data) * train_split)]

        print("train:", self.train.shape)
        print("train anomaly ratio:", np.mean(train_labels))
        print("val:", self.val.shape)
        print("test:", self.test.shape)
        print("test anomaly ratio:", np.mean(self.test_labels))

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        else:
            return np.float32(self.val[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class WADISegLoader(Dataset):
    def __init__(self, data_path, win_size, step, train_split, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data_val = pd.read_csv(data_path + '/train.csv', index_col=0)
        data_val = data_val.iloc[-(int)(len(data_val)* 0.05):, :-1]
        raw_data = pd.read_csv(data_path + '/test.csv', index_col=0)
        labels = raw_data.values[:, -1:]
        data = raw_data.values[:, :-1]
        data_train = data[:(int)(len(data) * train_split)]
        data_test = data[(int)(len(data) * train_split):]

        self.scaler.fit(data_train)
        self.train = self.scaler.transform(data_train)
        self.val = self.scaler.transform(data_val)
        self.test = self.scaler.transform(data_test)

        self.test_labels = labels[(int)(len(data) * train_split):]
        train_labels = labels[:(int)(len(data) * train_split)]

        print("train:", self.train.shape)
        print("train anomaly ratio:", np.mean(train_labels))
        print("val:", self.val.shape)
        print("test:", self.test.shape)
        print("test anomaly ratio:", np.mean(self.test_labels))

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        else:
            return np.float32(self.val[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



def get_loader_segment(data_path, batch_size, win_size=100, step=1, train_split=0.6, mode='train', num_workers=15, data_name='KDD'):
    if (data_name == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, train_split, mode)
    elif (data_name == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, step, train_split, mode)
    elif (data_name == 'SWaT'):
        dataset = SWaTSegLoader(data_path, win_size, step, train_split, mode)
    elif (data_name == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, step, train_split, mode)
    elif (data_name == 'WADI'):
        dataset = WADISegLoader(data_path, win_size, step, train_split, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)
    return data_loader


def get_dataset(data_path, win_size=100, step=100, mode='train', train_split=0.6, dataset='KDD'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, train_split, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, step, train_split, mode)
    elif (dataset == 'SWaT'):
        dataset = SWaTSegLoader(data_path, win_size, step, train_split, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, step, train_split, mode)
    elif (dataset == 'WADI'):
        dataset = WADISegLoader(data_path, win_size, step, train_split, mode)

    return dataset