import os
import random
import numpy as np
from os import path
import random
import math
import torch

from torch.utils.data import Dataset

class EEGDataset(Dataset):
    """
    The pytorch custom dataset class, used to represent the CHB-MIT data
    """

    def __init__(self, data, process_function=None):
        """
        :param data: an array of tuple, each is of form  (path_to_eeg_file, label)
        :param process_function: How we want to post process the data
        """
        self.data = data
        self.process_function = process_function

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath, label = self.data[index]
        mtx = np.load(fpath)
        if self.process_function:
            mtx = self.process_function(mtx)
        return mtx, label

class GANDataset(Dataset):
    def __init__(self, model, length):
        self.model = model
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        condition = np.squeeze(np.random.randint(2,size=1))
        return torch.squeeze(self.model.gen_synthetic(1, condition)), condition

class ChbmitFolder:
    """
    A class returning the chbmit info needed to put into the EEG dataset, ie: each function will return an array of tuple,
    of format (path, label)
    """
    def __init__(self, root_path, dtype, mapping, even=None):
        """
        :param root_path: The path to this Chbmit data folder
        :param mapping: A dictionary, with key as the state string, and value as the 0 / 1 for the state reprersentation
         EG:
        {
        'PREICTAL_0': 0,
        'INTERICTAL': 1,
        }
        :param even: get each state evenly (with same amount of datapoints), if even = 1, or
        multiple, representing the max amount of data the category can have
        """
        assert dtype in ["train", "test"], "dtype can only be train or test"
        self.root_path = root_path
        self.dtype = dtype
        self.mapping = mapping
        self.even = even

    def set_even(self, even):
        """
        set up if we want to generate evenly split data for each class
        :param even:
        :return:
        """
        self.even = even

    def set_dtype(self, dtype):
        """
        set up if we want to generate train or test data
        :param dtype:
        :return:
        """
        self.dtype = dtype
    def get_patient_data(self, patient_name):
        """
        Given the patient name under the chbmit folder, return the patient corresponding data, with set parameters
        :param patient_name: the name of the patients
        :return:  an array of tuple, each is of form  (path_to_eeg_file, label)
        """
        assert patient_name in os.listdir(self.root_path), "wrong patient name"
        acc = []
        root_path = path.join(self.root_path, patient_name)
        data_path = path.join(root_path, self.dtype)
        min_size = math.inf

        # get the even split number
        for state, val in self.mapping.items():
            seizure_path = path.join(data_path, state)
            min_size = min(len(os.listdir(seizure_path)), min_size)
        for state, val in self.mapping.items():
            seizure_path = path.join(data_path, state)
            tmp = [(path.join(seizure_path, f), val) for f in os.listdir(seizure_path)]
            if self.even is not None:
                # if we have to trim this type of data, shuffle to make a random selection
                random.shuffle(tmp)
                tmp = tmp[:min_size * self.even]
            acc.extend(tmp)
        return acc

    def get_all_data(self):
        acc = []
        for patient_name in os.listdir(self.root_path):
            acc.extend(self.get_patient_data(patient_name))
        return acc

if __name__ == "__main__":
    root_path = "/Users/kevin/Desktop/file_all/processed"
    mapping = {
    'preictal_1': 0,
    'interictal': 1,
    }
    chbmit_folder = ChbmitFolder(root_path, "train", mapping, True)
    a = chbmit_folder.get_patient_data("chb14")
    b = chbmit_folder.get_all_data()
    print(a)
    print(b)
    print("success")
