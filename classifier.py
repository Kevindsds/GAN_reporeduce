"""
The CLassifier Model

The structure is borrowed from the following papers:

Reference: N. D. Truong, L. Kuhlmann, M. R. Bonyadi, D. Querlioz, L. Zhou and O. Kavehei, "Epileptic Seizure
Forecasting With Generative Adversarial Networks," in IEEE Access, vol. 7, pp. 143999-144009, 2019,
doi: 10.1109/ACCESS.2019.2944691.
https://ieeexplore.ieee.org/document/8853232

K. Rasheed, J. Qadir, T. J. O’Brien, L. Kuhlmann and A. Razi, "A Generative Model to Synthesize EEG Data for
Epileptic Seizure Prediction," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 29,
pp. 2322-2332, 2021, doi: 10.1109/TNSRE.2021.3125023.
https://ieeexplore.ieee.org/document/9599660

"""
import torch
from torch import nn
import numpy as np
import sklearn.metrics as metrics

from data_loader import ALL_STATES

from tqdm import tqdm
import datasets as datasets

import torch
import datasets
from gan_model import DCGAN_i
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from refiner import SyntheticDataset, R_Train
import os
import shutil

class Model(nn.Module):
    """
    Quote from the original paper about the structure of Classifier Model:
        Features extracted by the three convolution blocks of the Discriminator are flattened and connected to a neural
        network consisting of 2 fully-connected layers with the output sizes 256 and 2, respectively. The former fully-
        connected layer uses sigmoid activation function while the latter uses soft-max activation function. Both of the
        two layers have drop-out rate of 0.5. Note that the two-layer neural network can be replaced with any other binary
        classifier.
    """

    def __init__(self, input_channels, num_classes):
        super(Model, self).__init__()
        self.total_num_feature = input_channels

        # self.input_layer = nn.Sequential(
        #     # input shape 3x32x128
        #     nn.ZeroPad2d((1, 2, 1, 2)),
        #     nn.Conv2d(input_channels, self.total_num_feature, 5, 2, 0, bias=False),
        # )
        #
        # self.main_layer = nn.Sequential(
        #     # input shape 16x16x64
        #     nn.ZeroPad2d((1, 2, 1, 2)),
        #     nn.Conv2d(self.total_num_feature, self.total_num_feature * 2, 5, 2, 0, bias=False),
        #     # output shape 32x8x32
        #     nn.ZeroPad2d((1, 2, 1, 2)),
        #     nn.Conv2d(self.total_num_feature * 2, self.total_num_feature * 4, 5, 2, 0, bias=False),
        #     # output shape 64x4x16
        #     nn.Flatten()
        # )
        #
        # self.output_layer = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(self.total_num_feature * 4 * 4 * 16, 256),
        #     nn.Sigmoid(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, num_classes),
        #     nn.Softmax(dim=1),
        # )

        self.input_layer = nn.Sequential(
            # input shape 3x32x128
            nn.Conv2d(input_channels, self.total_num_feature, 3, 1, 1, bias=False),
            # output shape 16x32x128
            nn.MaxPool2d(2),
            # output shape 16x16x64
        )

        self.main_layer = nn.Sequential(
            # input shape 16x16x64
            nn.Conv2d(self.total_num_feature, self.total_num_feature * 2, 3, 1, 1, bias=False),
            # output shape 32x16x64
            nn.MaxPool2d(2),
            # output shape 32x8x32
            nn.Conv2d(self.total_num_feature * 2, self.total_num_feature * 4, 3, 1, 1, bias=False),
            # output shape 64x8x32
            nn.MaxPool2d(2),
            # output shape 64x4x16
            nn.Flatten()
        )

        self.output_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.total_num_feature * 4 * 4 * 16, 128),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, data):
        h1 = self.input_layer(data)
        h2 = self.main_layer(h1)
        out = self.output_layer(h2)
        return out


class TestMetric:

    def __init__(self, probabilities, labels, num_classes, pos_label):
        self.test_probabilities = probabilities
        self.test_labels = labels
        self.class_num = num_classes
        self.positive_class = pos_label
        # print(probabilities)
        # print(labels)

    def predict_classes(self):
        """Return the predicted class of the given probabilities"""
        return np.argmax(self.test_probabilities, axis=1)

    def correct_by_class(self):
        """Return number of correctly predicted of each class"""
        total_correct = np.zeros(self.class_num)

        predictions = self.predict_classes()
        for state in range(self.class_num):
            correctness = predictions[(predictions == state) & (self.test_labels == state)]
            total_correct[state] += len(correctness)

        return total_correct

    def accuracy_by_class(self):
        """Return the accuracy of the prediction of each class"""
        total_correct = np.zeros(self.class_num)
        total = np.zeros(self.class_num)

        predictions = self.predict_classes()
        for state in range(self.class_num):
            correctness = predictions[(predictions == state) & (self.test_labels == state)]
            state_total = np.where(self.test_labels == state, 1, 0)
            total_correct[state] += len(correctness)
            total[state] += np.sum(state_total)

        return total_correct / total

    def accuracy_total(self):
        """Return the total accuracy of the prediction"""
        predictions = self.predict_classes()
        correctness = np.sum(np.where(self.test_labels == predictions, 1, 0))
        return correctness / len(predictions)

    def true_positive(self):
        """Number of subjects that are predicted positive and actually positive"""
        predictions = self.predict_classes()
        data = predictions[(predictions == self.positive_class) & (self.test_labels == self.positive_class)]
        return len(data)

    def true_negative(self):
        """Number of subjects that are predicted negative and actually negative"""
        predictions = self.predict_classes()
        data = predictions[(predictions != self.positive_class) & (self.test_labels != self.positive_class)]
        return len(data)

    def false_positive(self):
        """Number of subjects that are predicted positive but actually negative"""
        predictions = self.predict_classes()
        data = predictions[(predictions == self.positive_class) & (self.test_labels != self.positive_class)]
        return len(data)

    def false_negative(self):
        """Number of subjects that are predicted negative but actually positive"""
        predictions = self.predict_classes()
        data = predictions[(predictions != self.positive_class) & (self.test_labels == self.positive_class)]
        return len(data)

    def sensitivity(self):
        """Out of all the people that have the disease, how many got positive test results?"""
        tp = self.true_positive()
        fn = self.false_negative()
        return tp / (tp + fn)

    def specificity(self):
        """Out of all the people that do not have the disease, how many got negative results?"""
        tn = self.true_negative()
        fp = self.false_positive()
        return tn / (fp + tn)

    def precision(self):
        """Out of all the examples that predicted as positive, how many are really positive?"""
        tp = self.true_positive()
        fp = self.false_positive()
        return tp / (fp + fp)

    def recall(self):
        """Out of all the positive examples, how many are predicted as positive?"""
        return self.sensitivity()

    def roc_curve(self):
        pos_class = self.positive_class
        pred_proba = self.test_probabilities[:, pos_class]
        real_labels = self.test_labels
        fpr, tpr, threshold = metrics.roc_curve(real_labels, pred_proba, pos_label=pos_class)
        return fpr, tpr, threshold

    def roc_auc_score(self):
        """Return the AUC score of ROC curve"""
        fpr, tpr, threshold = self.roc_curve()
        return metrics.auc(fpr, tpr)

    # def f1_score(self):
    #     pos_class = self.positive_class
    #     pred_proba = self.test_probabilities[:, pos_class]
    #     real_labels = self.test_labels
    #     return metrics.f1_score(real_labels, pred_proba, pos_label=pos_class)


class Classifier:
    def __init__(self, data_channels, num_classes, device='cpu'):
        self.device = torch.device(device)
        self.model = Model(data_channels, num_classes)

        self.label_dim = num_classes
        # an array of <num_classes> one hot vectors
        self.onehot = nn.functional.one_hot(torch.arange(self.label_dim)).type(torch.float32)

    def train(self, train_data, num_epochs, learning_rate, betas=(0.5, 0.999), logging_on=True):

        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=betas)
        loss_criterion = nn.CrossEntropyLoss()

        losses = []
        for ep in range(num_epochs):
            with tqdm(train_data, unit='batch', leave=False, desc=f"Epoch {ep}") as tepoch:
                for i, (examples, labels) in enumerate(tepoch):
                    labels = labels.type(torch.long)
                    examples = examples.type(torch.float32)

                    prediction = self.model(examples.to(self.device))
                    actual = self.onehot[labels]
                    loss = loss_criterion(prediction, actual.to(self.device))
                    self.model.zero_grad()
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.detach().cpu())
                    tepoch.set_postfix_str(f"loss {loss.detach().cpu().item()}")


                    # if logging_on:
                    #     print(
                    #         f"Epoch {ep + 1}/{num_epochs} | Batch {i + 1}/{len(train_data)} :\tLoss: {loss}")
        # if logging_on:
        #     print('Done.')
        return losses

    def get_test_metric(self, test_data):
        """
        Return a TestMetric object that contains the predicted probability of each class and true labels of each
        data point.
        """
        self.model.to(self.device)
        self.model.eval()

        all_proba = []
        all_labels = []
        for i, (examples, labels) in enumerate(test_data):
            labels = labels.type(torch.long)
            examples = examples.type(torch.float32)

            probabilities = self.model(examples.to(self.device)).detach()
            all_proba.append(probabilities.cpu())
            all_labels.append(labels.cpu())

        all_proba = torch.cat(all_proba)
        all_labels = torch.cat(all_labels)
        return TestMetric(all_proba.numpy(), all_labels.numpy(), self.label_dim, ALL_STATES["PREICTAL_0"])

    def acc_total(self, test_data):
        """Return the accuracy of the prediction on the given test data"""
        metric = self.get_test_metric(test_data)
        acc = metric.accuracy_total()
        return acc

    def acc_by_class(self, test_data):
        """Return the accuracy of the prediction on the given test data of each class"""
        metric = self.get_test_metric(test_data)
        acc_by_class = metric.accuracy_by_class()

        result = {}
        for name, val in ALL_STATES.items():
            result[name] = acc_by_class[val]
        return result

    def calculate_roc_auc_score(self, test_data):
        """Return the AUC score of ROC curve"""
        metric = self.get_test_metric(test_data)
        return metric.roc_auc_score()

    def calculate_sensitivity(self, test_data):
        metric = self.get_test_metric(test_data)
        return metric.sensitivity()

    def calculate_specificity(self, test_data):
        metric = self.get_test_metric(test_data)
        return metric.specificity()

if __name__ == "__main__":
    mapping = {
        'preictal_0': 0,
        'interictal': 1,
    }

    batch_size = 64

    chbmit = datasets.ChbmitFolder("/home/tian/DSI/datasets/CHB-MIT-log", "train", mapping, True)
    chb_dataset = datasets.EEGDataset(chbmit.get_patient_data('chb01'))
    chb_dataloader = DataLoader(chb_dataset, batch_size=64, shuffle=True)
    dcgan = DCGAN_i(22, 2)
    dcgan.train(chb_dataloader, 1, 0.1)

    gan_dataset = datasets.GANDataset(dcgan, 100)
    gan_loader = DataLoader(gan_dataset, batch_size=64)

    rtrain = R_Train(22,128,32, chb_dataloader, None, 0.0001, device='cuda', DCGAN=dcgan)
    rtrain.train(10, 2,1,0.001)
    refiner_Dataset = datasets.GANDataset(rtrain, 1000)
    refiner_loader = DataLoader(refiner_Dataset, batch_size=64)


    def get_classifier():
        return Classifier(22, num_classes=len(mapping))

    classifier = get_classifier()
    classifier.train(refiner_loader, 100,0.001)
