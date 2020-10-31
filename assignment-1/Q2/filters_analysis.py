"""
Assigment-1 Part-B

[2.2] Filter Analysis

This file is having relevant code for 2.2.1 & 2.2.2

Run this script: 
    `python filters_analysis.py`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb

import seaborn as sns
import matplotlib.pyplot as plt


class Analyser(object):

    def __init__(self, model, batch_size=8):
        """
        Use this class for getting response for best 5 images for 2 filters corresponding to a particular layer

        All tensor based computations will be automatically happen on GPU, if its available
        """

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.batch_size = batch_size

        self.model = model

        state_dict = torch.load("final_modeling.pt", map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def filter_identification(self, dataset: torch.utils.data.Dataset, layer_no, resp1_f_idx, resp2_f_idx):
        """
        This method saves responses of top-5 images corresponding to particular filter indices.

        args:
            dataset: dataset of images
            layer_no: filter belong to which layer
            resp1_f_idx: index of filter in layer-no
            resp2_f_idx: index of filter in layer-no
        """

        dataset = torch.utils.data.DataLoader(dataset,
                                            batch_size=self.batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=4)

        resp1 = []; resp2 = []

        for batch, labels in tqdm(dataset, desc=f"Running 2.2.1 for layer-{layer_no} for filters- {resp1_f_idx} & {resp2_f_idx} "):
            batch, labels = batch.to(self.device), labels.to(self.device)
            _, resp = self.model(batch, layer_no=layer_no)
            resp1.append(resp[:, resp1_f_idx])
            resp2.append(resp[:, resp2_f_idx])

        resp1 = torch.cat(resp1)
        l_indices = torch.topk(resp1.max(-1).values.max(-1).values, k=5).indices

        resp2 = torch.cat(resp2)
        a_indices = torch.topk(resp2.max(-1).values.max(-1).values, k=5).indices

        print("indices: \n", l_indices, a_indices)

        for n in l_indices:
            sns.heatmap(resp1[n.item()].to(torch.device("cpu")))
            plt.savefig(f"assets/filter-analysis/img-{n.item()}_resp1-layer{layer_no}")
            plt.show()

        for n in a_indices:
            sns.heatmap(resp2[n.item()].to(torch.device("cpu")))
            plt.savefig(f"assets/filter-analysis/img-{n.item()}_resp2-layer{layer_no}")
            plt.show()

        return l_indices, a_indices

    def evaluate(self, data, filter_idx, layer_no):

        running_zero_filter_acc = 0
        steps = 0
        predictions = []
        labels = []

        pbar = tqdm(data, total=len(data), desc=f"Running 2.2.2 for layer-{layer_no} filter-{filter_idx}", leave=False)
        for batch in pbar:

            inputs, label = batch
            inputs = inputs.to(self.device)
            label = label.to(self.device)

            zero_filter_acc, pred = self.validation_step(inputs, label)
            running_zero_filter_acc += zero_filter_acc
            steps += 1
            predictions.extend(pred.tolist())
            labels.extend(label.tolist())

        return running_zero_filter_acc/steps, torch.tensor(predictions), torch.tensor(labels)

    @torch.no_grad()
    def filter_modification(self, data, layer_no, filter_idx1, filter_idx2,  wandb_name="filter_modification", wandb_project="CS6910-assignment1"):

        data = torch.utils.data.DataLoader(data,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        pin_memory=True,
                                        num_workers=4)

        _, fake_labels, labels = self.evaluate(data, None, None) 
        comp = (fake_labels == labels)

        for idx in (filter_idx1, filter_idx2):
            wandb.init(name=f"{wandb_name}_layer{layer_no}_filter{idx}", project=wandb_project)
            self.model.load_state_dict(torch.load("final_modeling.pt", map_location=self.device))
            self.model.to(self.device)
            self.model.zero_filter_(idx, layer_no)
            acc, preds, labels = self.evaluate(data, idx, layer_no)
            wandb.log({"accuracy": acc})

            new_comp = (preds != labels)
            indices = (new_comp*comp).tolist()

            mis_classified_labels = labels[indices]
            freq = pd.Series(mis_classified_labels.tolist()).value_counts()

            print(f"mis_classified_labels_layer{layer_no}_filter{idx}", freq)

    @torch.no_grad()
    def validation_step(self, inputs, label):

        out = self.model(inputs)

        _, pred = torch.max(out.detach(), 1)
        batch_acc = ((pred == label).type(torch.float)).mean()

        return batch_acc.item(), pred


if __name__ == "__main__":

    from net import Net

    # just change the layer no - [1,2,3,4,5]
    layer_no = 1
    filter_idx1 = 16
    filter_idx2 = 32

    dataset = torchvision.datasets.ImageFolder(root="data/test", transform=T.ToTensor())

    model = Net()
    analyser = Analyser(model)

    # filter-identification
    l_indices, a_indices = analyser.filter_identification(dataset, layer_no, filter_idx1, filter_idx2)
    print(l_indices, a_indices)

    # [OUTPUT]:
    # tensor([3113, 1935,  206, 1991,  552], device='cuda:0') tensor([2897,  847, 1793, 2582, 3098], device='cuda:0') tensor([1725, 1169,  519, 2689, 1673], device='cuda:0') tensor([2331, 2352, 1592, 1521, 2759], device='cuda:0') tensor([2663, 2652, 2604, 1539, 1799], device='cuda:0') tensor([1515,  287, 1087, 1553, 3221], device='cuda:0') tensor([1109,  703, 3143, 1954,    5], device='cuda:0') tensor([1784, 2125, 2386,  930, 1425], device='cuda:0') tensor([2985, 1508, 2664, 1544, 1549], device='cuda:0') tensor([2834, 2808, 2835, 1450, 2248], device='cuda:0')

    for idx in torch.cat([l_indices, a_indices]):
        T.ToPILImage()(dataset[idx.item()][0]).save(f"assets/filter-analysis/img-{idx.item()}.jpg")

    # running filter modification
    analyser.filter_modification(dataset, layer_no, filter_idx1, filter_idx2)
