"""
Assigment-1 Part-B

[2.1] Occlusion Sensitivity Experiment

This file is having relevant code for 2.1

Run this script: 
    `python occlusion.py`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import time
from tqdm import tqdm
import wandb

class Occlusion(object):

    def __init__(self, images, window_size=(5,5), stride=(1,1)):
        """
        Use this class for occluding image with gray-pixels (i.e=128)
        Args:
            images: images to occlude (bz, c, h, w)
            window_size: how large window to occulude
            stride: stride to follow while building filters

        All tensor based computations will be automatically happen on GPU, if its available
        """

        self.images = images
        self.window_size = window_size
        self.stride = stride
        
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        
        self.bz = images.size(0)
        self.channel = images.size(1)
        self.height = images.size(2)
        self.width = images.size(3)

    def prepare_filters(self, x_centres: list, y_centres: list) -> torch.Tensor:
        """
        This will generate filters for all centres- (x_centres, y_centres)

        filters are such that 0 will be at occluded portion & 1 in remaining region
        """

        h = int((self.window_size[0]-1)/2)
        w = int((self.window_size[1]-1)/2)

        filters = torch.zeros(self.bz, 1, self.channel, self.height, self.width).to(self.device)
        for cx, cy in zip(x_centres, y_centres):

            x_right = cx+w+1
            y_right = cy+h+1
            y_left = 0 if cy-h < 0 else cy-h
            x_left = 0 if cx-w < 0  else cx-w

            filter = torch.ones(self.bz, 1, self.channel, self.height, self.width).to(self.device)
            filter[:, :, :, y_left:y_right, x_left:x_right] = 0
            filters = torch.cat([filters, filter], dim=1)
        filters = filters[:, 1:, :, :]
        # -> (bz, num_filters, channels, height, width)

        print("filters are prepared")
        return filters

    def _occlude_images(self, x_centres: list, y_centres: list) -> torch.Tensor:

        images = self.images.unsqueeze(1).to(self.device)
        # -> bz, 1, c, h, w
        filters = self.prepare_filters(x_centres, y_centres)
        filters = filters.to(self.device)
        # -> bz, num_filters, c, h, w

        gray_val = torch.tensor(128).to(self.device)
        neg_filters = torch.logical_not(filters).type(torch.float)

        return (images*filters) + (neg_filters*gray_val)

    @property
    def occluded_images(self) -> torch.Tensor:

        # accumulating all the possible centres
        x_centres = []
        y_centres = []
        for i in range(0, self.width, self.stride[0]):
            for j in range(0, self.height, self.stride[1]):
                x_centres.append(i)
                y_centres.append(j)

        start = time.time()
        occ_images = self._occlude_images(x_centres, y_centres)

        print(f"time taken in occluding images- {time.time()-start}")
        # it will take just 20-30 sec on colab gpu for processing occlusions for 1 image

        return occ_images # -> bz, num_filters, c, h, w

    @staticmethod
    def show_image(img: torch.Tensor):
        transform = T.ToPILImage()
        img = img.to(torch.device("cpu"))
        img = transform(img)
        img.show()


class Predictor(object):

    def __init__(self, model: nn.Module, batch_size: int, wandb_name="occlusion-test", wandb_project="CS6910-assignment1"):
        
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        
        self.model = model

        state_dict = torch.load("final_modeling.pt", map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        wandb.init(name=wandb_name, project=wandb_project)

    def predict(self, occ_images: torch.Tensor, label: int):
        # this class is valid for only single image input but with multiple occulusions
        # occ_images -> (num_filters, 3, 84, 84)

        # for using dataloader, images must be on cpu
        occ_images = occ_images.to(torch.device("cpu"))
        dataset = torch.utils.data.DataLoader(occ_images,
                                            batch_size=self.batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=1)

        probs = []
        classes = []
        true_cls_probs = []

        for batch in tqdm(dataset):
            batch = batch.to(self.device)
            with torch.no_grad():
                out = self.model(batch)
                out = F.softmax(out.detach(), dim=1)
                pred_prob, pred = torch.max(out, 1)
                true_cls_prob = out[:, label]

            probs.extend(pred_prob.tolist())
            classes.extend(pred.tolist())
            true_cls_probs.extend(true_cls_prob.tolist())

        return probs, classes, true_cls_probs

    @torch.no_grad()
    def predict_orig(self, image: torch.Tensor, label: int):
        out = model(image.to(self.device))
        pred_prob, pred = torch.max(F.softmax(out, dim=1), 1)
        print(pred.item(), pred_prob.item())
        print(label)
        
        wandb.log({
          "Orig Image label": label,
          "Orig Image pred": pred.item(),
          "Orig Image prob": pred_prob.item()
        }, commit=True)
        return pred_prob.item()

    @staticmethod
    def log(true_cls_probs: list, probs: list, classes: list):

        for tp, p, c in zip(true_cls_probs, probs, classes):
            wandb.log({
                "true_cls_prob": tp,
                "pred_prob": p,
                "class": c
            }, commit=True)

        im = np.array(true_cls_probs)
        im = im.reshape(84, 84)
        sns.heatmap(im)
        plt.savefig(f"occ-test/occ-heatmap{n}")
        wandb.log({"Image heatmap": [wandb.Image(im)]})

        print("logged successfully")


if __name__ == "__main__":

    from net import Net

    model = Net()
    data = torchvision.datasets.ImageFolder(root="data/test", transform=T.ToTensor())
    
    nos = [505, 2200, 2856, 3200, 1600, 1700, 3000, 700, 2900, 8, 306]
    for n in [3200]:
        image, label = data[n] # taking some image

        T.ToPILImage()(image).save(f"occ-test/orig-{n}.jpg")

        image = image.unsqueeze(0)
        occ = Occlusion(image, window_size=(20,20), stride=(1,1))
        img = occ.occluded_images.squeeze(0)

        predictor = Predictor(model, batch_size=128, wandb_name=f"occ-test-{20,20}")
        orig_prob = predictor.predict_orig(image, label)
        probs, classes, true_cls_probs = predictor.predict(img, label=label)
        predictor.log(true_cls_probs, probs, classes)

        labels = [i for i in range(33)]
        y_true = [label for i in range(len(probs))]
        wandb.sklearn.plot_confusion_matrix(y_true, classes, labels)
