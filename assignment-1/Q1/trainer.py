import torch
import wandb
from tqdm import tqdm
import numpy as np

class Trainer(object):

    def __init__(self, model, args):
        self.model = model
        
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # for out of box peformance on gpu
            torch.backends.cudnn.benchmark = True

        self.model.to(self.device)

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, 
        #                                 momentum=args.momentum, weight_decay=args.weight_decay)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.epochs = args.epochs
        self.save_path = args.save_path
        self.args = args 

    def fit(self, tr_data, val_data, test_data):

        self.model_summary(self.model)

        # wandb.init(project=self.args.project, name=self.args.name, config=self.args.__dict__, save_code=True)

        try:
            tr_metric, val_metric = self.train(tr_data, val_data)
            print("Model training finished")
        
        except KeyboardInterrupt:    
            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
                }
            torch.save(state_dict, f"{self.save_path}.tar")

        test_loss, test_acc = self.evaluate(test_data)

        print("saving model")
        if self.save_path:
            state_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                    }
            torch.save(state_dict, f"{self.save_path}.tar")

        self.display_metrics(tr_metric, val_metric)

        print(f"Test data- (Loss, Accuracy): ({test_loss}, {test_acc*100})")
        wandb.log({
            "test_acc": test_acc,
            "test_loss": test_loss
        })

    def train(self, tr_data, val_data):

        tr_metric = []
        val_metric = []

        for e in range(self.epochs):
            running_loss = 0
            running_acc = 0
            
            self.model.train()

            pbar = tqdm(enumerate(tr_data), total=len(tr_data), desc=f"running epoch-{e}", leave=False)
            for batch_idx, batch in pbar:
                loss, batch_acc = self.training_step(batch, batch_idx)
                # logging loss in progress-bar
                pbar.set_postfix(loss=loss.item(), accuracy=batch_acc*100)
                running_loss += loss.item()
                running_acc += batch_acc

                wandb.log({"step_tr_loss": loss.item()}, commit=True)

            tr_loss = running_loss/(batch_idx+1)
            tr_acc = running_acc/(batch_idx+1)
            val_loss, val_acc = self.evaluate(val_data)

            tr_metric.append(tr_acc*100)
            val_metric.append(val_acc*100)

            wandb.log({"tr_loss": tr_loss,
                      "val_loss": val_loss,
                      "tr_acc": tr_acc,
                      "val_acc": val_acc,
                      "epoch": e}, commit = False)

            if self.save_path:
                save_status = self.assert_epoch_saving(val_metric, mode="max")
                if save_status:
                    torch.save(self.model.state_dict(), f"epoch_wts/epoch-{e}.pt")

        return tr_metric, val_metric

    def evaluate(self, val_data, display_bar=True):
        self.model.eval()
        running_loss = 0
        running_acc = 0
        steps = 0
        pbar = tqdm(val_data, total=len(val_data), desc="Validating .. ", leave=False) if display_bar else val_data
        for batch in pbar:
            loss, batch_acc = self.validation_step(batch)
            pbar.set_postfix(loss=loss.item(), accuracy=batch_acc*100)
            running_loss += loss.item()
            running_acc += batch_acc
            steps += 1
        return running_loss/steps, running_acc/steps

    def training_step(self, batch, batch_idx):

        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        out = self.model(inputs)
        loss = self.criterion(out, labels)
        loss = loss.mean()

        loss.backward()
        
        self.optimizer.step()
        for p in self.model.parameters():
            p.grad = None

        _, pred = torch.max(out.detach(), 1)

        batch_acc = ((pred == labels).type(torch.float)).mean()

        return loss, batch_acc.item()

    def validation_step(self, batch):

        with torch.no_grad():
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            out = self.model(inputs)
            loss = self.criterion(out, labels)
            loss = loss.mean()

            _, pred = torch.max(out.detach(), 1)
            batch_acc = ((pred == labels).type(torch.float)).mean()

        return loss, batch_acc.item()

    def display_metrics(self, tr_metrics: list, val_metrics: list):
        # round factors should be 3 for proper layout

        results = """
                    |--------------------------------------------|
                        epoch   |   tr_metric   |   val_metric   
                    |--------------------------------------------|"""

        for e, tr, val in zip(range(self.epochs), tr_metrics, val_metrics):
            res = """
                          {}     |     {}     |      {}     
                    |--------------------------------------------|""".format(
                        np.round(e, 3), np.round(tr, 3), np.round(val, 3)
                        )
            results += res
        print(results)

    def model_summary(self, model: torch.nn.Module):

        num = np.sum([p.nelement() for p in model.parameters()])

        s = {"Net": num}
        for n, layer in model.named_children():
            num = np.sum([p.nelement() for p in layer.parameters()])
            s.update({n: num})

        print("Layers | Parameters")
        for l, p in s.items():
            print("{} | {}".format(l, p))

    @staticmethod
    def assert_epoch_saving(val_metric: list, n: int = 3, mode: str = "min"):
        """
        Allows saving if loss decreases / accuracy increases
        n = 'min' corresponds to val_metric being loss-metric
        n = 'max' corresponds to val_metric being accuracy-metric

        Note:
            val_metric should be having current value of loss/accuracy
        """
        status = False
        if len(val_metric) < n+1:
            return True

        current_val = val_metric[-1]
        compr = val_metric[-n-2:-2]
        if mode == "min":
            compr = np.min(compr)
            if current_val < compr:
                status = True
        elif mode == "max":
            compr = np.max(compr)
            if current_val > compr:
                status = True
        else:
            raise ValueError("mode can be only either max or min")
        return status
