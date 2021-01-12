import torch
import wandb
from tqdm import tqdm


class Trainer(object):

    def __init__(self, model, args):
        self.model = model
        
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # for out of box peformance on gpu
            torch.backends.cudnn.benchmark = True

        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.epochs = args.epochs
        self.save_path = args.save_path
        self.training_id = args.training_id

    def fit(self, tr_data):

        try:
            self.train(tr_data)
            print("Model training finished")

        except KeyboardInterrupt:    
            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
                }
            torch.save(state_dict, f"{self.save_path}.tar")

        print("saving model")
        if self.save_path:
            state_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                    }
            torch.save(state_dict, f"{self.save_path}.tar")

    def train(self, tr_data):

        for e in range(self.epochs):
            running_loss = 0

            pbar = tqdm(enumerate(tr_data), total=len(tr_data), desc=f"running epoch-{e}", leave=False)
            for batch_idx, batch in pbar:
                self.model.train()
                loss = self.training_step(batch, batch_idx)
                pbar.set_postfix(loss=loss.item())
                running_loss += loss.item()
                wandb.log({"step_tr_loss": loss.item()}, commit=True)

            tr_loss = running_loss/(batch_idx+1)

            wandb.log({"tr_loss": tr_loss,
                      "epoch": e}, commit = False)

            if self.save_path:
                torch.save(self.model.state_dict(), f"{self.save_path}-epochs/epoch-{e}.pt")

    def evaluate(self, val_data, display_bar=True):
        self.model.eval()
        running_loss = 0
        steps = 0
        pbar = tqdm(val_data, total=len(val_data), desc="Validating .. ", leave=False) if display_bar else val_data
        for batch in pbar:
            loss = self.validation_step(batch)
            pbar.set_postfix(loss=loss.item())
            running_loss += loss.item()
            steps += 1
        return running_loss/steps

    def training_step(self, batch, batch_idx):

        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        if self.training_id == "cbow":
            out = self.model(inputs)
        elif self.training_id == "lstm_based":
            out = self.model(inputs, target=labels)
        elif self.training_id == "skip_gram":
            out = self.model(labels)
            labels = inputs
        
        loss = self.criterion(out, labels)
        loss = loss.mean()

        loss.backward()

        self.optimizer.step()
        for p in self.model.parameters():
            p.grad = None

        return loss

    @torch.no_grad()
    def validation_step(self, batch):

        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        out = self.model(inputs)
        loss = self.criterion(out, labels)
        loss = loss.mean()

        return loss
