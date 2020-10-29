import wandb

# my modules
from dataloader import DataLoader
from trainer import Trainer
from net import Net
from config import Config


if __name__ == "__main__":

    args = Config()
    wandb.init(config=args.__dict__, project="CS6910-assignment1", name=args.name)
    args = wandb.config
    print(dict(args))

    # setting up dataset
    dl = DataLoader(args)
    dl.setup()
    tr_data = dl.train_dataloader()
    val_data = dl.val_dataloader()
    test_data = dl.test_dataloader()

    # define model
    net = Net()

    # define trainer
    trainer = Trainer(net, args)
    trainer.fit(tr_data, val_data, test_data)
