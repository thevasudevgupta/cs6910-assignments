from dataclasses import dataclass, replace

@dataclass
class Config:

    batch_size: int = 32
    epochs: int = 48

    lr: float = 5e-3
    momentum: float = 0.9
    weight_decay: float = 1e-3
    random_flip_prob: float = None

    tr_root: str = "data/train/"
    val_root: str = "data/val/"
    test_root: str = "data/test/"
    save_path: str = "final_modeling"

    # wandb logging related args
    project: str = "CS6910-assignment1"
    name: str = None
