import time
import pickle

import draccus
from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch

@dataclass
class DatasetCollectionConfig:
    dataset_path: str = None
    seed: int = 1
    save_path_probe: str = None
    save_path_report: str = None
    probe_name: str = None
    
    def __post_init__(self):
        if self.dataset_path is None:
            raise ValueError("`dataset_path` should be defined")
        
        if self.save_path_probe is None:
            self.save_path_probe = f"checkpoints/"
        
        if self.save_path_report is None:
            self.save_path_report = f"reports/"
        
        if self.probe_name is None:
            self.probe_name = f"box-world-probe__{int(time.time())}"

@draccus.wrap()
def train_probe(args: DatasetCollectionConfig):
    dataset = torch.load(args.dataset_path, weights_only=False)

    features_tensor, targets_tensor = dataset.tensors
    features_tensor = features_tensor.detach().cpu().numpy()
    targets_tensor = targets_tensor.detach().cpu().numpy()

    X_train, X_test, y_train, y_test = train_test_split(features_tensor, targets_tensor, test_size=0.25, random_state=args.seed, stratify=targets_tensor)

    probe = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
    )
    probe.fit(X_train, y_train)
    with open(f"{args.save_path_probe}{args.probe_name}.pkl", "wb") as f:
        pickle.dump(probe, f)

    y_pred = probe.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3, 5],
        target_names=[
            "move_up",
            "move_down",
            "move_left",
            "move_right",
            "never_visited",
        ],
        output_dict=True,
        zero_division=0,
    )

    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(f"{args.save_path_report}{args.probe_name}.csv", index=False)

if __name__ == "__main__":
    train_probe()