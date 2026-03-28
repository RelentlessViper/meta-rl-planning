import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from torch.utils.data import random_split

from dataclasses import dataclass
from typing import Any, Optional, Dict
import draccus
from tqdm import trange
from sklearn.metrics import f1_score, classification_report
import wandb
import numpy as np
import pandas as pd
from tqdm import trange

@dataclass
class TrainConfig:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = False
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "rl2-darkroom-meta"
    save_model: bool = False
    dataset_paths: list[str] = None
    save_report: bool = False
    verbose: bool = False

    probe_parameters: Optional[Dict[str, Any]] = None
    loss_type: str = "cross_entropy"
    focal_gamma: float = 2
    test_frac: float = 0.2
    num_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    l1_lambda: float = 1e-5

    # Dark Room arguments
    room_size: int = 5

    def __post_init__(self):
        self.run_name = f"{self.exp_name}__{self.seed}__{int(time.time())}"
        if self.dataset_paths == None or not self.dataset_paths:
            raise ValueError(f"`dataset_paths` must be filled and have `len` >= 1, got: {self.dataset_paths}")

class ProbeDataset(Dataset):
    def __init__(self, path):
        hidden_states, _, _, grid_states = torch.load(path)
        self.hidden_states = hidden_states.float()
        self.grid_states = grid_states.long()

    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, idx):
        hidden = self.hidden_states[idx]
        grid = self.grid_states[idx]

        # flatten and map -1 (NEVER VISITED) to 5
        grid = grid.view(-1)
        grid = torch.where(grid == -1, torch.tensor(5), grid)

        return hidden, grid

class GridActionProbe(nn.Module):
    def __init__(self, hidden_dim=515, num_actions=6, grid_size=5):
        super().__init__()
        self.num_actions = num_actions
        self.grid_size = grid_size
        self.linear = nn.Linear(hidden_dim, grid_size ** 2 * num_actions)

    def forward(self, hidden_states):
        x = self.linear(hidden_states) # [B,5x5x6]
        return x.reshape((-1, self.grid_size ** 2, self.num_actions)) # [B,25,6]

def make_splits(dataset, test_frac=0.2):
    n = len(dataset)
    n_test = int(n * test_frac)
    n_train = n - n_test
    return random_split(dataset, [n_train, n_test])

class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma=2,
        alpha=None,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        probs = torch.exp(-ce)

        if self.alpha is not None:
            weights = self.alpha[targets]
            return (weights * (1 - probs) ** self.gamma * ce).mean()
        else:
            return ((1 - probs) ** self.gamma * ce).mean()

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, eps=1e-8):
        probs = torch.softmax(logits, dim=1)

        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=logits.shape[1]).float()

        probs = probs.reshape(-1, probs.shape[1])
        targets_onehot = targets_onehot.reshape(-1, targets_onehot.shape[1])

        intersection = (probs * targets_onehot).sum(dim=0)
        union = probs.sum(dim=0) + targets_onehot.sum(dim=0)

        dice = (2 * intersection + eps) / (union + eps)

        return 1 - dice.mean()

def train_probe(
    dataset_path,
    epochs,
    batch_size,
    lr,
    lambda_l1,
    probe_parameters=None,
    device="cpu",
    track=False,
    verbose=True,
    save_report=False,
    test_frac=0.2,
    room_size=5,
):
    # Load Data
    dataset = torch.load(dataset_path, weights_only=False)
    train_ds, test_ds = make_splits(dataset, test_frac=test_frac)

    if probe_parameters is not None:
        model = GridActionProbe(**probe_parameters).to(device)
    else:
        model = GridActionProbe(grid_size=room_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    reports = []
    for epoch in trange(epochs):
        model.train()
        total_loss = 0

        for hidden, target in train_loader:
            hidden = hidden.to(device)
            target = target.to(device)

            logits = model(hidden) # [B,25,6]
            B, _, A = logits.shape
            logits_flat = logits.reshape((-1, A)) # [Bx25,6]
            target_fixed = target.clone().reshape(-1).long() # [Bx25]
            ce_loss = criterion(logits_flat, target_fixed)

            # Add L1 penalty (to mimic L1-weight decay used in papers)
            l1_penalty = lambda_l1 * sum(
                p.abs().sum() for p in model.linear.parameters()
            )

            loss = ce_loss + l1_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for hidden, target in test_loader:
                hidden = hidden.to(device)
                target = target.to(device).reshape(-1) # [Bx25]

                logits = model(hidden) # [B,25,6]
                B, _, A = logits.shape
                logits_flat = logits.reshape((-1, A)) # [Bx25,6]

                preds = logits.argmax(dim=-1).reshape(-1) # [Bx25]

                all_preds.append(preds.cpu())
                all_targets.append(target.cpu().long())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        val_f1 = f1_score(all_targets, all_preds, average="weighted")
        if verbose and epoch % 3 == 0:
            print(classification_report(all_targets, all_preds))
        if save_report:
            report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
            reports.append(report)

        if track:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_f1_weighted": val_f1,
            })

    if save_report:
        return model, reports

    return model

def summarize_trial_reports(reports):

    # Collect all the "row keys" (class names + 'accuracy' + 'macro avg' etc)
    all_rows = list(reports[0].keys())

    rows = []

    for row_key in all_rows:
        if row_key not in ('0', '1', '2', '3', '4', '5'):
            continue

        precisions = [r[row_key]["precision"] for r in reports]
        recalls    = [r[row_key]["recall"] for r in reports]
        f1s        = [r[row_key]["f1-score"] for r in reports]
        supports   = [r[row_key]["support"] for r in reports]

        rows.append({
            "class": row_key,
            "precision": f"{np.round(np.mean(precisions) * 100, 2)} +/- {np.round(np.std(precisions) * 100, 2)}",
            "recall": f"{np.round(np.mean(recalls) * 100, 2)} +/- {np.round(np.std(recalls) * 100, 2)}",
            "f1": f"{np.round(np.mean(f1s) * 100, 2)} +/- {np.round(np.std(f1s) * 100, 2)}",
            "support": np.mean(supports, dtype=np.int32),
        })

    df = pd.DataFrame(rows)

    return df

@draccus.wrap()
def train(
    args: TrainConfig,
):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    models = {}

    # Initialize wandb
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            name=args.run_name,
            config={
                "lr": args.learning_rate,
                "batch_size": args.batch_size,
                "lambda_l1": args.l1_lambda,
                "epochs": args.num_epochs,
            }
        )

    summary_reports = []

    for probe_idx, path in enumerate(args.dataset_paths):
        if args.verbose:
            print(f"Current probe: {probe_idx}")
        res = train_probe(
            path,
            args.num_epochs,
            args.batch_size,
            args.learning_rate,
            args.l1_lambda,
            args.probe_parameters,
            device,
            args.track,
            args.verbose,
            args.save_report,
            args.test_frac,
            args.room_size,
        )

        if args.save_report:
            model, reports = res[0], res[1]
            summary_reports.append(summarize_trial_reports(reports))
        else:
            model = res
        
        if len(args.dataset_paths) == 1:
            models[f"all_trials"] = model
        else:  
            models[f"trial_{probe_idx}"] = model

    # Save reports locally
    if args.save_report:
        for idx, report in enumerate(summary_reports):
            print(f"Probe {idx} summary:")
            print(report)
            print("=" * 15)

            report_path = os.path.join(f"reports/{args.run_name}/probe_{idx}_metrics.csv")
            os.makedirs(f"reports/{args.run_name}", exist_ok=True)
            report.to_csv(report_path, index=False)

    # Model saving setup
    if args.save_model:
        best_model_path = os.path.join("checkpoints", f"{args.run_name}_best.pt")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(
            {
                (k + "state_dict"): v.state_dict() for k, v in models.items()
            },
            best_model_path,
        )

if __name__ == "__main__":
    train()