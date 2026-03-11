import os

import draccus
from dataclasses import dataclass
from tqdm import trange, tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader

@dataclass
class DatasetPreprocessingConfig:
    dataset_name: str = "1x1-probe-dataset-5x5g"
    seed: int = 1
    probe_type: str = "1x1"
    existing_dataset_path: str = None
    save_path: str = None
    cuda: bool = True
    
    def __post_init__(self):
        if self.existing_dataset_path is None:
            raise ValueError("`existing_dataset_path` must be filled")
        if not self.save_path:
            self.save_path = f"datasets/{self.dataset_name}"

class TupleDataset(Dataset):
    def __init__(self, tuple_dataset):
        self.length = len(tuple_dataset[0])
        for tensor in tuple_dataset:
            assert len(tensor) == self.length, (
                "Tensors in given tuple must have the same length"
            )
        self.data = tuple_dataset
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.data)
    
def process_batch_to_grid_samples(hidden_states, actions, observations, grid_states, trial_idxs):
    """
    Convert a batch of experiences into grid cell samples.
    """
    batch_size = hidden_states.shape[0]
    grid_size = grid_states.shape[-1]
    
    positions = torch.arange(grid_size * grid_size)
    position_one_hot = F.one_hot(positions, num_classes=grid_size * grid_size).float()
    
    # Expand hidden states to match number of grid cells
    expanded_hidden = hidden_states.unsqueeze(1).expand(-1, grid_size * grid_size, -1)
    expanded_trial_idxs = trial_idxs.unsqueeze(1).expand(-1, grid_size * grid_size, -1)
    
    # Combine hidden states with position encodings
    expanded_positions = position_one_hot.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Concatenate hidden states with position encodings
    # Result: (batch_size, 25, 512 + 25 + 3)
    combined_features = torch.cat(
        [
            expanded_hidden,
            expanded_positions,
            expanded_trial_idxs,
        ], 
        dim=-1
    )
    
    targets = grid_states.view(batch_size, -1)

    # Replace -1.0 class with 5.0
    targets[targets == -1.0] = 5.0
    
    num_samples = batch_size * grid_size * grid_size
    
    # Final shapes:
    # features: (num_samples, 512 + 25)
    # targets: (num_samples,)

    return combined_features.view(num_samples, -1), targets.view(num_samples),

@draccus.wrap()
def preprocess_dataset(
    args: DatasetPreprocessingConfig,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    existing_dataset = TupleDataset(torch.load(args.existing_dataset_path))

    dataloader = DataLoader(
        dataset=existing_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type != "cpu" else False,
    )

    os.makedirs(args.save_path, exist_ok=True)
    all_features, all_targets = [], []
    for idx, (hidden_states, actions, observations, grid_states, trial_idxs) in enumerate(tqdm(dataloader, desc="Preprocessing")):
        features, targets = process_batch_to_grid_samples(
            hidden_states,
            actions,
            observations,
            grid_states,
            trial_idxs,
        )
        all_features.append(features)
        all_targets.append(targets)

    all_features = torch.cat(all_features)
    all_targets = torch.cat(all_targets)

    # Remove duplicates
    unique_mask = []
    seen = set()
    for i in trange(all_features.shape[0], desc="Removing duplicates"):
        # Create a hashable representation
        key = torch.cat([
            all_features[i].reshape(-1),
            all_targets[i].reshape(-1),
        ]).cpu().numpy().tobytes()

        if key not in seen:
            seen.add(key)
            unique_mask.append(i)

    unique_indices = torch.tensor(unique_mask, dtype=torch.long)

    all_features = all_features[unique_indices]
    all_targets = all_targets[unique_indices]

    temp_dataset = TensorDataset(
        all_features,
        all_targets,
    )

    del all_features, all_targets

    torch.save(temp_dataset, f"{args.save_path}/processed_dataset.pt")

if __name__ == "__main__":
    preprocess_dataset()