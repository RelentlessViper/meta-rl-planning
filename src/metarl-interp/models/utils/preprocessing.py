import torch

def one_hot(
    idx,
    size,
):
    x = torch.zeros(size)
    if len(size) == 2:
        for i in range(size[0]):
            x[i][int(idx[i])] = 1.0
    else:
        x[idx] = 1.0
    return x.float()


def one_hot_to_idx(one_hot_tensor):
    if one_hot_tensor.dim() == 2:
        return torch.argmax(one_hot_tensor, dim=1).cpu().numpy()
    else:
        return torch.argmax(one_hot_tensor).item()