import torch
import numpy as np

# log prob implementation

def zscore(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std

def log_prob_individual(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:

    print(f"{torch.sum(((x - mean)**2 / std) + torch.log(std))=}")
    print(f"{((x - mean)**2 / std) + torch.log(std)=}")
    print(f"{torch.log(std)=}")
    print(f"{((x - mean)**2 / std)=}")
    print(f"{(x.shape[-1] * 2 * np.pi)=}")

    return -.5 * (torch.sum(((x - mean)**2 / std) + torch.log(std), dim=1) +(x.shape[-1] * np.log(2 * np.pi)))

mean = torch.Tensor([1.]*2)
std = torch.Tensor([1.]*2)

sample_point = torch.Tensor([[1.]*2, [2.]*2, [3.]*2])

print(f"{torch.mean(torch.exp(log_prob_individual(sample_point, mean, std)))=}")
print(f"{log_prob_individual(sample_point, mean, std)=}")