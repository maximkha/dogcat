import torch
import numpy as np

# log prob implementation

def zscore(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std

def log_prob_individual(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return -.5 * (torch.sum(((x - mean)**2 / std) + torch.log(std), dim=1) +(x.shape[-1] * np.log(2 * np.pi)))

def log_prob(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    distances = (x.repeat(mean.shape[0], 1).view(mean.shape[0], x.shape[0], mean.shape[1]) -  mean.view(mean.shape[0], 1, mean.shape[1]))**2
    comp_stds = std.repeat(1,x.shape[0]).view(mean.shape[0],x.shape[0],mean.shape[1])
    return -.5 * (torch.sum((distances / comp_stds) + torch.log(comp_stds), axis=2) +(x.shape[-1] * np.log(2 * np.pi)))

mean = torch.Tensor([[1.]*2, [2.]*2])
std = torch.Tensor([[1.]*2, [1.]*2])

sample_point = torch.Tensor([[1.]*2, [2.]*2, [3.]*2])

from sklearn.neighbors import KernelDensity
kde = KernelDensity(kernel='gaussian', bandwidth=1.).fit(np.array([[1.,1.]]))
X = np.array([[1, 1], [2, 2], [3, 3]])
kde.score_samples(X)

# print(f"{torch.mean(torch.exp(log_prob_individual(sample_point, mean, std)))=}")
print(f"{log_prob_individual(sample_point, mean[0], std[0])=}")
print(f"{log_prob_individual(sample_point, mean[1], std[1])=}")

print(f"{log_prob(sample_point.view(3, 2), mean.view(2, 2), std.view(2, 2))=}")
print(f"{kde.score_samples(X)=}")