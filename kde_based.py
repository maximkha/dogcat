import torch
import numpy as np

# log prob implementation
def log_prob(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    distances = (x.repeat(mean.shape[0], 1).view(mean.shape[0], x.shape[0], mean.shape[1]) -  mean.view(mean.shape[0], 1, mean.shape[1]))**2
    comp_stds = std.repeat(1,x.shape[0]).view(mean.shape[0],x.shape[0],mean.shape[1])
    return -.5 * (torch.sum((distances / comp_stds) + torch.log(comp_stds), axis=2) +(x.shape[-1] * np.log(2 * np.pi)))

def multi_log_prob(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    log_probs = log_prob(x, mean, std)
    return torch.log(torch.sum(torch.exp(log_probs), axis=0)) -np.log(log_probs.shape[0])

mean = torch.Tensor([[1.]*2, [2.]*2])
std = torch.Tensor([[1.]*2, [1.]*2])

sample_point = torch.Tensor([[1.]*2, [2.]*2, [3.]*2])

from sklearn.neighbors import KernelDensity
kde = KernelDensity(kernel='gaussian', bandwidth=1.).fit(np.array([[2.,2.],[1.,1.]]))
X = np.array([[1, 1], [2, 2], [3, 3]])
kde.score_samples(X)

print(f"{log_prob(sample_point.view(3, 2), mean.view(2, 2), std.view(2, 2))=}")
print(f"{multi_log_prob(sample_point.view(3, 2), mean.view(2, 2), std.view(2, 2))=}")
print(f"{log_prob(sample_point.view(3, 2), mean.view(2, 2), std.view(2, 2)).shape=}")
# torch.log(torch.mean(torch.exp(vals), axis=0))

print(f"{kde.score_samples(X)=}")