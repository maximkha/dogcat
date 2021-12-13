import collections
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

def avgdist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.min(torch.sum((A.repeat(B.shape[0], 1).view(B.shape[0], A.shape[0], B.shape[1]) -  B.view(B.shape[0], 1, B.shape[1]))**2, dim=2), dim=1)[0])

def rotate(xypairs: torch.Tensor, theta) -> torch.Tensor:
    xypairs_rotated = torch.zeros_like(xypairs)
    xypairs_rotated[:, 0] = (xypairs[:, 0] * torch.cos(theta)) - (xypairs[:, 1] * torch.sin(theta))
    xypairs_rotated[:, 1] = (xypairs[:, 0] * torch.sin(theta)) + (xypairs[:, 1] * torch.cos(theta))
    return xypairs_rotated

import sklearn.datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

a_sample = sklearn.datasets.make_s_curve(n_samples=400, noise=0.25, random_state=None)[0][:, [0, 2]]
a_sample = torch.Tensor(a_sample).to(device)

b_sample = sklearn.datasets.make_s_curve(n_samples=400, noise=0.25, random_state=None)[0][:, [0, 2]]
b_sample = torch.Tensor(b_sample).to(device)

theta = torch.rand(1)[0] * np.pi * 2

b_sample_rotated = rotate(b_sample, theta)

test_thetas = torch.linspace(0, 2 * np.pi, 1000)
losses = []
thetas = []
for test_theta in test_thetas:
    test_rotation = rotate(a_sample, test_theta)
    # loss = avgdist(b_sample_rotated, test_rotation)
    # loss = avgdist(test_rotation, b_sample_rotated)
    loss = .5*(avgdist(b_sample_rotated, test_rotation) + avgdist(test_rotation, b_sample_rotated))

    losses.append(loss.detach().cpu().numpy())
    thetas.append(test_theta.detach().cpu().numpy())

#display optimal
est_low = thetas[np.argmin(losses[:500])]
print(f"{est_low=}")
print(f"{est_low+(np.pi*2)=}")
print(f"{est_low+(np.pi)=}")
print(f"{theta=}")

import matplotlib.pyplot as plt
plt.plot(thetas, losses)
plt.vlines(theta, np.min(losses)*.9, np.max(losses)*1.1, color="red")
plt.vlines(est_low, np.min(losses)*.9, np.max(losses)*1.1, color="green")
plt.show()

plt.scatter(*(b_sample_rotated.T.detach().cpu().numpy()))
plt.scatter(*(a_sample.T.detach().cpu().numpy()))
plt.show()

plt.scatter(*(b_sample_rotated.T.detach().cpu().numpy()))
plt.scatter(*(rotate(a_sample, torch.Tensor(est_low).to(device)).T.detach().cpu().numpy()))
plt.show()