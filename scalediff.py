import collections
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

def avgdist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.min(torch.sum((A.repeat(B.shape[0], 1).view(B.shape[0], A.shape[0], B.shape[1]) -  B.view(B.shape[0], 1, B.shape[1]))**2, dim=2), dim=1)[0])

import sklearn.datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

a_sample = sklearn.datasets.make_s_curve(n_samples=400, noise=0.25, random_state=None)[0][:, [0, 2]]
a_sample = torch.Tensor(a_sample).to(device)

b_sample = sklearn.datasets.make_s_curve(n_samples=400, noise=0.25, random_state=None)[0][:, [0, 2]]
b_sample = torch.Tensor(b_sample).to(device)

mu = (torch.rand(1)[0] * 2) -1

b_sample_scaled = b_sample * mu

test_mus = torch.linspace(-5, 5, 1000)
losses = []
mus = []
for test_mu in test_mus:
    test_scale = a_sample * test_mu
    # loss = avgdist(b_sample_scaled, test_scale)
    # loss = avgdist(test_scale, b_sample_scaled)

    loss = .5*(avgdist(b_sample_scaled, test_scale) + avgdist(test_scale, b_sample_scaled))

    losses.append(loss.detach().cpu().numpy())
    mus.append(test_mu.detach().cpu().numpy())

#display optimal
est_low = mus[np.argmin(losses[:500])]
print(f"{est_low=}")
print(f"{mu=}")

import matplotlib.pyplot as plt
plt.plot(mus, losses)
plt.vlines(mu, np.min(losses)*.9, np.max(losses)*1.1, color="red")
plt.vlines(est_low, np.min(losses)*.9, np.max(losses)*1.1, color="green")
plt.show()

plt.scatter(*(b_sample_scaled.T.detach().cpu().numpy()))
plt.scatter(*(a_sample.T.detach().cpu().numpy()))
plt.show()

plt.scatter(*(b_sample_scaled.T.detach().cpu().numpy()))
plt.scatter(*((a_sample * torch.Tensor(est_low).to(device)).to(device)).T.detach().cpu().numpy())
plt.show()