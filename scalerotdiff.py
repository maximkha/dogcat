import torch
import torch.nn as nn
import numpy as np

# nice functions
def avgdist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.min(torch.sum((A.repeat(B.shape[0], 1).view(B.shape[0], A.shape[0], B.shape[1]) -  B.view(B.shape[0], 1, B.shape[1]))**2, dim=2), dim=1)[0])

def rotate_scale(xypairs: torch.Tensor, theta, scale) -> torch.Tensor:
    xypairs_rotated = torch.zeros_like(xypairs)
    xypairs_rotated[:, 0] = (xypairs[:, 0] * torch.cos(theta)) - (xypairs[:, 1] * torch.sin(theta)) * scale
    xypairs_rotated[:, 1] = (xypairs[:, 0] * torch.sin(theta)) + (xypairs[:, 1] * torch.cos(theta)) * scale
    return xypairs_rotated

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create fake data
import sklearn.datasets

a_sample = sklearn.datasets.make_s_curve(n_samples=250, noise=0.25, random_state=None)[0][:, [0, 2]]
a_sample = torch.Tensor(a_sample).to(device)

b_sample = sklearn.datasets.make_s_curve(n_samples=250, noise=0.25, random_state=None)[0][:, [0, 2]]
b_sample = torch.Tensor(b_sample).to(device)

true_mu = (torch.rand(1)[0] * 2) -1
true_theta = torch.rand(1)[0] * np.pi * 1

b_sample_transformed = rotate_scale(b_sample, true_theta, true_mu)

# sample loss as a function of anlge (theta) and scale (mu)

test_mus = torch.linspace(-1, 1, steps=50).to(device)
test_thetas = torch.linspace(0, 1 * np.pi, steps=50).to(device)

surf_mu = []
surf_theta = []
surf_loss = []

for test_mu in test_mus:
    for test_theta in test_thetas:
        surf_mu.append(test_mu.detach().cpu().numpy())
        surf_theta.append(test_theta.detach().cpu().numpy())

        test_transform = rotate_scale(a_sample, test_theta, test_mu)

        loss = .5 * (avgdist(b_sample_transformed, test_transform) + avgdist(test_transform, b_sample_transformed))
        # loss = avgdist(b_sample_transformed, test_transform)

        surf_loss.append(loss.detach().cpu().numpy())

import matplotlib.pyplot as plt
from matplotlib import cm
# import matplotlib
# matplotlib.use("gtk4agg")

from matplotlib.ticker import MaxNLocator

plt.scatter(*(b_sample_transformed.T.detach().cpu().numpy()))
plt.scatter(*(a_sample.T.detach().cpu().numpy()))
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print(f"{np.array([np.max(surf_loss)])}")

ax.set_xlabel("Mu")
ax.set_ylabel("Theta")
ax.set_zlabel("Loss")

ax.scatter(np.ones((2,)) * true_mu.detach().cpu().numpy(), np.ones((2,)) * true_theta.detach().cpu().numpy(), np.array([0, np.max(surf_loss)]), color="red")

# https://matplotlib.org/stable/tutorials/colors/colormaps.html
surf = ax.plot_trisurf(surf_mu, surf_theta, surf_loss, cmap=cm.viridis, linewidth=0)
fig.colorbar(surf)
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))
fig.tight_layout()

plt.show()

# ax.stem(x, y, z)

test_transform = rotate_scale(a_sample, true_theta, true_mu)
true_loss = .5 * (avgdist(b_sample_transformed, test_transform) + avgdist(test_transform, b_sample_transformed))
print(f"{true_mu=}, {true_theta=}, {true_loss=}")

ind = np.argmin(surf_loss)
est_mu = surf_mu[ind]
est_theta = surf_theta[ind]
est_loss = surf_loss[ind]

print(f"{est_mu=}, {est_theta=}, {est_loss=}")

test_transform = rotate_scale(a_sample, torch.Tensor(est_theta).to(device), torch.Tensor(est_mu).to(device))
plt.scatter(*(b_sample_transformed.T.detach().cpu().numpy()))
plt.scatter(*(test_transform.T.detach().cpu().numpy()))
plt.show()