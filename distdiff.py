import collections
from math import floor
import torch
import torch.optim as optim
import torch.nn as nn

# hours of work :)
def avgdist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # standard best point:
    # torch.min(torch.sum((A.repeat(B.shape[0], 1).view(B.shape[0], A.shape[0], B.shape[1]) -  B.view(B.shape[0], 1, B.shape[1]))**2, dim=2), dim=1)[0]

    # return torch.sum(torch.min(torch.sum((A.repeat(B.shape[0], 1).view(B.shape[0], A.shape[0], B.shape[1]) -  B.view(B.shape[0], 1, B.shape[1]))**2, dim=2), dim=1)[0])
    # same as above
    # return torch.sum(torch.sqrt(torch.min(torch.sum((A.repeat(B.shape[0], 1).view(B.shape[0], A.shape[0], B.shape[1]) -  B.view(B.shape[0], 1, B.shape[1]))**2, dim=2), dim=1)[0]))
    
    # better
    # return torch.mean(torch.min(torch.sum((A.repeat(B.shape[0], 1).view(B.shape[0], A.shape[0], B.shape[1]) -  B.view(B.shape[0], 1, B.shape[1]))**2, dim=2), dim=1)[0])
    # same as above

    return torch.mean(torch.min(torch.sum((A.repeat(B.shape[0], 1).view(B.shape[0], A.shape[0], B.shape[1]) -  B.view(B.shape[0], 1, B.shape[1]))**2, dim=2), dim=1)[0])

    # return torch.max(torch.min(torch.sum((A.repeat(B.shape[0], 1).view(B.shape[0], A.shape[0], B.shape[1]) -  B.view(B.shape[0], 1, B.shape[1]))**2, dim=2), dim=1)[0])
    # same as above

    # return torch.min(torch.mean(torch.sum((A.repeat(B.shape[0], 1).view(B.shape[0], A.shape[0], B.shape[1]) -  B.view(B.shape[0], 1, B.shape[1]))**2, dim=2), dim=1)[0]) # bad
    # return torch.sum(torch.sum(torch.sum((A.repeat(B.shape[0], 1).view(B.shape[0], A.shape[0], B.shape[1]) -  B.view(B.shape[0], 1, B.shape[1]))**2, dim=2), dim=1)[0]) # bad

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import sklearn.datasets

# a_dist = torch.distributions.normal.Normal(torch.tensor([0.0, 0.0]), torch.tensor([.1, .1]))
# a_sample = a_dist.sample((500,)).to(device)
a_sample = sklearn.datasets.make_s_curve(n_samples=1000, noise=0.125, random_state=None)[0][:, [0, 2]]
# print(a_sample)
a_sample = torch.Tensor(a_sample).to(device)

# b_dist = torch.distributions.normal.Normal(torch.tensor([-1., 2.]), torch.tensor([4.0, 5.0]))
# b_dist = torch.distributions.normal.Normal(torch.tensor([-.1, 0.5]), torch.tensor([0.25, 0.1]))
# b_sample = a_sample.detach() * 2 + 1 # b_dist.sample((500,))
# b_dist = torch.distributions.normal.Normal(torch.tensor([-1., 2.]), torch.tensor([4.0, 5.0]))
# b_sample = b_dist.sample((500,)).to(device)

b_sample = sklearn.datasets.make_s_curve(n_samples=1000, noise=0.125, random_state=None)[0][:, [0, 2]]
# print(a_sample)
b_sample = torch.Tensor(b_sample).to(device)

# random theta
# rotation = torch.rand(1)[0]*3.14*2
# b_sample_rotated = b_sample.detach().clone()
# b_sample_rotated[:, 0] = (b_sample[:, 0] * torch.cos(rotation)) - (b_sample[:, 1] * torch.sin(rotation))
# b_sample_rotated[:, 1] = (b_sample[:, 0] * torch.sin(rotation)) + (b_sample[:, 1] * torch.cos(rotation))
# b_sample = b_sample_rotated.detach()

# print(avgdist(a_sample, b_sample))

# a_norm = (a_sample - a_sample.mean(dim=0)) / a_sample.std(dim=0)

# b_norm = (b_sample - b_sample.mean(dim=0)) / b_sample.std(dim=0)

import matplotlib.pyplot as plt
plt.scatter(*(a_sample.cpu().T), alpha=.5)
plt.scatter(*(b_sample.cpu().T), alpha=.5)

affine_bijector = nn.Linear(2, 2, False).to(device) # affine

print(f"{list(affine_bijector.parameters())[0].det()=}")
plt.show()

# plt.scatter(*(a_sample.T), alpha=.5)
# plt.scatter(*(affine_bijector(a_sample).detach().T.numpy()), alpha=.5)
# plt.show()

# plt.scatter(*(b_sample.T), alpha=.5)
# plt.scatter(*(affine_bijector(a_sample).detach().T.numpy()), alpha=.5)
# plt.show()

# optimizer = optim.Adagrad(affine_bijector.parameters(), lr=0.01)
# optimizer = optim.SGD(affine_bijector.parameters(), lr=0.01, momentum=.1)
optimizer = optim.Adam(affine_bijector.parameters(), lr=0.01)
# optimizer = optim.Adadelta(affine_bijector.parameters())

lossf = avgdist # nn.MSELoss() # avgdist

b_size = floor(a_sample.shape[0]*1)

for epoch in range(500):
    optimizer.zero_grad()

    outputs = affine_bijector(a_sample[torch.randperm(a_sample.shape[0])[:b_size]])
    # loss = lossf(outputs, b_sample)
    # loss = avgdist(b_sample, outputs)
    compare_sample = b_sample[torch.randperm(b_sample.shape[0])[:b_size]]

    loss = .5 * (lossf(outputs, compare_sample) + lossf(compare_sample, outputs))
    print(f"{loss=}")
    loss.backward()
    optimizer.step()

print(list(affine_bijector.parameters()))

# plt.scatter(*(a_sample.T), alpha=.5)
# plt.scatter(*(affine_bijector(a_sample).detach().T.numpy()), alpha=.5)
# plt.show()

plt.scatter(*(b_sample.cpu().T), alpha=.5)
plt.scatter(*(affine_bijector(a_sample).detach().T.cpu().numpy()), alpha=.5)
# plt.scatter([-1.], [2.], color="red")
plt.show()