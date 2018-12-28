import torch
import torch.nn.functional
import torchvision
# import matplotlib.pyplot as plt

# source: https://nextjournal.com/gkoehler/pytorch-mnist

def load_mnist(batch_size_train, batch_size_test, norm_mean, norm_std):
  train_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.MNIST('data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 norm_mean, norm_std)
                             ])),
                            batch_size=batch_size_train, shuffle=False)
  # to use just a sample of the dataset, include the sampler argument; sampler and shuffle argument cannot be used together
  # batch_size=batch_size_train, sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices))


  # load the test examples of MNIST
  test_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.MNIST('data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 norm_mean, norm_std)
                             ])),
                              batch_size=batch_size_test, shuffle=False)
  # to use just a sample of the dataset, include the sampler argument; sampler and shuffle argument cannot be used together
  # batch_size=batch_size_test, sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices))
  return train_loader, test_loader

def load_permuted_mnist(batch_size_train, batch_size_test, norm_mean, norm_std):
  permutation = torch.randperm(28*28)
  permutate = lambda x: x.view(-1, 1)[permutation].view(-1, 28, 28)
  train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(
                                   norm_mean, norm_std),
                                 torchvision.transforms.Lambda(permutate)
                               ])),
    batch_size=batch_size_train, shuffle=False)

  # load the test examples of MNIST
  test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(
                                   norm_mean, norm_std),
                                 torchvision.transforms.Lambda(permutate)
                               ])),
    batch_size=batch_size_test, shuffle=False)

  return train_loader, test_loader


def load_partial_train_mnist(batch_size_train, batch_size_test, norm_mean, norm_std, to_remove, train_len=60000):
  # print('loading')
  new_indices = []
  for i in range(0, train_len):
    if i not in to_remove:
      new_indices.append(i)

  # print(to_remove, len(new_indices))

  test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(
                                   norm_mean, norm_std)
                               ])),
    batch_size=batch_size_test, shuffle=False)

  train_dataset = torchvision.datasets.MNIST('data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 norm_mean, norm_std)
                             ]))
  subset = torch.utils.data.Subset(train_dataset, new_indices)
  # print(len([i for i, (data, label) in enumerate(subset)]), len(new_indices))
  train_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size_train, shuffle=False)
  return train_loader, test_loader