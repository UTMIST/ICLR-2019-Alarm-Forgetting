import torch
import torch.nn.functional
import torchvision
# import matplotlib.pyplot as plt
import numpy as np

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


# ------------ CIRFAR 10 ----------------------

class Cutout(object):
  """Randomly mask out one or more patches from an image.
    Args:
    n_holes (int): Number of patches to cut out of each image.
    length (int): The length (in pixels) of each square patch.
  """
  def __init__(self, n_holes, length):
    self.n_holes = n_holes
    self.length = length

  def __call__(self, img):
    """
      Args:
        img (Tensor): Tensor image of size (C, H, W).
      Returns:
        Tensor: Image with n_holes of dimension length x length cut out of it.
    """
    h = img.size(1)
    w = img.size(2)

    mask = np.ones((h, w), np.float32)

    for n in range(self.n_holes):
      y = np.random.randint(h)
      x = np.random.randint(w)

      y1 = np.clip(y - self.length // 2, 0, h)
      y2 = np.clip(y + self.length // 2, 0, h)
      x1 = np.clip(x - self.length // 2, 0, w)
      x2 = np.clip(x + self.length // 2, 0, w)

      mask[y1: y2, x1: x2] = 0.

      mask = torch.from_numpy(mask)
      mask = mask.expand_as(img)
      img = img * mask

    return img


def load_cifar_10(train_batch_size, test_batch_size, cutout_len=10, cutout_holes=1):
  train_transforms = torchvision.transforms.Compose([])
  train_transforms.transforms.append(torchvision.transforms.RandomCrop(32, padding=4))
  train_transforms.transforms.append(torchvision.transforms.RandomHorizontalFlip())
  train_transforms.transforms.append(torchvision.transforms.ToTensor())
  train_transforms.transforms.append(torchvision.transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]]))
  train_transforms.transforms.append(Cutout(n_holes=cutout_holes, length=cutout_len))

  test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])])

  train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                   train=True,
                                   transform=train_transforms,
                                   download=True)

  test_dataset = torchvision.datasets.CIFAR10(root='data/',
                                  train=False,
                                  transform=test_transforms,
                                  download=True)

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

  return train_loader, test_loader