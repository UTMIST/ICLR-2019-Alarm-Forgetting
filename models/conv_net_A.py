import torch
import torchvision

# source: https://nextjournal.com/gkoehler/pytorch-mnist

# set the hyperparameters
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

# choosing the random seed
# we would have to use 10 different specific seeds to later run the training
# that many times to compare stability of forgetting events
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# load the training examples of MNIST
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

# load the test examples of MNIST
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

# test whether the examples are actually loaded properly by printing out the
# shape of the tensor
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

# should output [ 1000, 1, 28, 28]
print(example_data.shape)
