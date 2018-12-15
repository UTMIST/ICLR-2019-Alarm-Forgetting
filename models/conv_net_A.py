import torch
import torch.nn
import torch.optim
import torch.autograd
import torch.nn.functional
import torchvision

# source: https://nextjournal.com/gkoehler/pytorch-mnist

# set the hyperparameters
n_epochs = 1
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
# print(example_data.shape)


class MNISTNet(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, padding=2)
    self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5, padding=2)
    self.connected1 = torch.nn.Linear(28 * 28 * 20, 50)
    self.connected2 = torch.nn.Linear(50, 10)

  def forward(self, x):
    y = self.conv1(x)
    y = torch.nn.functional.relu(y)
    y = self.conv2(y)
    y = torch.nn.functional.relu(y)
    y = y.view(y.size(0), -1)
    y = torch.nn.functional.relu(self.connected1(y))
    y = torch.nn.functional.softmax(self.connected2(y))
    # y = torch.nn.functional.log_softmax(self.connected2(y))
    return y

  def pred(self, x):
    v, i = self.forward(x).max(dim=1)
    return i


def train_model(net, loader, epochs):
  loss = torch.nn.CrossEntropyLoss()
  # loss = torch.nn.NLLLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
  for epoch in range(0, epochs):
    for i, (data, label) in enumerate(loader):
      x, y = torch.autograd.Variable(data), torch.autograd.Variable(label)
      optimizer.zero_grad()
      y_pred = net(x)
      loss_v = loss(y_pred, y)

      print(loss_v, epoch, i)
      loss_v.backward()
      optimizer.step()


def verify_model(net, loader):
  total = 0
  correct = 0
  for i, (data, label) in enumerate(loader):
    x= torch.autograd.Variable(data)
    pred = list(net.pred(x).numpy())
    label_lst = list(label.numpy())
    assert len(pred) == len(label_lst)
    total += len(pred)
    correct += sum([1 if p == l else 0 for p, l in zip(pred, label_lst)])
  return correct/total


if __name__ == "__main__":
  # test the code
  nn = MNISTNet()
  train_model(nn, train_loader, n_epochs)
  accuracy = verify_model(nn, test_loader)
  print("final accuracy is ", accuracy)