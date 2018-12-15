import torch
import torch.nn
import torch.optim
import torch.autograd
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


class MNISTNet(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1_conv = torch.nn.Conv2d(1, 10, kernel_size=5)
    self.activation1 = torch.nn.ReLU()
    self.layer2_conv = torch.nn.Conv2d(10, 20, kernel_size=5)
    self.activation2 = torch.nn.ReLU()
    self.layer3_connected = torch.nn.Linear(20 * 20 * 20, 50)
    self.activation3 = torch.nn.ReLU()
    self.layer4_connected = torch.nn.Linear(50, 10)
    self.activation4 = torch.nn.Softmax()

  def forward(self, x):
    y = self.activation1(self.layer1_conv(x))
    y = self.activation2(self.layer2_conv(y))
    y = y.view(y.size(0), -1)
    y = self.activation3(self.layer3_connected(y))
    y = self.activation4(self.layer4_connected(y))
    return y


def train_model(net, loader, epochs):
  loss = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
  for epoch in range(0, epochs):
    for i, (data, label) in enumerate(loader):
        x, y = torch.autograd.Variable(data), torch.autograd.Variable(label)
        optimizer.zero_grad()
        y_pred = net(x)
        loss_v = loss(y_pred, y)

        print(loss_v, epoch, i)
        loss_v.backward()
        optimizer.step()


if __name__ == "__main__":
  # test the code
  nn = MNISTNet()
  train_model(nn, train_loader, n_epochs)
