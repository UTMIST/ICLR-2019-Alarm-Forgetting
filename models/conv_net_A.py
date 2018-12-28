import torch
import torch.nn
import torch.optim
import torch.autograd
import torch.nn.functional
import torchvision
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import sys
# note: this line is underlined red in my IDE but it actually runs. If you see it underlined just ignored it
import data_loaders
import csv
from time import sleep

# set the hyperparameters
n_epochs = 30
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

norm_mean = [0.1307]
norm_std = [0.3081]

# choosing the random seed
# we would have to use 10 different specific seeds to later run the training
# that many times to compare stability of forgetting events
random_seed = 1
torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)

# load the training examples of MNIST
# train_loader = torch.utils.data.DataLoader(
#   torchvision.datasets.MNIST('../data/', train=True, download=True,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  norm_mean, norm_std)
#                              ])),
#   # to use just a sample of the dataset, include the sampler argument; sampler and shuffle argument cannot be used together
#   # batch_size=batch_size_train, sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices))
#   batch_size=batch_size_train, shuffle = False)
#
# # load the test examples of MNIST
# test_loader = torch.utils.data.DataLoader(
#   torchvision.datasets.MNIST('../data/', train=False, download=True,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  norm_mean, norm_std)
#                              ])),
#   # to use just a sample of the dataset, include the sampler argument; sampler and shuffle argument cannot be used together
#   # batch_size=batch_size_test, sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices))
#   batch_size=batch_size_test, shuffle = False)
#
# # test whether the examples are actually loaded properly by printing out the
# # shape of the tensor
# # examples = enumerate(test_loader)
# # batch_idx, (example_data, example_targets) = next(examples)
#
# # should output [ 1000, 1, 28, 28]
# # print(example_data.shape)


class MNISTNet(torch.nn.Module):
  def __init__(self):
    WIDTH_PIXELS = 28
    HEIGHT_PIXELS = 28
    super().__init__()
    self.num_training_examples = 60000
    self.num_test_examples = 10000
    self.forgetting_events = np.zeros(self.num_training_examples)
    self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, padding=2)
    self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5, padding=2)
    self.connected1 = torch.nn.Linear(WIDTH_PIXELS * HEIGHT_PIXELS * 20, 50)
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


def train_model(net, loader, epochs, verbose=False, less_intensive=False, sleep_time=0.5, record_forgetting=True):
  loss = torch.nn.CrossEntropyLoss()
  # loss = torch.nn.NLLLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

  prev_acc = [-1 for i in range(net.num_training_examples)]
  acc = [0 for i in range(net.num_training_examples)]

  for epoch in range(0, epochs):
    for i, (data, label) in enumerate(loader):
      x, y = torch.autograd.Variable(data), torch.autograd.Variable(label)
      optimizer.zero_grad()
      y_pred = net(x)

      # make a prediction and get the labels for the current batch
      y_class = list(net.pred(x).numpy())
      label_lst = list(label.numpy())

      # compute which labels are correct in the current batch
      acc[i*batch_size_train:min((i+1)*batch_size_train, net.num_training_examples)] = [1 if p == l else 0 for p, l in zip(y_class, label_lst)]

      # compare previous accuracies to compute forgetting events for the current batch of training examples
      if record_forgetting:
        if epoch > 0:
            a = prev_acc[i*batch_size_train:min((i+1)*batch_size_train, net.num_training_examples)]
            b = acc[i*batch_size_train:min((i+1)*batch_size_train, net.num_training_examples)]
            forgetting_indices = [False for i in range(net.num_training_examples)]
            forgetting_indices[i*batch_size_train:min((i+1)*batch_size_train, net.num_training_examples)] = [True if a_i - b_i == 1 else False for a_i, b_i in zip(a, b)]
            net.forgetting_events[forgetting_indices] += 1

        prev_acc[:] = acc[:]

      # compute the loss and take the gradient step
      loss_v = loss(y_pred, y)
      if verbose:
        print(loss_v, epoch, i)
      loss_v.backward()
      optimizer.step()
      if less_intensive:
        sleep(sleep_time)

  # after training the model, set all the unlearned examples to have a distinct forgetting events to distinguish
  # it from unforgettable and forgettable examples
  if record_forgetting:
    unlearned_indices = [False for k in range(net.num_training_examples)]
    for j, (data, label) in enumerate(loader):
        d = torch.autograd.Variable(data)
        pred = list(net.pred(d).numpy())
        label_lst = list(label.numpy())
        unlearned_indices[j*batch_size_train:min((j+1)*batch_size_train, net.num_training_examples)] = [True if p != l else False for p, l in zip(pred, label_lst)]

    unlearned_indices = [True if p and l == 0 else False for p, l in zip(unlearned_indices, net.forgetting_events)]
    net.forgetting_events[unlearned_indices] = sys.maxsize

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

def generate_forgetting_events_stats(net):
    num_forgettable_examples = sum([1 if x > 0 and x < sys.maxsize else 0 for x in net.forgetting_events])
    num_unlearned_examples = sum([1 if x == sys.maxsize else 0 for x in net.forgetting_events])
    num_unforgettable_examples = net.num_training_examples - num_forgettable_examples - num_unlearned_examples

    return (num_forgettable_examples, num_unlearned_examples, num_unforgettable_examples)


def write_forgetting_events_mnist(fn, net):
  fields = ['index', 'forgetting_events']
  dts = [{'index': i, 'forgetting_events': net.forgetting_events[i]} for i in range(0, len(net.forgetting_events))]
  with open(fn, mode='w') as f:
    writer = csv.DictWriter(f, fields)
    writer.writeheader()
    writer.writerows(dts)



if __name__ == "__main__":
  # test the code
  from multiprocessing import Process

  def run_mnist_experiment(seed, permuted, sleep_t):
    torch.manual_seed(seed)
    if not permuted:
      train_loader, test_loader = data_loaders.load_mnist(batch_size_train, batch_size_test, norm_mean, norm_std)
      file_nm = '../experiments/seed' + str(seed) + '/mnist_forgetting.csv'
    else:
      train_loader, test_loader = data_loaders.load_permuted_mnist(batch_size_train,
                                                                   batch_size_test, norm_mean, norm_std)
      file_nm = '../experiments/seed' + str(seed) + '/permuted_mnist_forgetting.csv'

    print('running seed', seed, 'on', file_nm)
    nn = MNISTNet()
    train_model(nn, train_loader, n_epochs, verbose=True, less_intensive=True, sleep_time=sleep_t)

    write_forgetting_events_mnist(file_nm, nn)
    print("finished writing!")
    accuracy = verify_model(nn, test_loader)
    print("final accuracy", accuracy, ", seed", seed, ", is permuted", permuted)
    print("seed", seed, "has", generate_forgetting_events_stats(nn))
    print("finished", seed)

  sleep_tm = 0.3
  # seeds = [7]
  # seeds = [31]
  # seeds = [35]
  # seeds = [81]
  run_mnist_experiment(7, False, sleep_tm)
  # processes = []
  # for i in range(0, len(seeds)):
  #   sd = seeds[i]
  #   pc_p = Process(target=run_mnist_experiment, args=(sd, True, sleep_tm))
  #   pc_m = Process(target=run_mnist_experiment, args=(sd, False, sleep_tm))
  #   pc_p.start()
  #   pc_m.start()
  #   processes.append(pc_p)
  #   processes.append(pc_m)
  #
  # print("in total", len(processes), "processes")

  # for p in processes:
  #   p.join()
