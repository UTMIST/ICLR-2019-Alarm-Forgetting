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
import train_and_verify_models
import forgetting_events
import csv
from time import sleep

# # set the hyperparameters
n_epochs = 1
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

norm_mean = [0.1307]
norm_std = [0.3081]

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

if __name__ == "__main__":
  # test the code
  from multiprocessing import Process

  def run_mnist_experiment(seed, permuted, sleep_t):
    torch.manual_seed(seed)
    if not permuted:
      train_loader, test_loader = data_loaders.load_mnist(batch_size_train, batch_size_test, norm_mean, norm_std)
      file_nm = 'experiments/seed' + str(seed) + '/mnist_forgetting.csv'
    else:
      train_loader, test_loader = data_loaders.load_permuted_mnist(batch_size_train,
                                                                   batch_size_test, norm_mean, norm_std)
      file_nm = 'experiments/seed' + str(seed) + '/permuted_mnist_forgetting.csv'

    print('running seed', seed, 'on', file_nm)
    nn = MNISTNet()
    train_and_verify_models.train_model(nn, train_loader, n_epochs, batch_size_train, learning_rate, momentum, verbose=True, less_intensive=True, sleep_time=sleep_t)

    forgetting_events.write_forgetting_events_mnist(file_nm, nn)
    print("finished writing!")
    accuracy = train_and_verify_models.verify_model(nn, test_loader)
    print("final accuracy", accuracy, ", seed", seed, ", is permuted", permuted)
    print("seed", seed, "has", forgetting_events.generate_forgetting_events_stats(nn))
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
