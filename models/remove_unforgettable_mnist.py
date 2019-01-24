import conv_net_A as net1
import data_loaders
import csv
import torch
from random import sample, seed

n_epochs = 30
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
norm_mean = [0.1307]
norm_std = [0.3081]

def unforgettable_indices(csv_fn):
  indices = []
  with open(csv_fn, mode='r') as f:
    reader = csv.DictReader(f)
    for r in reader:
      if float(r['forgetting_events']) <= 0:
        indices.append(int(r['index']))
  return indices

def indices_to_remove(remove_percentage, indices):
  # remove the indices
  l = len(indices)
  rl = int(remove_percentage * l)
  return sample(indices, rl)


def rd_indices_to_remove(remove_percentage, total, indices):
  # randomly remove from a larger set of indices
  rl = int(remove_percentage * total)
  return sample(indices, rl)


def test_remove_unforgettable(rm_percentages, csv_fn, batch_size_train, batch_size_test, norm_mean, norm_std, epochs,
                              torch_seed, rd_seed, less_intensive=False, sleep_tm=0.5, return_on_interrupt=True,
                              random_remove=False):
  # random remove set false means to remove unforgettable examples otherwise remove randomly from the larger population
  #   the same number of examples
  seed(rd_seed)
  torch.manual_seed(torch_seed)
  unf_i = unforgettable_indices(csv_fn)
  accuracies = []
  try:
    for rm_p in rm_percentages:
      if not random_remove:
        to_rm = indices_to_remove(rm_p, unf_i)
      else:
        to_rm = rd_indices_to_remove(rm_p, len(unf_i), [i for i in range(0, 60000)])
      # print(len(to_rm))
      train_l, test_l = data_loaders.load_partial_train_mnist(batch_size_train, batch_size_test,
                                                              norm_mean, norm_std, set(to_rm))
      # print('loaded')
      print(len([i for i, (data, label) in enumerate(train_l)]))
      nn = net1.MNISTNet()
      net1.train_model(nn, train_l, epochs, verbose=True, less_intensive=less_intensive, sleep_time=sleep_tm, record_forgetting=False)
      accuracy = net1.verify_model(nn, test_l)
      print('accuracy:', accuracy, 'by removing', rm_p, 'of unforgettable')
      accuracies.append({'remove_percentage': rm_p, 'accuracy': accuracy})
  except KeyboardInterrupt:
    if return_on_interrupt:
      return accuracies
    else:
      raise KeyboardInterrupt
  return accuracies


if __name__ == '__main__':
  # rm_percentages = [0.6, 0.525, 0.45, 0.375, 0.3, 0.225, 0.15, 0.075, 0]
  # rm_percentages = [0.6, 0.3]

  s = input('remove percentage\n:')
  rm_percentages = [float(s)]
  # rm_percentages = [0.98, 0.95, 0.92]
  # acc = test_remove_unforgettable(rm_percentages, '../experiments/seed1/mnist_forgetting.csv',
  #                           batch_size_train, batch_size_test, norm_mean, norm_std, n_epochs, 1, 1,
  #                                 less_intensive=True, sleep_tm=0.3, return_on_interrupt=True)

  # acc = test_remove_unforgettable(rm_percentages, '../experiments/seed1/mnist_forgetting.csv',
  #                           batch_size_train, batch_size_test, norm_mean, norm_std, n_epochs, 1, 1,
  #                                 less_intensive=True, sleep_tm=0.3, return_on_interrupt=True, random_remove=True)
  acc1 = test_remove_unforgettable(rm_percentages, '../experiments/seed1/mnist_forgetting.csv',
                                  batch_size_train, batch_size_test, norm_mean, norm_std, n_epochs, 1, 1,
                                  less_intensive=False, sleep_tm=0.0, return_on_interrupt=True, random_remove=False)

  acc2 = test_remove_unforgettable(rm_percentages, '../experiments/seed1/mnist_forgetting.csv',
                                   batch_size_train, batch_size_test, norm_mean, norm_std, n_epochs, 1, 1,
                                   less_intensive=False, sleep_tm=0.0, return_on_interrupt=True, random_remove=True)

  # with open('../experiments/remove_unforggetable_examples_mnist/rm_accuracies_unforgettable_30_epoch.csv', 'w') as f:
  with open('../experiments/remove_unforggetable_examples_mnist/rm_unforgettable_30_epoch_' + s +'.csv', 'w') as f:
  # with open('../experiments/remove_unforggetable_examples_mnist/rm_accuracies_not_only_unforgettable2.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['remove_percentage', 'accuracy'])
    writer.writeheader()
    writer.writerows(acc1)

  with open('../experiments/remove_unforggetable_examples_mnist/rm_random_30_epoch_' + s + '.csv', 'w') as f:
    # with open('../experiments/remove_unforggetable_examples_mnist/rm_accuracies_not_only_unforgettable2.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['remove_percentage', 'accuracy'])
    writer.writeheader()
    writer.writerows(acc2)
