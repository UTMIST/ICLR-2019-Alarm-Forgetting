import csv
import ast
import resnet18
import data_loaders
from random import seed, sample
import torch

RD_SEED = 1
TORCH_SEED = 1

TRAIN_BATCH_SIZE = 100
TEST_BATCH_SIZE = 100
CUTOUT_LEN = 16
EPOCHS = 50
LR = 0.1
ORIGINAL_BATCH_SIZE = 100
USEGPU = True

def read_forgetting_events(csv_fn, original_batch_size=100):
  pts = []
  with open(csv_fn, mode='r') as f:
    reader = csv.DictReader(f)
    for r in reader:
      pts.append({'index_info': ast.literal_eval(r['index_info']),
                  'forgetting_events': ast.literal_eval(r['forgetting_events']),
                  'index': recover_index(ast.literal_eval(r['index_info']), original_batch_size)})
  return pts

def recover_index(index_info, batch_size):
  return index_info[0] * batch_size + index_info[1]


def find_remove_index(percentage, pts, is_random):
  pts_to_remove = int(percentage * len(pts))
  if pts_to_remove > len(pts):
    raise ValueError('percentage too high')
  if not is_random:
    pts.sort(key=lambda i: i['forgetting_events'])
    return [pts[i]['index'] for i in range(0, pts_to_remove)]
  else:
    return [p['index'] for p in sample(pts, pts_to_remove)]

def train_after_remove(net, n_epoch, lr, torch_seed, use_gpu, train_batch_size,
                       test_batch_size, rm_percentage, remove_by_random,
                       csv_fn, do_cutout=True, cutout_len=10, cutout_holes=1,
                       momentum=0.9, verbose=True):
  dt_pts=read_forgetting_events(csv_fn, original_batch_size=ORIGINAL_BATCH_SIZE)
  to_rm = find_remove_index(rm_percentage, dt_pts, remove_by_random)
  train_loader, test_loader = data_loaders.load_partial_cifar_10(train_batch_size, test_batch_size, to_rm,
                                                                 do_cutout, cutout_len, cutout_holes)
  resnet18.train_net(net, train_loader, n_epoch, lr, torch_seed, use_gpu, momentum, verbose, collect_stat=False)
  acc = resnet18.verify_net(net, test_loader, False, use_gpu)
  return {'remove_percentage': rm_percentage, 'remove_by_random': remove_by_random, 'accuracy': acc}

if __name__ == '__main__':
  seed(RD_SEED)
  percentage = float(input('input remove percentage:\n'))
  remove_randomly = input('press y to remove randomly:\n') == 'y'
  fn = '../experiments/seed7/200epoch_cifar10_forgetting_conv_net_B_seed7.csv'
  if USEGPU:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
  nn = resnet18.ResNet18()

  print(train_after_remove(nn, EPOCHS, LR, TORCH_SEED, USEGPU, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, percentage,
                           remove_randomly, fn, True, cutout_len=CUTOUT_LEN, verbose=True))

  pass
