# the entire structure is from https://github.com/uoguelph-mlrg/Cutout

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000
CUTOUT_LEN = 16
EPOCHS = 1
LR = 0.1
TORCH_SEED = 1



def conv3x3(in_planes, out_planes, stride=1):
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(in_planes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion * planes)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, in_planes, planes, stride=1):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(self.expansion * planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion * planes)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out


class ResNet(nn.Module):
  def __init__(self, block, num_blocks, num_classes=10):
    super(ResNet, self).__init__()
    self.in_planes = 64

    self.conv1 = conv3x3(3, 64)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    self.linear = nn.Linear(512 * block.expansion, num_classes)

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out


def ResNet18(num_classes=10):
  return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def train_net(net, train_loader, n_epoch, lr, torch_seed, use_gpu=False, momentum=0.9, verbose=True):
  # the use_gpu functionality not implemented yet
  torch.manual_seed(torch_seed)
  loss = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                  momentum=momentum, nesterov=True, weight_decay=5e-4)

  prev_acc = {}
  acc = {}

  for epoch in range(n_epoch):
    for i, (data, labels) in enumerate(train_loader):
      net.zero_grad()
      y_pred = net(data)

      xentropy_loss = loss(y_pred, labels)
      xentropy_loss.backward()
      optimizer.step()

      if verbose:
        print(xentropy_loss, i, epoch)


def verify_net(net, test_loader, verbose=True):
  correct = 0
  total = 0
  for i, (dts, labels) in enumerate(test_loader):
    y_pred = net(dts)
    pred = torch.max(y_pred.data, 1)[1]
    total += labels.size(0)
    correct += (pred == labels.data).sum().item()
    if verbose:
      print(correct, total, i)
  return correct / total

if __name__ == '__main__':
  import data_loaders
  nn = ResNet18()
  train_dt_loader, test_dt_loader = data_loaders.load_cifar_10(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, cutout_len=CUTOUT_LEN)
  print(len(train_dt_loader.dataset), len(test_dt_loader.dataset))
  # train_net(nn, train_dt_loader, EPOCHS, LR, TORCH_SEED)
  # verify_net(nn, test_dt_loader)

