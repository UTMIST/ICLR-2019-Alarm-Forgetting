import torch
from time import sleep
import sys

def train_model(net, loader, epochs, batch_size_train, learning_rate, momentum, verbose=False, less_intensive=False, sleep_time=0.5):
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
