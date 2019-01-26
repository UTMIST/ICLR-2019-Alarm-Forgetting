import torch
import os
import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
import torchvision
from six.moves import cPickle

def simple_load_cifar10(train_batch_size):
    train_transforms = torchvision.transforms.Compose([])
    train_transforms.transforms.append(torchvision.transforms.ToTensor())

    train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                   train=True,
                                   transform=train_transforms,
                                   download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)

    return train_loader

SEED = 1
torch.manual_seed(SEED)

folder_prefix = "experiments/seed" + str(SEED) + "/"
filename = folder_prefix + "forgetting_events_cifar10_seed" + str(SEED) + ".csv"

df = pd.read_csv(filename)
sorted_df = df.sort_values(by=['forgetting_events'])

least_forgetting_indices = sorted_df['index_info'][:10].values.tolist()
least_forgetting_indices = [literal_eval(i) for i in least_forgetting_indices]
most_forgetting_indices  = sorted_df['index_info'][-10:].values.tolist()
most_forgetting_indices = [literal_eval(i) for i in most_forgetting_indices]

least_forget_indices_converted = [i*100 + j for (i,j) in least_forgetting_indices]
most_forget_indices_converted = [i*100 + j for (i,j) in most_forgetting_indices]

X = np.zeros((50000, 32, 32, 3))
for i in range(5):
    cifar_filename = 'data/cifar-10-batches-py/data_batch_' + str(i+1)
    with open(cifar_filename, 'rb') as f:
        datadict = cPickle.load(f,encoding='latin1')
        f.close()
    x = datadict["data"]
    x = x.reshape(10000, 3, 32, 32).transpose(0,2,3,1)
    X[10000*i:10000*(i+1),:,:,:] = x.copy()

all_indices = least_forget_indices_converted + most_forget_indices_converted
# Visualizing CIFAR 10
fig, axes1 = plt.subplots(2,10,figsize=(8,8), sharex=True)
for j in range(2):
    if j == 0:
        axes1[j][5].set_title('Top 10 unforgettable Examples')
    if j == 1:
        axes1[j][5].set_title('Top 10 Forgettable Examples')
    for k in range(10):
        m = all_indices[j*10+k]
        axes1[j][k].set_axis_off()
        axes1[j][k].imshow((X[m:m+1][0] * 255).astype(np.uint8))

plt.show()
