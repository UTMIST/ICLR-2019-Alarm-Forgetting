import sys
import csv


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
