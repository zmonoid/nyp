import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('log', metavar='LOG', help='path to log')
args = parser.parse_args()

val_loss = []
train_loss = []

with open(args.log, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if '*' not in line:
            continue

        if 'Train' in line and 'Loss' in line:
            train_loss.append(line.split(' ')[-1])
        elif 'Val' in line and 'Loss' in line:
            val_loss.append(line.split(' ')[-1])

val_loss = map(float, val_loss)
train_loss = map(float, train_loss)

plt.plot(train_loss)
plt.plot(val_loss)
plt.show()
plt.savefig('haha.png')
