
import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def plot(train_counter, train_losses, test_counter, test_losses):
    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


# used to show the sample images
def show(mnist_dataset, example_targets, no_images, truth_or_pred):
    plt.figure()
    for i in range(no_images):
        # set the plot to be 2 by 3
        plt.subplot(int(no_images / 3), 3, i + 1)
        # set it to be a tight plot
        plt.tight_layout()
        # set a few parameters
        plt.imshow(mnist_dataset[i][0], cmap='gray', interpolation='none')
        plt.title("{}: {}".format(truth_or_pred, example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def load_data(batch_size, batch_size_test, training_size):
    train_loader = torch.utils.data.DataLoader(
        Subset(torchvision.datasets.MNIST(  './data', 
                                            train=True, download=True,
                                            transform =   torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                          ])), 
                                            range(training_size)),
        batch_size = batch_size, 
        shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', 
                                   train=False, 
                                   download=True,
                                   transform =  torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, 
        shuffle=True
        )
    return train_loader, test_loader

# training method to train the network to identify digits on a dataset of 60000 digit images
def train(epoch, network, optimizer, train_loader, train_losses, train_counter, log_interval, flag):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            if flag == 'main':
                torch.save(network.state_dict(), './models/model.pth')
                torch.save(optimizer.state_dict(), './models/optimizer.pth')


# testing method to test the accuracy of the trained mode with 1000 images
def test(network, test_loader, test_losses):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    epochs = 5
    training_size = 25000
    batch_size = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 42

    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False

    train_loader, test_loader = load_data(batch_size, batch_size_test, training_size)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(f'sample_data.shape:\n{example_data.shape}')
    show(example_data, example_targets, 6, 'Ground Truth')
    nn = Net()
    optimizer = optim.SGD(nn.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    # evaluate the model first before training
    test(nn, test_loader, test_losses)
    for epoch in range(1, epochs + 1):
        train(epoch, nn, optimizer, train_loader, train_losses, train_counter, log_interval, flag)
        test(nn, test_loader, test_losses)

    # training and testing plot
    plot(train_counter, train_losses, test_counter, test_losses)


if __name__ == '__main__':
    main()
