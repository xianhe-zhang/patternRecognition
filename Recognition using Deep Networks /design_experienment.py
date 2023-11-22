import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from build_a_model import load_data, train



class Net(nn.Module):
    # initiate the network
    def __init__(self, activation_fn, drop_out_rate):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout(drop_out_rate)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.activation_fn = activation_fn

    # forward pass of the network
    def forward(self, x):
        x = self.activation_fn(F.max_pool2d(self.conv1(x), 2))
        x = self.activation_fn(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    



# show the first few example of the training data
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


def test(network, test_loader, test_losses, variable_tuple, epoch, record):
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
    acc = 100. * correct / len(test_loader.dataset)
    if epoch == 3:
        record[variable_tuple] = round(acc.item(), 4)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))


# plot the different variations of the testing result of the network
def plot(train_counter, train_losses, test_counter, test_losses):
    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


# main/driver method of the experiment
def main():
    # some parameters
    n_epochs = 3
    batch_size = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 42
    flag = 'experiment'

    # eliminate randomness while developing
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False

    activation_list = ['relu', 'sigmoid', 'tanh', 'hardswish']
    activation_fn_list = [F.relu, F.sigmoid, F.tanh, F.hardswish]
    dropout_rate_list = [0.2, 0.3, 0.4, 0.5]
    training_data_size = [5000, 10000, 30000, 60000]

    record = {}
    for i in range(len(activation_fn_list)):
        for j in range(len(dropout_rate_list)):
            for k in range(len(training_data_size)):
                # load the mnist dataset
                train_loader, test_loader = load_data(batch_size, batch_size_test, training_data_size[k])
                examples = enumerate(test_loader)
                batch_idx, (example_data, example_targets) = next(examples)
                print(example_data.shape)

                # plot the first 6 digit images
                # show(example_data, example_targets, 6, 'Ground Truth')

                # initialize the network

                network = Net(activation_fn_list[i], dropout_rate_list[j])
                optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

                train_losses = []
                train_counter = []
                test_losses = []
                test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

                variable_tuple = (activation_list[i], dropout_rate_list[j], training_data_size[k])
                print('activation function: {}, dropout rate: {}, training data size: {}'.format(activation_list[i],
                                                                                                 dropout_rate_list[j],
                                                                                                 training_data_size[k]))

                # evaluate the model first before training
                test(network, test_loader, test_losses, variable_tuple, 1, record)
                for epoch in range(1, n_epochs + 1):
                    train(epoch, network, optimizer, train_loader, train_losses, train_counter, log_interval, flag)
                    test(network, test_loader, test_losses, variable_tuple, epoch, record)
                # training and testing plot
                # plot(train_counter, train_losses, test_counter, test_losses)

    sorted_dict = dict(sorted(record.items(), key=lambda item: item[1], reverse=True))

    for idx, ((activation, dropout_rate, training_size), value) in enumerate(sorted_dict.items()):
        print('rank: {} -> variation: ({}, {}, {}), accuracy: {}%'.format((idx + 1), activation, dropout_rate,
                                                                          training_size, value))
    # Print the sorted dictionary as a dictionary
    print(sorted_dict)


# start of this program
if __name__ == '__main__':
    main()