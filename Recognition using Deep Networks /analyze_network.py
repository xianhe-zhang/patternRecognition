
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from build_a_model import Net, load_data

# Task 2
def plot_first_layer(data):
    plt.figure()
    for i, weight in enumerate(data):
        plt.subplot(3, 4, i+1)
        plt.imshow(data[i][0])
        plt.title('filter {}'.format(i+1))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def filter2d(weights):
    batch_size = 64
    batch_size_test = 1000
    training_size = 60000

    train_loader, test_loader = load_data(batch_size, batch_size_test, training_size)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    image = example_data[0].numpy()

    print(weights.shape)
    with torch.no_grad():
        j = 1
        for i in range(weights.shape[0]):
            kernal = weights[i][0].astype(np.float32)
            plt.subplot(5, 4, j)
            plt.imshow(kernal, cmap='gray', interpolation='none')
            plt.title('filter {}'.format(i))
            plt.xticks([])
            plt.yticks([])

            plt.subplot(5, 4, (i + 1) * 2)
            output = cv2.filter2D(image, -1, kernal)
            output = output.reshape([28, 28])
            plt.imshow(output, cmap='gray', interpolation='none')
            plt.title('output {}'.format(i))
            plt.xticks([])
            plt.yticks([])
            j += 2
        plt.show()
        
def main():
    model = Net()
    first_layer_weight = model.conv1.weight.detach().numpy()
    plot_first_layer(first_layer_weight)
    filter2d(first_layer_weight)
    print(first_layer_weight)

if __name__ == '__main__':
    main()
