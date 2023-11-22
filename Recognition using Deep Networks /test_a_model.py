
from build_a_model import load_data, Net, show

import torch.optim as optim
import torch
import torchvision.models as models
import torch.nn as nn
import cv2
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def threshold_img(file):
    folder_path = 'handwritten_digits'
    img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
    thresh_val = 128
    result, thresh = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
    return cv2.bitwise(thresh)
# Task 1-E
def predict(model, data, targets):
    predictions = list()
    with torch.no_grad():
        for i in range(len(data)):
            img = data[i]
            target = targets[i]
            output = model(img)
            _, predicted = torch.max(output,1)
            pred = predicted.item()
            predictions.append(pred)
            print("target/predicted: {}/{}".format(target, pred))
            for j in range(output.size(1)):
                print(f"Output feature {j}: {output[0][j]:.2f}")
    return predictions

# Task 1-F
def predict_handwritten(model):
    h_data = []
    h_target = [i for i in range(10)]

    folder_path = 'handwritten_digits'
    filenames = os.listdir(folder_path)
    sorted_filenames = sorted(filenames)
    for f in sorted_filenames:
        img = threshold_img(f)
        tensor = transforms.ToTensor()(img)
        h_data.append(tensor)

    predictions = predict(model, h_data, h_target)
    plot(h_data,predictions,10, 'Prediction')

def plot(mnist_dataset, example_targets, no_images, truth_or_pred):
    plt.figure()
    for i in range(no_images):
        # set the plot to be 2 by 3
        plt.subplot(4, 3, i+1)
        # set it to be a tight plot
        plt.tight_layout()
        # set a few parameters
        plt.imshow(mnist_dataset[i][0], cmap='gray', interpolation='none')
        plt.title("{}: {}".format(truth_or_pred, example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def main():
    batch_size = 64
    batch_size_test = 1000

    model = Net()
    model.eval()
    network_stat = torch.load('./models/model.pth')
    model.load_state_dict(network_stat)
    train_loader, test_loader = load_data(batch_size, batch_size_test)
    test = enumerate(test_loader)
    batch_idx, (test_data, test_targets) = next(test)
    predictions = predict(model, test_data, test_targets)
    show(test_data, predictions, 9, 'Predication')
    predict_handwritten(model)

if __name__ == '__main__':
    main()