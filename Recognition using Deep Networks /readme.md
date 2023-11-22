## Name
Xianhe Zhang

## Link
https://github.com/xianhe-zhang/patternRecognition

## Project Description
The project focuses on how to build, train, analyze, and modify a deep network for a recognition
task. Basically, there are a few subtasks, including training a model from scratch and test it,
analyze the model itself by examining its first layer, reuse the pretrained model, and execute
self-designed experiments.

## TOOLS
- vscode
- Python3
- MacOS
- OpenCV
- PyTorch

## How to start running
1. Use `python3 {file}` command line directly to run any .py file you wanna run.

### Code Explaination 
- `build_a_model.py` - trains a  network on digits and test on 1000 with an accuracy over 90%
- `test_a_model.py` - test the trained network on self-handwritten digits
- `analyze_network.py` - print out the 10 weights of the first layer and the images after the weights being applied
- `training_greek.py` - use pre-trained model to identify greek letters.
- `design_exeperiment.py` - pick up 3 dimensions(activation, training size, dropout rate) to conduct experiment to see which set of configuration have better prediction score.

#### Extensions
- `live_digit_recognition.py` - Identify the digit in a live camera system.


## Time Travel
### 2 days

