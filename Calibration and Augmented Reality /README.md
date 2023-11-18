## Name
Xianhe Zhang

## Link
https://github.com/xianhe-zhang/patternRecognition

## Project Description
This project deals a lot with the chessboard, how to detect the corner, how to project, and eventually build a augmented reality object on the 2D frame.

## TOOLS
- vscode
- make
- MacOS

## How to start running
1. `make`  in your terminal. This will allow you to have two executable files.
2. run `./main` to start the project

### Complete operation guide  
- Required:
  - Press `s` : save the current corner/point data and image. Be sure to add at least 5 sets of data before calibrating the camera.
  - Press `c` : Calibrate the camera, will also update camera matrix and distortion coefficients, which are necessary to project objects.
  - Press `f` : Caculate SFIT feature and draw them
  - Press `p` : Calculate camera position and project 3D Axes and custom shape.
  - Press `r` : Clear both effect of SFIT feature and projection.
  - Press `q` : Quit the project
- Extension
  - ArUco: This is automatically integrated into the feature to improve augmented reality quality.

## Time Travel
### 4 days (1 day left)

