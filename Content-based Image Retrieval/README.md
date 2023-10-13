## Name
Xianhe Zhang

## Link
https://github.com/xianhe-zhang/patternRecognition

## TOOLS
- vscode
- make
- MacOS

## How to run
1. `make`  in your terminal. This will allow you to have two executable files.
2. run `output/build` to canculate features of all imgs.
3. run `output/match` with **other args** (see below)

### Complete guide to run `output/match`  

The complete cmd should look like `match ${target_img_name} ${matching_method} ${data_base} ${distance} ${num_of_imgs} `   

- target_img_name 
- matching_method
  - baseline
  - histogram
  - multihistogram
  - texture
  - hsv (extension)
- data_base (olympus)
- distance
  - squared_distance
  - intersection
- num_of_imgs



## Time Travel
### 3 days (5 left)