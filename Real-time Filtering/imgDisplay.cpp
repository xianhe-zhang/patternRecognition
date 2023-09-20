#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>


using namespace cv;


int main() {
    std::string image_path = samples::findFile("./input/dog.jpeg");
    Mat img = imread(image_path, IMREAD_COLOR);

    if(img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    imshow("Dog view", img);
    int k = waitKey(0);

    while (true) {
        if (k == 'q' || k == 'Q') {
            break;
        }

        if(k == 's'){
            imwrite("./output/img/dog.jpeg", img);
        }
    }
    return 0;
}