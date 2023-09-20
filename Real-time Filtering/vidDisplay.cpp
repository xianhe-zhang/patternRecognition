#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ctime>
#include <sstream>
#include <iostream>
#include <iomanip>

#include "filter.h"

using namespace cv;



int main(int argc, char *argv[]) {
        cv::VideoCapture *capdev;
        
        // open the video device
        capdev = new cv::VideoCapture(0);

        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }

        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);

        cv::namedWindow("Video", 1); // identifies a window
        cv::Mat frame;

        for(;;) {
                *capdev >> frame; // get a new frame from the camera, treat as a stream
                if( frame.empty() ) {
                  printf("frame is empty\n");
                  break;
                }                
                cv::imshow("Video", frame);

                // see if there is a waiting keystroke
                char key = cv::waitKey(10);
                if( key == 'q') {
                    break;
                } 
                
                
                // TASK - 2
                if( key == 's') {
                    std::time_t now = std::time(nullptr);
                    std::tm *ltm = std::localtime(&now);
                    std::stringstream filepath;
                    filepath << "./output/img/" 
                        << std::setw(2) << std::setfill('0') << ltm->tm_hour << "-" 
                        << std::setw(2) << std::setfill('0') << ltm->tm_min << "-" 
                        << std::setw(2) << std::setfill('0') << ltm->tm_sec << ".jpg"; 

                    std::cout << filepath.str() << std::endl;
                    // 保存图片
                    cv::imwrite(filepath.str(), frame);
                    printf("Frame saved to %s\n", filepath.str().c_str());
                }

                // Task - 3
                if (key == 'g') {
                    // use opencv function to change img to greyscale
                    // cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

                    std::string image_path = samples::findFile("./input/dog.jpeg");
                    Mat colorImage = cv::imread(image_path, IMREAD_COLOR);         
                    cv::Mat grayscaleImage;
                    cv::cvtColor(colorImage, grayscaleImage, cv::COLOR_BGR2GRAY); // 转换为灰度图像
                    cv::imwrite("./output/img/dog.jpeg", grayscaleImage);
                    
                }

                // Task - 4
                if (key == 'h') {
                    cv::Mat greyFrame;
                    greyscale(frame, greyFrame);
                    frame = greyFrame;
                }
                
        }

        delete capdev;
        return(0);
}