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
        std::string image_path = samples::findFile("./input/dog.jpeg");
        Mat colorImage = cv::imread(image_path, IMREAD_COLOR);
    
        while(true) {
            *capdev >> frame;
            // get a new frame from the camera, treat as a stream
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
                cv::Mat grayscaleImage;
                cv::cvtColor(colorImage, grayscaleImage, cv::COLOR_BGR2GRAY); // 转换为灰度图像
                cv::imwrite("./output/img/grayscaleImage_dog.jpeg", grayscaleImage);
            }

            // Task - 4
            if (key == 'h') {
                cv::Mat greyFrame;
                greyscale(colorImage, greyFrame);
                cv::imwrite("./output/img/greyFrame_dog.jpeg", greyFrame);
            }
            
            // Task - 5
            if (key == 'b') {
                cv::Mat gusFrame;
                // convert frame to greyscale using function built from scrach
                blur5x5(colorImage, gusFrame);
                cv::imwrite("./output/img/gus_dog.jpeg", gusFrame);
            }

            // Task - 6   
            if (key == 'x') {
                // initilize parameters
                cv::Mat displayFrameX;
                cv::Mat sobelxFrame;
                sobelX3x3(colorImage, sobelxFrame);
                cv::convertScaleAbs(sobelxFrame, displayFrameX);
                cv::imwrite("./output/img/display_x_dog.jpeg", displayFrameX);
                
            
            }

            if (key == 'y') {
                // initilize parameters
                cv::Mat displayFrameY;
                cv::Mat sobelyFrame;
                sobelY3x3(colorImage, sobelyFrame);
                // convert negative values to positive for display purposes
                cv::convertScaleAbs(sobelyFrame, displayFrameY);
                cv::imwrite("./output/img/display_y_dog.jpeg", displayFrameY);
            }

            // TASK - 7
            if (key == 'm') {
                cv::Mat magFrame;
                cv::Mat displayFrame;
                cv::Mat sobelX;
                cv::Mat sobelY;
                // leverage other functions to implement this feature
                sobelX3x3(colorImage, sobelX);
                sobelY3x3(colorImage, sobelY);
                magnitude(sobelX, sobelY, displayFrame);
                // to make sure converted values are within the range.
                cv::convertScaleAbs(displayFrame, magFrame);
                cv::imwrite("./output/img/mag_dog.jpeg", magFrame);
            }

            // Task - 8
            if (key == 'l' ) {
                // initilize parameters
                cv::Mat quanFrame;
                // call blurQuantize function to filter image
                blurQuantize(colorImage, quanFrame, 10);
                cv::imwrite("./output/img/bq_dog.jpeg", quanFrame);
            }

            // Task - 9
            if (key == 'c') {
                cv::Mat cartoonF;
                cartoon(colorImage, cartoonF, 10, 25);
                cv::imwrite("./output/img/carton_dog.jpeg", cartoonF);        
            }

            // Task - 10
            if (key == 'r') {
                // initilize parameters
                cv::Mat reverseFrame;
                // call negative function to filter image
                reverse(colorImage, reverseFrame);
                cv::imwrite("./output/img/reverse_dog.jpeg", reverseFrame);   
                
            }

            // Extension - additional filter
            if (key == 'n') {
                // initilize parameters
                cv::Mat laplacianFrame;
                cv::Mat finalFrame;
                
                laplacian(colorImage, laplacianFrame);
                // convert negtive to positives
                cv::convertScaleAbs(laplacianFrame, finalFrame);
                cv::imwrite("./output/img/laplacian_dog.jpeg", finalFrame); 
            }
        }

        delete capdev;
        return(0);
}


