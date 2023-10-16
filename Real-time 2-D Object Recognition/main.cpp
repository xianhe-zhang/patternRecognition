#include <opencv2/opencv.hpp>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <algorithm>
#include <vector>
#include <map>
#include <filesystem>

#include "./utils/csv_util.h"
#include "functions.h"



bool sortByValue(const std::pair<char *, float> &a, const std::pair<char *, float> &b) {
    return a.second < b.second;
}
// 1. how to threshold the input value
// 2. morphological process -> erosion/dialation to clean up binary image
// 3. segment the imgae into regions
// 4. calculate each region's feature


// 5. collect training data -> 按下按键，可以存放入新的training set
// 6. implement Euclidean distance metric 
// 7. another different K-nearest classifier 
// 8. evaluate -> confusion matrix
// 9. Video


int main(int argc, char *argv[]) { 

    cv::VideoCapture *capdev;
    capdev = new cv::VideoCapture(0);

    if( !capdev->isOpened() ) {
        printf("Unable to open video device\n");
        return(-1);
    }

    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                   (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame;


    std::vector<std::pair<char *, double>> result; // vector to store the result.

    // int components = 1; // the val is especially for Task3/4

    for(;;) {
        char csv_folder[512] = "database/";


        *capdev >> frame;
        if(frame.empty()) {
            std::cout << "cannot get the frame from camera" << std::endl;
            return -1;
        }

        // Task-1: Threshold image to seperate foreground and background
        cv::Mat bin_img;
        threshold(frame, bin_img);

        // Task-2: Clean up image that we get from thresholding.
        cv::Mat d_img;
        cv::Mat d_clean_img;
        cv::Mat e_img;
        cv::Mat e_clean_img;

        grassfire(bin_img, d_img, 255);
        clean(d_img, d_clean_img, "dialation");
        grassfire(d_clean_img, e_img, 0);
        clean(e_img, e_clean_img, "erosion");
    

        // Task-3&-4: Optional
        

        // even we don't have to calculate for seperated regions as Task 3/4 required, but we do need to get the feature scores for each frame.
        std::vector<double> features; 
        char flag;
        char key = cv::waitKey(10);


        getFeatureScores(frame, features);

        if ( key == 'n' ) {
            char *object = new char[256];
            // Ask user to enter the name
            std::cout << "Enter a name for the file: ";
            std::cin.getline(object, 256);
            std::cout << std::endl;

            // to complete the object name.
            std::strcat(csv_folder, object);
            std::strcat(csv_folder,".csv");

            append_image_data_csv(csv_folder, object, features, 0);
            getStd(); // mysterious.
        }

        if ( key == 'c' || key == 'k' || flag == 'c' || flag == 'k' ) {
            cv::Point p1(50,50);
            cv::Point p2(50,100);
            cv::Scalar font_color(255, 255, 255);
            int font_face = cv::FONT_HERSHEY_COMPLEX;
            double font_scale = 1;
            int thickness = 2;

            if (key == 'c' || flag == 'c') {
                getScaledEuclidean(features, result);
                cv::putText(frame, "SED", p2, font_face, font_scale, font_color, thickness);
                flag = 'c';
            }

            if (key == 'k' || flag == 'k') {
                getKNs(features, result);
                cv::putText(frame, "KNN", p2, font_face, font_scale, font_color, thickness);
                flag = 'k';
            }
            sort(result.begin(), result.end(), sortByValue);
            std::vector<char *> sorted_list;
            for (auto &p : result) {
                sorted_list.push_back(p.first);
            }
            cv::putText(frame, sorted_list[0], p1, font_face, font_scale, font_color, thickness);
        }   
        
        if ( key == 'r' ) { 
            flag = '0';
            result.clear();
            result.resize(0);
        }
        
        if ( key == 'b' || flag == 'b') {
            frame = bin_img;
            flag = 'b';
        }

        if ( key == 'e' || flag == 'e' ) {
            frame = e_clean_img;
            flag = 'e';
        }

        if ( key == 's' ) {
            cv::imwrite("save_capture.png", frame);
            std::string filename;
            std::string png = ".png";
            while (true) {
                std::cout << "enter img name please: ";
                std::getline(std::cin, filename);
                int res = rename("save_capture", (filename + png).c_str());
                if (res == 0) { break; }
            }
        }

        cv::imshow("WOW!", frame); 
        if (key == 'q' || key == 'Q') { 
            cv::destroyAllWindows();
            break;
        }

        if ( key == 'f' ) {
            getConfusionMatrix();
        }

    }


    return (0);
}
