#include <opencv2/opencv.hpp>
#include "functions.h"


int main(int argc, char* argv[]) {
    cv::VideoCapture *capdev;
    
    capdev = new cv::VideoCapture(0);

    if (!capdev -> isOpened()) {
        std::cout << "Unable to open cam" << std::endl;
        return -1;
    }  
    cv::Size refs((int)capdev -> get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev -> get(cv::CAP_PROP_FRAME_HEIGHT));
    
    printf("Video size %d %d\n", refs.width, refs.height);

    cv::namedWindow("P4", 1);
    cv::Mat frame;


    std::vector<cv::Point2f> corner_set;
    std::vector<cv::Vec3f> point_set;
    std::vector<std::vector<cv::Point2f>> corner_list;
	std::vector<std::vector<cv::Vec3f>> point_list;
	

    cv::Mat camera_matrix;
    cv::Mat distortion_coefficients;


    char state;
    int numOfArg = argc;

    while(true) { 
        if (numOfArg > 1) {
            frame = cv::imread(argv[1]);
        } else {
            *capdev >> frame;
            if(frame.empty()) {
            std::cout << "cannot get the frame from camera" << std::endl;
            return -1;
        }
        }

        char key = cv::waitKey(100);
        // bool isFound = canGetChessboardConners(frame, key, )
        bool ifProcess = process_chessboard_corners(frame, corner_set, point_set);
        if (!ifProcess) {
            continue;
        }
        //Extension
        aruco(frame);
        
        // 0. 检测+计算  ✅
        // 1. s -> store image -> only write into ✅
        // 2. c -> 初始化/计算/保存 


        // 3. p -> Project / show 3D 两个task放在一起呗 无所谓
        // 4. f -> SIFT feature
        


        // Task 2
        // to add 
        if (key == 's' || key == 'S') {
            if (corner_set.empty()) {
                std::cout << "corner set is empty, please retry" << std::endl;
            } else {
                std::cout << "Ready to add more corner set, corner_lise size before: " << corner_list.size() << std::endl;
                save_calibration_image(corner_set, point_set, corner_list, point_list);
                std::cout << "Corner Added Successfully, corner_list size: " << corner_list.size() << std::endl;
                cv::imwrite("highlighted.png", frame);
            }
        }  

        if (key == 'c' || key == 'C') {
            if (corner_list.size() <= 5) {
                std::cout << "Clibration Failed due to insufficient images, please press 's' to add more than 5 images, now have: " << corner_list.size() << std::endl;
            } else {
                cv::Mat calibration_frame = frame.clone();
                calibrate_camera(calibration_frame, corner_list, point_list, camera_matrix, distortion_coefficients);
            }
        } 

        if (key == 'p' || key == 'P' || state == 'p' || state == 'P') {
            calculate_and_project(frame, corner_set, point_set);
            state = 'p';
        }


        if (key == 'f' || key == 'F' || state == 'f' || state == 'F') {
            get_sift_feature(frame);
            state = 'f';
        }

        if (key == 'r' || key == 'R') {
            state = '\0';
        }
        if (key == 'q' || key == 'Q') {
            break;
        }
        cv::imshow("P4", frame);

    }


    return 0;
}