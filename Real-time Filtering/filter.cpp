#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ctime>
#include <sstream>
#include <iostream>
#include <iomanip>

#include "filter.h";


int greyscale(cv::Mat &src, cv::Mat &dst) {
    cv::Mat channels[3];
    cv::split(src, channels);
    cv::Mat b = channels[0];
    cv::Mat g = channels[1];
    cv::Mat r = channels[2];
    channels[1] = channels[0];
    channels[2] = channels[0];
    cv::merge(channels, 3, dst);
    std::cout << "greyscale() complete"<< std::endl;
    return 0;
}


int blur5x5(cv::Mat &src, cv::Mat &dst) {

    // allocate temp space
    cv::Mat temp;
    temp = src.clone();

    // loop through src and apply vertical filter
    // go through rows
    for (int i = 2; i < src.rows-2; i++) {
        
        // source row pointer
        cv::Vec3b *rowptr1 = src.ptr<cv::Vec3b>(i-2);
        cv::Vec3b *rowptr2 = src.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *rowptr3 = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *rowptr4 = src.ptr<cv::Vec3b>(i+1);
        cv::Vec3b *rowptr5 = src.ptr<cv::Vec3b>(i+2);
    
        // destination pointer
        cv::Vec3b *tempptr = temp.ptr<cv::Vec3b>(i);

        // go through columes
        for (int j = 0; j < src.cols; j++) {
    
            // go though each color channels
            for (int k = 0; k < 3; k++) {
                tempptr[j][k] = (1 * rowptr1[j][k] + 2 * rowptr2[j][k] + 4 * rowptr3[j][k] +
                                2 * rowptr4[j][k] + 1 * rowptr5[j][k])/10;
            }
        } 
        
    }

    // allocate destination space
    dst = temp.clone();

    // loop through temp and apply horizontal filter
    // go through cols
    for (int i = 0; i < temp.rows; i++) {
        
        // temp col pointer
        cv::Vec3b *rowptr = temp.ptr<cv::Vec3b>(i);
    
        // destination pointer
        cv::Vec3b *destptr = dst.ptr<cv::Vec3b>(i);

        // go through rows
        for (int j = 2; j < temp.cols-2; j++) {
    
            // go though each color channels
            for (int k = 0; k < 3; k++) {
                // calculated blured channels
                destptr[j][k] = (1 * rowptr[j-2][k] + 2 * rowptr[j-1][k] + 4 * rowptr[j][k] +
                                2 * rowptr[j+1][k] + 1 * rowptr[j+2][k])/10;
            }
        } 
    }

    return 0;
}


