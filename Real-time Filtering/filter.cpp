#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ctime>
#include <sstream>
#include <iostream>
#include <iomanip>

#include "filter.h"


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
        // src ptr
        cv::Vec3b *rowptr1 = src.ptr<cv::Vec3b>(i-2);
        cv::Vec3b *rowptr2 = src.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *rowptr3 = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *rowptr4 = src.ptr<cv::Vec3b>(i+1);
        cv::Vec3b *rowptr5 = src.ptr<cv::Vec3b>(i+2);
    
        // dest ptr
        cv::Vec3b *tempptr = temp.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
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
        // update relevant ptrs
        cv::Vec3b *rowptr = temp.ptr<cv::Vec3b>(i);
        cv::Vec3b *destptr = dst.ptr<cv::Vec3b>(i);

        for (int j = 2; j < temp.cols-2; j++) {
            for (int k = 0; k < 3; k++) {
                // calculated blured channels
                destptr[j][k] = (1 * rowptr[j-2][k] + 2 * rowptr[j-1][k] + 4 * rowptr[j][k] +
                                2 * rowptr[j+1][k] + 1 * rowptr[j+2][k])/10;
            }
        } 
    }

    return 0;
}


int sobelX3x3( cv::Mat &src, cv::Mat &dst ) {
    cv::Mat temp;    
    temp.create(src.size(), CV_16SC3);

    // loop through src and apply vertical filter
    for (int i = 1; i < src.rows-1; i++) {
        cv::Vec3b *rowptr1 = src.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *rowptr2 = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *rowptr3 = src.ptr<cv::Vec3b>(i+1);
        cv::Vec3s *tempptr = temp.ptr<cv::Vec3s>(i);

        // go through columes
        for (int j = 0; j < src.cols; j++) {
            
            // go though each color channels
            for (int k = 0; k < 3; k++) {
                tempptr[j][k] = (1 * rowptr1[j][k] + 2 * rowptr2[j][k] + 1 * rowptr3[j][k])/4;
            }
        }
    }

    // allocate destination space
    dst.create(temp.size(), CV_16SC3);
    for (int i = 0; i < temp.rows; i++) {
        cv::Vec3s *rowptr = temp.ptr<cv::Vec3s>(i);
        cv::Vec3s *destptr = dst.ptr<cv::Vec3s>(i);
        
        for (int j = 1; j < temp.cols-1; j++) {
            for (int k = 0; k < 3; k++) {
                // calculate sobel x channels 
                destptr[j][k] = -1 * rowptr[j-1][k] + 1 * rowptr[j+1][k];
            }
        } 
    }
    
    return 0;
}


int sobelY3x3( cv::Mat &src, cv::Mat &dst ) {
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    for (int i = 1; i < src.rows-1; i++) {
        // source 
        cv::Vec3b *rowptr1 = src.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *rowptr2 = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *rowptr3 = src.ptr<cv::Vec3b>(i+1);
        // destination 
        cv::Vec3s *tempptr = temp.ptr<cv::Vec3s>(i);

        for (int j = 0; j < src.cols; j++) {            
            for (int k = 0; k < 3; k++) {
                tempptr[j][k] = -1 * rowptr1[j][k] + 1 * rowptr3[j][k];
            }
        }
    }
        
    dst.create(temp.size(), CV_16SC3);

    for (int i = 0; i < temp.rows; i++) {

        cv::Vec3s *rowptr = temp.ptr<cv::Vec3s>(i);
        cv::Vec3s *destptr = dst.ptr<cv::Vec3s>(i);
        
        for (int j = 1; j < temp.cols-1; j++) {
            for (int k = 0; k < 3; k++) {
                destptr[j][k] = (1 * rowptr[j-1][k] + 2 * rowptr[j][k] + 1 * rowptr[j+1][k])/4;
            }
        } 
    }
    
    return 0;
}

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ) {
    dst = sx.clone();

    for (int i = 0; i < sx.rows; i++) {
        
        cv::Vec3s *rowptrsx = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s *rowptrsy = sy.ptr<cv::Vec3s>(i);    
        cv::Vec3s *dstptr = dst.ptr<cv::Vec3s>(i);

        for (int j = 0; j < sx.cols; j++) {
            for (int k = 0; k < 3; k++) {
                
                dstptr[j][k] = sqrt((rowptrsx[j][k] * rowptrsx[j][k]) + (rowptrsy[j][k] * rowptrsy[j][k]));
            }
        }
    }
    return 0;
}


int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ) {
    cv::Mat blur;
    blur5x5(src, blur);
    dst = blur.clone();

    // size of buckets
    int bucketSize = 255/levels;

    for (int i = 0; i < blur.rows; i++) {
        cv::Vec3b *rowptr = blur.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstptr = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < blur.cols; j++) {
            for (int k = 0; k < 3; k++) {
                // quantization
                int xt = rowptr[j][k] / bucketSize;
                dstptr[j][k] = xt * bucketSize;
            }
        }
    }
    return 0;
}

int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold ) {
    // initilize all the useful parameters
    cv::Mat sobelx;
    cv::Mat sobely;
    cv::Mat mag;
    cv::Mat quantize;

    // get the magnitude image
    sobelX3x3(src, sobelx);
    sobelY3x3(src, sobely);
    magnitude(sobelx, sobely, mag);

    // quantize and blur
    blurQuantize(src, quantize, levels);

    // allocate destination space
    dst = quantize.clone();

    for (int i = 0; i < dst.rows; i++) {
        
        // src pointer
        cv::Vec3s *rowptr = mag.ptr<cv::Vec3s>(i);

        // destination pointer
        cv::Vec3b *dstptr = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < dst.cols; j++) {
           // loop thru all the magnitude
            if (rowptr[j][0] > magThreshold || rowptr[j][1] > magThreshold || rowptr[j][2] > magThreshold) {
                // make channels that are greater thatn the threashold black
                dstptr[j][0] = 0;
                dstptr[j][1] = 0;
                dstptr[j][2] = 0;
            }
        }
    }
    return 0;
}

int reverse(cv::Mat &src, cv::Mat &dst) {

    // initilize destination image
    dst = src.clone();
    // loop thru src rows
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *rowptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstptr = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            // make all channels negative
            dstptr[j][0] = 255 - rowptr[j][0];
            dstptr[j][1] = 255 - rowptr[j][1];
            dstptr[j][2] = 255 - rowptr[j][2];
        }
    }
    return 0;
}

int laplacian(cv::Mat &src, cv::Mat &dst) {

    dst.create(src.size(), CV_16SC3);

    for (int i = 1; i < src.rows-1; i++) {
        cv::Vec3b *rowptr1 = src.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *rowptr2 = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *rowptr3 = src.ptr<cv::Vec3b>(i+1);
        cv::Vec3s *dstptr = dst.ptr<cv::Vec3s>(i);

        for (int j = 1; j < src.cols-1; j++) {
            // loop thru color channels
            for (int k = 0; k < 3; k++) {
                dstptr[j][k] =  1 * rowptr1[j][k] +
                                1 * rowptr2[j-1][k] + -4 * rowptr2[j][k] + 1 * rowptr2[j+1][k] +
                                1 * rowptr3[j][k];
            }
        }
    }
    return 0;
}