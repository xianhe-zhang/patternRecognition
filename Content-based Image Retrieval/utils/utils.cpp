#include <opencv2/opencv.hpp> 
#include "csv_util.h"

// Split img into 2 halfs.
int half_image(cv::Mat &image, cv::Mat &half_1, cv::Mat &half_2) {
    int height = image.size().height;
    half_1 = image(cv::Rect(0, 0, image.size().width, height/2));
    half_2 = image(cv::Rect(0, height/2, image.size().width, height/2));
    return 0;
}

// To create RGB histograms
int get_RGB_Hist(cv::Mat &src, cv::Mat &hist) {
    const int size = 8;
    const int divisor = 256 / size;
    int i, j;
    // init 3D matrix
    int dimensions[3] = {size, size, size};
    hist = cv::Mat::zeros(3, dimensions, CV_32S);

    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *sptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            int r = sptr[j][2] / divisor; 
            int g = sptr[j][1] / divisor; 
            int b = sptr[j][0] / divisor; 
            hist.at<int>(r, g, b)++;
        }
    }
    return 0;
}

// calculate the histogram of sobel image
int get_magnitude_hist(cv::Mat &image, std::vector<float> &magnitude_vect) {
    cv::Mat gray;
    cv::Mat hist;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    // Apply the Sobel filter
    cv::Mat sobelx, sobely;
    cv::Sobel(gray, sobelx, CV_32F, 1, 0);
    cv::Sobel(gray, sobely, CV_32F, 0, 1);
    // Compute the magnitude and direction of the gradient
    cv::Mat magnitude, direction;
    cv::cartToPolar(sobelx, sobely, magnitude, direction, true);
    const int size = 256;
    const int divisor = 256 / size;
    int i, j, k;
    int dimention[1] = {size};
    hist = cv::Mat::zeros(1, dimention, CV_32S);

    // loop through the image and assign the pixels to bins
    for (int i = 0; i < gray.rows; i++) {
        cv::Vec3b *sptr = gray.ptr<cv::Vec3b>(i);
        for (int j = 0; j < gray.cols; j++) {
            int n = sptr[j][0] / divisor; // R index
            hist.at<int>(n)++;

        }
    }
    for (int i = 0; i < size; i++) {
        magnitude_vect.push_back((float)hist.at<int>(i));
    }

    return 0;
}

// This is to convert histogram into 1D vecrtor
int convert_hist_to_1d(cv::Mat &hist, std::vector<float> &vec) {
    for (int i = 0; i < 512; i++) { vec.push_back((float) hist.at<int>(i)); }
    return 0;
}

// This is to calculate the intersection of 2 Histogram
float intersection (std::vector<float> &ha, std::vector<float> &hb) {
    double intersection = 0.0;
    double ta = 0.0, tb = 0.0; // Total
    for (int i = 0; i < ha.size(); i++) { 
        ta += ha[i];
        tb += hb[i];
    }
    for (int i = 0; i < ha.size(); i++) {
        double af = ha[i] / ta;
        double bf = hb[i] / tb;
        intersection += af < bf ? af : bf;
    }
    return (1 - intersection);
} 

// toGet
float get_ssd(std::vector<float> &ha, std::vector<float> &hb) {
    float distance = 0;
    for (int i = 0; i < hb.size(); i++) {
        distance += pow((ha[i] - hb[i]), 2);
    }
    return distance;
}


// This is to get HSV metrics for each img.
int get_hsv(cv::Mat &src, cv::Mat &hsv) {
    
    // init all vars we need.
    const int size = 8;
    const int divisor = 256 / size;
    int i, j;
    int dimensions[3] = {size, size, size};
    hsv = cv::Mat::zeros(3, dimensions, CV_32S);

    // loop to assign pixels into buckets.
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *sptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            int hue = sptr[j][0] / divisor; 
            int saturation = sptr[j][1] / divisor;
            int value = sptr[j][2] / divisor;
            hsv.at<int>(hue, saturation, value)++;
        }
    }

    return 0;
}