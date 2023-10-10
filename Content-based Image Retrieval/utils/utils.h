#include <opencv2/opencv.hpp> 
// Split img into 2 halfs.
int half_image(cv::Mat &image, cv::Mat &half_1, cv::Mat &half_2);
// To create RGB histograms
int get_RGB_Hist(cv::Mat &src, cv::Mat &res);

// This is to convert histogram into 1D vecrtor
int convert_hist_to_1d(cv::Mat &src, std::vector<float> &res);

// This is to calculate the intersection of 2 Histogram
float intersection(std::vector<float> &h1, std::vector<float> &h2);
// calculat the hsv histogram
int get_hsv(cv::Mat &image, cv::Mat &hsv_hist);

// This is to get HSV metrics for each img.
float get_ssd(std::vector<float> &ha, std::vector<float> &hb);

// calculate the histogram of sobel image
int get_magnitude_hist(cv::Mat &magnitude, std::vector<float> &magnitude_vect);