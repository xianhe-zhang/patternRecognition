#include <opencv2/opencv.hpp>
int baseline_match(cv::Mat &image, std::vector<std::pair<char *, float>> &imagevec, char *metric);
int histogram_match(cv::Mat &image, std::vector<std::pair<char *, float>> &imagevec, char *metric);
int multi_histogram_match(cv::Mat &image, std::vector<std::pair<char *, float>> &imagevec, char *metric);
int color_and_texture_match(cv::Mat &image, std::vector<std::pair<char *, float>> &imagevec, char *metric);
int hsv_match(cv::Mat &image, std::vector<std::pair<char *, float>> &imagevec, char *metric);