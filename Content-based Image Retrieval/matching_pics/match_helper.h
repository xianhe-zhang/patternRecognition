#include <opencv2/opencv.hpp>
// baseline match function - match 2 images solely based on the center 9 x 9 pixles
int baseline_match(cv::Mat &image, std::vector<std::pair<char *, float>> &imagevec, char *metric);
// histogram match function -  calculate the similirity of the images histogram.csv with the given target image using intersection metric
int histogram_match(cv::Mat &image, std::vector<std::pair<char *, float>> &imagevec, char *metric);
// multi histogram match function - calculate the similirity of the images (combining the data in the top_histogram.csv 
// and down_histogram.csv with the given target image using intersection metric
int multi_histogram_match(cv::Mat &image, std::vector<std::pair<char *, float>> &imagevec, char *metric);
// multi histogram match function - calculate the similirity of the images (combining the data in the color_histogram.csv 
// and sobel_histogram.csv with the given target image using intersection metric
int color_and_texture_match(cv::Mat &image, std::vector<std::pair<char *, float>> &imagevec, char *metric);
// special function built to calculate how similar the images are with the target image that has banana in them using intersection
// metrics 
int hsv_match(cv::Mat &image, std::vector<std::pair<char *, float>> &imagevec, char *metric);