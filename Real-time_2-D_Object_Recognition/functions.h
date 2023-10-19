#include <opencv2/opencv.hpp>

// To threshold the img to seperate fore/back-ground.
int threshold(cv::Mat &src, cv::Mat &dest); 
// To implement grassfire transform.
int grassfire(cv::Mat& input, cv::Mat& output, int morph);
// To implememt morphological processing.
int clean (cv::Mat &src, cv::Mat &dest, char* morph);
// To calculate feature scores and store them into a csv file
int getFeatureScores(cv::Mat& img, cv::Mat& bin_img, cv::Mat& labels, std::vector<double> &features, int component);
// To get standard deviation.
int getStd();
// To get scaled euclidean distance.
int getScaledEuclidean(std::vector<double>& src_vector, std::vector<std::pair<char*, double>>& result);
// To get K-Nearest neighbors.
int getKNs(std::vector<double>& src_vector, std::vector<std::pair<char*, double>>& result);
// To get the confusion matrix to evaluate model performance.
int getConfusionMatrix();
// To conduct connected component analysis.
int conductCCA(cv::Mat &bin_img, cv::Mat &output, cv::Mat &labels, int num_of_components, int size);
