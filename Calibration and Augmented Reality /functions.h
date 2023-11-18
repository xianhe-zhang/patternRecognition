#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#include <iostream>
#include <fstream>


// Task1 - Detect and Extract Chessboard Corners
bool process_chessboard_corners(cv::Mat &frame, std::vector<cv::Point2f> &corner_set, std::vector<cv::Vec3f> &point_set);

// Task2 - Select Calibration Images
int save_calibration_image(std::vector<cv::Point2f> &corner_set, std::vector<cv::Vec3f> &point_set, std::vector<std::vector<cv::Point2f>> &corner_list, std::vector<std::vector<cv::Vec3f>> &point_list);

// Task3 - Calibrate the Camera
int calibrate_camera(cv::Mat &frame, std::vector<std::vector<cv::Point2f>> &corner_list, std::vector<std::vector<cv::Vec3f>> &point_list, cv::Mat &camera_matrix , cv::Mat &distortion_coefficients);


// Task456 - Calculate & Project 
int calculate_and_project(cv::Mat &frame, std::vector<cv::Point2f> &corner_set, std::vector<cv::Vec3f> &point_set);

// Taks7 - Calculate SFIT Feature
int get_sift_feature(cv::Mat &frame);

// Extension - ArUco
int aruco(cv::Mat &frame);
