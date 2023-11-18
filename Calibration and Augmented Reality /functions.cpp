#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#include <iostream>
#include <fstream>


// Task1 - Detect and Extract Chessboard Corners
bool process_chessboard_corners(cv::Mat &frame, std::vector<cv::Point2f> &corner_set, std::vector<cv::Vec3f> &point_set) {
    int WIDTH = 6;
    int LENGTH = 9;
    
    cv::Size pattern_size(WIDTH, LENGTH);  // call the constructor 
    
    // frame -> grayscale
    cv::Mat gray;
    std::cout << "p1" << std::endl;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    std::cout << "p2" << std::endl;
    // Detect corners and store those corners into Set.
    bool pattern_found = cv::findChessboardCorners(gray, pattern_size, corner_set, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
    std::cout << "p3" << std::endl;
    if(pattern_found) {
        cv::cornerSubPix(gray, corner_set, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 30, 0.1));
        std::cout << "p4" << std::endl;
        cv::drawChessboardCorners(frame, pattern_size, cv::Mat(corner_set), pattern_found);
        std::cout << "Number of corners detected: " << corner_set.size() << std::endl;
        std::cout << "Coordinates of the first corner: " << "(" << corner_set[0].x << "," << corner_set[0].y << ")" << std::endl;
        // with new corner set, to calculate new point set
        cv::Vec3f point_3d(0.0f, 0.0f, 0.0f);
        if (point_set.empty()) {

            for (int i = 0; i < LENGTH; i++) {
                point_3d.val[0] = i;
                for (int j = 0; j < WIDTH; j++) {
                    point_3d.val[1] = j;
                    point_set.push_back(point_3d);
                }
            }
        }
    } else {
        std::cout << "No corner detected" << std::endl;
    }
    return pattern_found;
    
};


// Task2 - Select Calibration Images
// Store current corner & point.
int save_calibration_image(std::vector<cv::Point2f> &corner_set, std::vector<cv::Vec3f> &point_set, std::vector<std::vector<cv::Point2f>> &corner_list, std::vector<std::vector<cv::Vec3f>> &point_list) {
    corner_list.push_back(corner_set);
    point_list.push_back(point_set);
    return 0;
};


bool checkSizes(const std::vector<std::vector<cv::Point2f>> &corner_list,
                const std::vector<std::vector<cv::Vec3f>> &point_list) {
    if (corner_list.size() != point_list.size()) {
        std::cerr << "Error: The number of images and the number of object points sets do not match." << std::endl;
        return false;
    }

    for (size_t i = 0; i < corner_list.size(); ++i) {
        if (corner_list[i].size() != point_list[i].size()) {
            std::cerr << "Error: The number of corners and object points do not match for image " << i << std::endl;
            return false;
        }
    }

    return true;
}

// Task3- Clibrate the camera.
int calibrate_camera(cv::Mat &frame, std::vector<std::vector<cv::Point2f>> &corner_list, std::vector<std::vector<cv::Vec3f>> &point_list, cv::Mat &camera_matrix , cv::Mat &distortion_coefficients) {
    // initilize cameraMatrix
    camera_matrix = cv::Mat::eye(3, 3, CV_64FC1);
    camera_matrix.at<double>(0, 2) = double(frame.cols/2);
    camera_matrix.at<double>(1, 2) = double(frame.rows/2);
    distortion_coefficients = cv::Mat::zeros(1, 5, CV_64FC1);

    // print out the cameraMatrix and distCoeffs to compare after calibrateCamera
    std::cout << "camera_matrix before: " << camera_matrix << std::endl;
    std::cout << "distortion_coefficients before: " << distortion_coefficients << std::endl;  
    bool pass = checkSizes(corner_list, point_list); 


    cv::Mat rvecs, tvecs;
    cv::Mat std_deviations_intrinsic, std_deviations_extrinsic, per_view_errors;
    int CALIB_FIX_ASPECT_RATIO;
    double reprojection_error = cv::calibrateCamera(point_list, corner_list, cv::Size(frame.cols, frame.rows), camera_matrix, distortion_coefficients, rvecs, tvecs,
                                                    std_deviations_intrinsic, std_deviations_extrinsic, per_view_errors, CALIB_FIX_ASPECT_RATIO, 
                                                    cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 30, 0.1));

    std::cout << "camera_matrix after: " << camera_matrix << std::endl;
    std::cout << "distortion_coefficients after: " << distortion_coefficients << std::endl;
    std::cout << "reprojection_error: " << reprojection_error << std::endl;        

    cv::FileStorage fs_camera_mat("camera_matrix.yaml", cv::FileStorage::WRITE);
    fs_camera_mat << "camera_matrix" << camera_matrix;
    fs_camera_mat.release();

    // save distCoeffs
    cv::FileStorage fs_dist("distortion_coefficients.yaml", cv::FileStorage::WRITE);
    fs_dist << "distortion_coefficients" << distortion_coefficients;
    fs_dist.release();

    return 0;
};



// Task456 
//  1. Calculate the current camera (rotation & translation)
//  2. Project 3D Axes
//  3. Create a virtual project
int calculate_and_project(cv::Mat &frame, std::vector<cv::Point2f> &corner_set, std::vector<cv::Vec3f> &point_set) {
    cv::Size size(6, 9);
    // initilize cameraMatrix, distCoeffs, rvecs, tvecs
    cv::Mat camera_matrix, distortion_coefficients;
    cv::Mat rvecs, tvecs;

    // get cameraMatrix, distCoeffs from file
    cv::FileStorage camera_matrix_fs("camera_matrix.yaml", cv::FileStorage::READ);
    camera_matrix_fs["camera_matrix"] >> camera_matrix;
    cv::FileStorage distortion_coefficients_fs("distortion_coefficients.yaml", cv::FileStorage::READ);
    distortion_coefficients_fs["distortion_coefficients"] >> distortion_coefficients;

    // TODO: Optional
    // This is to repair the image corners.
    cv::Mat mask(frame.size(), CV_8UC1, cv::Scalar(255));
    cv::drawChessboardCorners(mask, size, corner_set, true);
    cv::inpaint(frame, mask, frame, 5, cv::INPAINT_TELEA);

    // calculate rotation and translation
    cv::solvePnP(point_set, corner_set, camera_matrix, distortion_coefficients, rvecs, tvecs);

    // print out rotation and translate in real-time
    std::cout << "rotation vector: " << rvecs << std::endl;
    std::cout << "translation vector: " << tvecs << std::endl;


    //////////////////////// below will be the task 5, trying to porject 3d Axes
    std::vector<cv::Point3f> axisPoints;
    axisPoints.emplace_back(0.0f, 0.0f, 0.0f);
    axisPoints.emplace_back(2.0f, 0.0f, 0.0f);
    axisPoints.emplace_back(0.0f, 2.0f, 0.0f);
    axisPoints.emplace_back(0.0f, 0.0f, 2.0f);

    // Define the colors of the axes
    std::vector<cv::Scalar> axesColors;
    axesColors.push_back(cv::Scalar(0, 0, 255)); // red for x-axis
    axesColors.push_back(cv::Scalar(0, 255, 0)); // green for y-axis
    axesColors.push_back(cv::Scalar(255, 0, 0)); // blue for z-axis

    // project 3d points to 2d plane
    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(axisPoints, rvecs, tvecs, camera_matrix, distortion_coefficients, projectedPoints);

    // Draw the projected points on the image
    cv::line(frame, projectedPoints[0], projectedPoints[3], axesColors[2], 10);
    cv::line(frame, projectedPoints[0], projectedPoints[2], axesColors[1], 10);
    cv::line(frame, projectedPoints[0], projectedPoints[1], axesColors[0], 10); 

    //////////////////////// below will create another shape
    std::vector<cv::Point3f> axisPoints_y;
    axisPoints_y.emplace_back(2.0f, 5.0f, 0.0f); // 0
    axisPoints_y.emplace_back(3.0f, 5.0f, 0.0f); // 1
    axisPoints_y.emplace_back(6.0f, 5.0f, 0.0f); // 2
    axisPoints_y.emplace_back(7.0f, 5.0f, 0.0f); // 3
    axisPoints_y.emplace_back(4.0f, 4.0f, 0.0f); // 4
    axisPoints_y.emplace_back(5.0f, 4.0f, 0.0f); // 5
    axisPoints_y.emplace_back(4.0f, 3.0f, 0.0f); // 6
    axisPoints_y.emplace_back(5.0f, 3.0f, 0.0f); // 7
    axisPoints_y.emplace_back(4.0f, 0.0f, 0.0f); // 8
    axisPoints_y.emplace_back(5.0f, 0.0f, 0.0f); // 9

    axisPoints_y.emplace_back(2.0f, 5.0f, 1.0f); // 10
    axisPoints_y.emplace_back(3.0f, 5.0f, 1.0f); // 11
    axisPoints_y.emplace_back(6.0f, 5.0f, 1.0f); // 12
    axisPoints_y.emplace_back(7.0f, 5.0f, 1.0f); // 13
    axisPoints_y.emplace_back(4.0f, 4.0f, 1.0f); // 14
    axisPoints_y.emplace_back(5.0f, 4.0f, 1.0f); // 15
    axisPoints_y.emplace_back(4.0f, 3.0f, 1.0f); // 16
    axisPoints_y.emplace_back(5.0f, 3.0f, 1.0f); // 17
    axisPoints_y.emplace_back(4.0f, 0.0f, 1.0f); // 18
    axisPoints_y.emplace_back(5.0f, 0.0f, 1.0f); // 19

    std::vector<cv::Point2f> projectedPoints_y;
    cv::projectPoints(axisPoints_y, rvecs, tvecs, camera_matrix, distortion_coefficients, projectedPoints_y);
    // Draw the projected points on the image 
    cv::line(frame, projectedPoints_y[0], projectedPoints_y[1], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[1], projectedPoints_y[4], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[4], projectedPoints_y[5], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[5], projectedPoints_y[2], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[2], projectedPoints_y[3], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[3], projectedPoints_y[7], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[7], projectedPoints_y[9], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[9], projectedPoints_y[8], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[8], projectedPoints_y[6], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[6], projectedPoints_y[0], axesColors[1], 5); 

    cv::line(frame, projectedPoints_y[10], projectedPoints_y[11], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[11], projectedPoints_y[14], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[14], projectedPoints_y[15], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[15], projectedPoints_y[12], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[12], projectedPoints_y[13], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[13], projectedPoints_y[17], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[17], projectedPoints_y[19], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[19], projectedPoints_y[18], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[18], projectedPoints_y[16], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[16], projectedPoints_y[10], axesColors[1], 5);
    
    cv::line(frame, projectedPoints_y[0], projectedPoints_y[10], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[1], projectedPoints_y[11], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[2], projectedPoints_y[12], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[3], projectedPoints_y[13], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[4], projectedPoints_y[14], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[5], projectedPoints_y[15], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[6], projectedPoints_y[16], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[7], projectedPoints_y[17], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[8], projectedPoints_y[18], axesColors[1], 5); 
    cv::line(frame, projectedPoints_y[9], projectedPoints_y[19], axesColors[1], 5);

    // Define the vertices of the polygon to be filled
    std::vector<cv::Point> pts_top;
    pts_top.push_back(projectedPoints_y[10]);
    pts_top.push_back(projectedPoints_y[16]);
    pts_top.push_back(projectedPoints_y[18]);
    pts_top.push_back(projectedPoints_y[19]);
    pts_top.push_back(projectedPoints_y[17]);
    pts_top.push_back(projectedPoints_y[13]);
    pts_top.push_back(projectedPoints_y[12]);
    pts_top.push_back(projectedPoints_y[15]);
    pts_top.push_back(projectedPoints_y[14]);
    pts_top.push_back(projectedPoints_y[11]);

    std::vector<cv::Point> pts_left;
    pts_left.push_back(projectedPoints_y[0]);
    pts_left.push_back(projectedPoints_y[6]);
    pts_left.push_back(projectedPoints_y[8]);
    pts_left.push_back(projectedPoints_y[18]);
    pts_left.push_back(projectedPoints_y[16]);
    pts_left.push_back(projectedPoints_y[10]);

    std::vector<cv::Point> pts_bottom;
    pts_bottom.push_back(projectedPoints_y[8]);
    pts_bottom.push_back(projectedPoints_y[9]);
    pts_bottom.push_back(projectedPoints_y[19]);
    pts_bottom.push_back(projectedPoints_y[18]);

    std::vector<cv::Point> pts_right;
    pts_right.push_back(projectedPoints_y[3]);
    pts_right.push_back(projectedPoints_y[7]);
    pts_right.push_back(projectedPoints_y[9]);
    pts_right.push_back(projectedPoints_y[19]);
    pts_right.push_back(projectedPoints_y[17]);
    pts_right.push_back(projectedPoints_y[13]);
    
    std::vector<cv::Point> pts_up;
    pts_up.push_back(projectedPoints_y[0]);
    pts_up.push_back(projectedPoints_y[1]);
    pts_up.push_back(projectedPoints_y[4]);
    pts_up.push_back(projectedPoints_y[5]);
    pts_up.push_back(projectedPoints_y[2]);
    pts_up.push_back(projectedPoints_y[3]);
    pts_up.push_back(projectedPoints_y[13]);
    pts_up.push_back(projectedPoints_y[12]);
    pts_up.push_back(projectedPoints_y[15]);
    pts_up.push_back(projectedPoints_y[14]);
    pts_up.push_back(projectedPoints_y[11]);
    pts_up.push_back(projectedPoints_y[10]);
    
    cv::fillPoly(frame, std::vector<std::vector<cv::Point>>({ pts_top }), cv::Scalar(0, 0, 0));
    cv::fillPoly(frame, std::vector<std::vector<cv::Point>>({ pts_left }), cv::Scalar(0, 0, 0));
    cv::fillPoly(frame, std::vector<std::vector<cv::Point>>({ pts_bottom }), cv::Scalar(0, 0, 0));
    cv::fillPoly(frame, std::vector<std::vector<cv::Point>>({ pts_right }), cv::Scalar(0, 0, 0));
    cv::fillPoly(frame, std::vector<std::vector<cv::Point>>({ pts_up }), cv::Scalar(0, 0, 0));

    return 0;

    
  
};  



// FEATURE
int get_sift_feature(cv::Mat &frame) {
    cv::Mat gray_img;
    cv::cvtColor(frame, gray_img, cv::COLOR_BGR2GRAY);

    // Create SIFT object
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // Detect keypoints
    std::vector<cv::KeyPoint> keypoints;
    sift->detect(gray_img, keypoints);

    // Draw keypoints on image
    cv::Mat imgWithKeypoints;
    drawKeypoints(frame, keypoints, frame, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    return 0;
}

// Below is extension
int aruco(cv::Mat &frame) {
    cv::Mat overlay_img;
    cv::Mat warpedImage;
    overlay_img = cv::imread("cf.png");

    // detect overlay image corners
    std::vector<cv::Point2f> imageCorners;
    imageCorners.push_back(cv::Point2f(0, 0));
    imageCorners.push_back(cv::Point2f(overlay_img.cols, 0));
    imageCorners.push_back(cv::Point2f(overlay_img.cols, overlay_img.rows));
    imageCorners.push_back(cv::Point2f(0, overlay_img.rows));

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);

    detectorParams.markerBorderBits = 2;
    detectorParams.adaptiveThreshWinSizeMax = 75;
    detectorParams.adaptiveThreshWinSizeMin = 45;
    detectorParams.adaptiveThreshConstant = 7;
    detectorParams.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;

    detector.detectMarkers(frame, markerCorners, markerIds);

    if (markerIds.size() > 0) {
        cv::Mat warpMatrix = cv::getPerspectiveTransform(imageCorners, markerCorners[0]);
        cv::warpPerspective(overlay_img, warpedImage, warpMatrix, frame.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        frame = frame + warpedImage;
    }
    
    return 0;
}