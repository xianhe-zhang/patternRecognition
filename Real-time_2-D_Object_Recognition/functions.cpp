#include <opencv2/opencv.hpp>
#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "functions.h"
#include "./utils/csv_util.h"



int threshold(cv::Mat &src, cv::Mat &dest) {
    cv::Mat blur_img;
    cv::Mat gray_img;

    cv::GaussianBlur(src, blur_img, cv::Size(5,5), 0, 0);
    cv::cvtColor(blur_img, gray_img, cv::COLOR_RGB2GRAY);
    // 80 is ususally picked for thresholding a gray img
    cv::threshold(gray_img, dest, 80, 255, 1);

    return 0;
}


int grassfire(cv::Mat &src, cv::Mat &dest, int morph) { 
    typedef cv::Vec<uchar, 1> Vec1b;
    int id = 1;

    dest = cv::Mat::zeros(src.size(), CV_8UC1);

    for (int i = 1; i < src.rows; i++) { 
        Vec1b *dest_uptr = dest.ptr<Vec1b>(i-1);
        Vec1b *dest_cptr = dest.ptr<Vec1b>(i);

        Vec1b *src_ptr = src.ptr<Vec1b>(i);

        for (int j = 1; j < src.cols; j++) {
            int current = src_ptr[j][0];
            int dest_u = dest_uptr[j][0];
            int dest_c = dest_cptr[j-1][0];

            if (current == morph) { 
                dest_cptr[j][0] = 0;
            } else {
                int min;
                min = dest_u < dest_c ? dest_u : dest_c;
                dest_cptr[j][0] = min + 1;
            }
        }
    }

    for (int i = src.rows - 1; i > 0; i--) {
        Vec1b *dest_dptr = dest.ptr<Vec1b>(i+1);
        Vec1b *dest_cptr = dest.ptr<Vec1b>(i);

        Vec1b *src_ptr = src.ptr<Vec1b>(i);

        for (int j = src.cols; j > 0; j--) {
            int current = src_ptr[j][0];
            int dest_d = dest_dptr[j][0];
            int dest_c = dest_cptr[j+1][0];

            if (current == morph) { 
                dest_cptr[j][0] = 0;
            } else {
                int min;
                int dest_cur = dest_cptr[j][0];
                min = dest_d < dest_c ? dest_d : dest_c;
                min = min + 1;
                dest_cptr[j][0] = dest_cur < min ? dest_cur : min;
            }
        }
    }

    return 0;
}


int clean(cv::Mat &src, cv::Mat &dest, char* morph) {
    typedef cv::Vec<uchar, 1> Vec1b;
    dest = cv::Mat::zeros(src.size(), CV_8UC1);

    for (int i = 0; i < src.rows; i++) {
        Vec1b *src_ptr = src.ptr<Vec1b>(i);
        Vec1b *dest_ptr = dest.ptr<Vec1b>(i);

        for (int j = 0; j < src.cols; j++) {
            int cur_p = src_ptr[j][0];

            if (strcmp(morph, "dilation") == 0) {
                if (cur_p < 15) {
                    dest_ptr[j][0] = 225;
                } else {
                    dest_ptr[j][0] = 0;
                }
            } else if (strcmp(morph, "erosion") == 0) {
                if (cur_p < 15) {
                    dest_ptr[j][0] = 0;
                } else {
                    dest_ptr[j][0] = 255;
                }
            }
        }
    }
    return 0;

}

// To calculate feature scores and store them into a csv file
int getFeatureScores(cv::Mat& img, cv::Mat& bin_img, cv::Mat& labels, std::vector<double> &features, int component) {
    cv::Mat label_flag = (labels == component);
    cv::Moments moments = cv::moments(label_flag, true);

    double x = moments.m10/ moments.m00;
    double y = moments.m01/ moments.m00;
    double t = 0.5 * atan2(2 * moments.mu11, moments.mu20 - moments.mu02);

    cv::Point2f center(x,y);
    cv::Point2f p1(center.x + std::cos(t)*100, center.y + std::sin(t)*100);
    cv::Point2f p2(center.x - std::cos(t)*100, center.y - std::sin(t)*100);

    double hum[10];
    cv::HuMoments(moments, hum);

    std::vector<std::vector<cv::Point>> c;
    cv::findContours(label_flag, c, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::RotatedRect box;
    for (int i = 0; i < c.size(); i++) {
        box = cv::minAreaRect(c[i]);
        cv::Point2f vertices[4];
        box.points(vertices);

        std::vector<cv::Point> box_p;
        for (int j = 0; j < 4; j++) {
            box_p.push_back(vertices[j]);
        }

        std::vector<std::vector<cv::Point>> box_c;
        box_c.push_back(box_p);

        cv::drawContours(img, box_c, 0, cv::Scalar(0, 0, 255), 2);
    }

    // Caculation
    double b_a = box.size.width * box.size.height;
    double f_a = cv::countNonZero(label_flag);
    double fp = f_a / b_a;
    double w_h =  box.size.width / box.size.height;

    cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 3);
    
    hum[7] = t;
    hum[8] = fp;
    hum[9] = w_h;

    int size = sizeof(hum)/sizeof(double);
    // hum is an array, vector is dynamic 
    // array is more suitable for func var. after return, func var mem will be discard.
    std::vector<double> temp(hum, hum+size);
    features = temp;
    return 0;
}


// To get standard deviation.
int getStd() {
    std::vector<char *> filenames; // File de Name
    std::vector<std::vector<double>> data; // 
    std::vector<double> mean_vec(10, 0.0);
    std::vector<double> std_vec(10, 0.0);
    char standard_dev[] = "standard_dev"; 
    char csv_std[] = "std.csv";
    std::string folder_path = "database";

    int num_data = 0;
    // For-loop to tranverse every file in 
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {    
        std::string file_path = entry.path().string();
        char *f_path = &file_path[0];
        read_image_data_csv(f_path, filenames, data, 0);
        // loop through each row in a data csv file
        for (int i = 0; i < data.size(); i++) {
            // loop through each feature in a single row of data.
            for (int j = 0; j < data[i].size(); j++) {
                double mean = (num_data * mean_vec[j] + data[i][j]) / (num_data + 1);
                double diff = data[i][j] - mean;
                double st_d = 0.0;
                if (num_data != 0) {
                    st_d = ((num_data / (num_data + 1)) * std_vec[j]) + ((diff * diff) / num_data);
                } 
                mean_vec[j] = mean;
                std_vec[j] = st_d;
            }
            num_data += 1;
        }

    }
    // save the calculated standard deviation
    append_image_data_csv(csv_std, standard_dev, std_vec, 1);
    return 0;

}
// To get scaled euclidean distance.
int getScaledEuclidean(std::vector<double>& src_vector, std::vector<std::pair<char*, double>>& result) {
    // initialize some variables
    std::vector<char *> filenames;
    std::vector<char *> std_filenames;    
    char std_csv[] = "std.csv";
    std::vector<std::vector<double>> data;
    std::vector<std::vector<double>> std_data;
    read_image_data_csv(std_csv, std_filenames,std_data , 0);

    std::string folder_path = "database";
    // loop through the database -> calculate euclidean_distance for each file -> store the score -> decide the identification. 
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {    
        std::string file_path = entry.path().string();
        char *f_path = &file_path[0];
        std::cout << "Getting " << file_path << std::endl;
        // read the data from teh csv files 
        read_image_data_csv(f_path, filenames, data, 0);
        for (int i = 0; i < data.size(); i++) {
            double euclidean_distance = 0.0;
            // loop through each row in a data csv file
            for (int j = 0; j < data[i].size(); j++) {
                double scaled_diff = (src_vector[j] - data[i][j]) / std_data[0][j];
                euclidean_distance = scaled_diff * scaled_diff;
            }
            // save the calculated euclidean distance to a vector and sort them to get the shortest distance
            result.push_back(std::make_pair(filenames[i], euclidean_distance));
        }
    }
    return 0;
}
// To get K-Nearest neighbors.
int getKNs(std::vector<double>& src_vector, std::vector<std::pair<char*, double>>& result) {
    // initialize some variables
    char std_csv[] = "std.csv";
    std::vector<char *> std_filenames;    
    std::vector<std::vector<double>> std_data;
    // read data from the data csv files
    read_image_data_csv(std_csv, std_filenames, std_data, 0);
    std::vector<char *> filenames;
    std::vector<std::vector<double>> data;
    std::string folder_path = "database";
    // loop through the csv files in the database
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {        
        // initialize some variables
        std::string file_path = entry.path().string();
        char *f_path = &file_path[0];
        std::vector<double> euclidean_vec;
        read_image_data_csv(f_path, filenames, data, 0);
        double k_nearest_distance = 0.0;
        // loop through every row of data
        for (int i = 0; i < data.size(); i++) {
            double euclidean_distance = 0.0;
            // loop through every feature in one row of data and calculate the scaler euclidean difference
            for (int j = 0; j < data[i].size(); j++) {
                double scaled_diff = (src_vector[j] - data[i][j]) / std_data[0][j];
                euclidean_distance = scaled_diff * scaled_diff;
            }
            // save the calculated euclidean distance for later use
            euclidean_vec.push_back(euclidean_distance);
        }

        std::sort(euclidean_vec.begin(), euclidean_vec.end());

        for (int i = 0; i < 3; i++) {
            k_nearest_distance += euclidean_vec[i];
        }
        result.push_back(std::make_pair(filenames[2], k_nearest_distance));

        // clear out the vectors for the next loop
        euclidean_vec.clear();
        euclidean_vec.resize(0);
        data.clear();
        data.resize(0);
        filenames.clear();
        filenames.resize(0);
    }
    return 0;
}
// To get the confusion matrix to evaluate model performance.
int getConfusionMatrix() {
    // k - nearest neighbor 
    std::vector<int> actual = { 0, 0, 0, 2, 1, 1, 2,1, 2};
    std::vector<int> predicted = { 1, 0, 0, 1, 2, 1, 2, 1, 2};
    int num_classes = 3;

    // initialize the confusion matrix
    cv::Mat confusionMatrix = cv::Mat::zeros(num_classes, num_classes, CV_32S);

    // iterate over the test data and update the confusion matrix
    for (int i = 0; i < actual.size(); i++) {
        confusionMatrix.at<int>(actual[i], predicted[i])++;
    }

    // print the confusion matrix to a CSV file
    std::ofstream outfile("confusion_matrix.csv");
    if (outfile.is_open()) {
        // print the header row
        outfile << "Actual Result\\Predicted Result,";
        for (int i = 0; i < num_classes; i++) {
            outfile << i << ",";
        }
        outfile << std::endl;

        // print the matrix rows
        for (int i = 0; i < num_classes; i++) {
            outfile << i << ",";
            for (int j = 0; j < num_classes; j++) {
                outfile << confusionMatrix.at<int>(i, j) << ",";
            }
            outfile << std::endl;
        }

        outfile.close();
    } else {
        std::cout << "Error: could not open file for writing" << std::endl;
    }

    return 0;
}
// To conduct connected component analysis.
int conductCCA(cv::Mat &bin_img, cv::Mat &output, cv::Mat &labels, int num_of_components, int size) {
    cv::Mat stats, centroids;
    int get_components = cv::connectedComponentsWithStats(bin_img, labels, stats, centroids);
    num_of_components = get_components;

    std::vector<cv::Vec3b> colors(get_components);
    // This is the background color.
    colors[0] = cv::Vec3b(0, 0, 0);
    for (int i = 1; i < get_components; i++) {
        colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
    }
    
    output.create(bin_img.size(), CV_8UC3);
    for (int i = 0; i < bin_img.rows; i++) {
        for (int j = 0; j < bin_img.cols; j++) {
            int label = labels.at<int>(i, j);
            if (label != 0 && stats.at<int>(label, cv::CC_STAT_AREA) >= size) {
                output.at<cv::Vec3b>(i, j) = colors[label];
            } else {
                output.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
            }
        }
    }

    return 0;
}
