
#include <opencv2/opencv.hpp>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <algorithm>
#include <vector>
#include <map>

#include "../utils/csv_util.h"
#include "match_helper.h"


bool sorter(const std::pair<char*, float> &a,
                 const std::pair<char*, float> &b) {
    return a.second < b.second;
}


int main(int argc, char *argv[]) {

    if (argc != 6)
    {
        printf("usage: %s<match type> <directiory name> <image name> <number of output images>\n", argv[0]);
        exit(-1);
    }

    char *image_name = argv[1];
    char *match_type = argv[2];
    char *dir_name = argv[3];
    char *metric = argv[4];
    int ntop = std::atoi(argv[5]); // the number of output images
    
    // To form a path
    char imagepath[256] = "../";
    std::strcat(imagepath, dir_name);
    std::strcat(imagepath, "/");
    std::strcat(imagepath, image_name);

    cv::Mat image;
    std::vector<std::pair<char *, float>> imagevec;

    // read input image
    image = cv::imread(imagepath, cv::IMREAD_ANYCOLOR);

    // check if image file can be opened
    if (image.empty()) {
        std::cout << "Could not open image" << std::endl;
        return -1;
    }

    if (std::strcmp(match_type, "baseline") == 0) {
        // different matching algo
        // baseline match
        baseline_match(image, imagevec, metric);
    } else if (std::strcmp(match_type, "histogram") == 0) {
        // histogram match
        histogram_match(image, imagevec, metric);
    } else if (std::strcmp(match_type, "multihistogram") == 0) {
        // multi histogram match method
        multi_histogram_match(image, imagevec, metric);
    } else if (std::strcmp(match_type, "texture") == 0) {
        // color and texture match
        color_and_texture_match(image, imagevec, metric);
    } else if (std::strcmp(match_type, "hsv") == 0) {
        // hsv match
        hsv_match(image, imagevec, metric);
    }

    // sort the distance between target image and images in database
    sort(imagevec.begin(), imagevec.end(), sorter);

    int i = 0;

    // get the sort image name in a separate vector
    std::vector<char *> sorted_imagename;
    for (auto &p : imagevec) {
        sorted_imagename.push_back(p.first);
    }
    
    // show target image
    char target_dir[256] = "../";
    char target_dir_name[256];
    std::strncpy(target_dir_name, dir_name, 256);
    char target_image_name[256];
    std::strncpy(target_image_name, image_name, 256);
    std::strcat(target_dir, target_dir_name);
    std::strcat(target_dir, "/");
    std::strcat(target_dir, target_image_name);
    cv::Mat target = cv::imread(target_dir);
    cv::namedWindow("target image", cv::WINDOW_NORMAL);
    cv::imshow("target image", target);

    // display the top N images
    for (int i = 1; i < ntop + 1; i++) {
        char top_dir[256] = "../";
        char top_dir_name[256];
        std::strncpy(top_dir_name, dir_name, 256);
        char top_image_name[256];
        std::strncpy(top_image_name, image_name, 256);
        std::strcat(top_dir, top_dir_name);
        std::strcat(top_dir, "/");
        std::strcat(top_dir, sorted_imagename[i]);
        cv::Mat top = cv::imread(top_dir);
        char window_name[256] = "top ";
        std::string s = std::to_string(i);
        const char *n = s.c_str();
        std::strcat (window_name, n);
        std::strcat (window_name, " ");        
        std::strcat (window_name, sorted_imagename[i]);        
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::imshow(window_name, top);
    }

    // quit on q/Q
    while (true) {
        int k = cv::waitKey(0); 
        if (k == 'q') {
            break;
        } else if (k == 'Q') {
            break;
        }
    }

    return 0;
}
