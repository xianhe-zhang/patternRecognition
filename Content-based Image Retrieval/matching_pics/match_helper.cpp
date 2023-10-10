#include <vector>
#include <opencv2/opencv.hpp>

#include "../utils/csv_util.h"
#include "../utils/utils.h"

int baseline_match(cv::Mat &image, std::vector<std::pair<char *, float>> &imagevec, char *metric) {
    std::vector<char *> filenames;
    std::vector<std::vector<float>> data;
    char csvFile[] = "../csv_files/baseline.csv";
    std::vector<float> target_vect;
    // loop through each pixel in the image ang only the center 9 pixels
    int nrow = image.rows / 2 - 4;
    int ncol = image.cols / 2 - 4;

    for (int i = nrow; i < nrow + 9; i++)
    {
        cv::Vec3b *rowptr = image.ptr<cv::Vec3b>(i);
        for (int j = ncol; j < ncol + 9; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                target_vect.push_back(rowptr[j][k]);
            }
        }
    }

    read_image_data_csv(csvFile, filenames, data, 0);
    // For each data in our db, we will do the math, simply store all scores in the vec.
    for (int i = 0; i < data.size(); i++) {
        float distance = 0;
        
        // to get SSD or intersection, will record these metrics. The only thing left to do is to sort & pick in the main()
        if (std::strcmp(metric, "squared_difference") == 0) {
            distance = get_ssd(target_vect, data[i]);  
        } else if (std::strcmp(metric, "intersection") == 0) {
            distance = intersection(target_vect, data[i]);            
        }

        imagevec.push_back(std::make_pair(filenames[i], distance));
    }
    return 0;
}

// histogram match function -  calculate the similirity of the images histogram.csv with the given target image using intersection metric
int histogram_match(cv::Mat &image, std::vector<std::pair<char *, float>> &imagevec, char *metric) {
    // initilize target hist
    cv::Mat target_hist;

    // initilize target vector
    std::vector<float> target_flat_hist;

    // calculate target histogram
    get_RGB_Hist(image, target_hist);

    // flatten out the target histogram
    convert_hist_to_1d(target_hist, target_flat_hist);

    std::vector<char *> filenames;
    std::vector<std::vector<float>> data;
    char csv_file[] = "../csv_files/histogram.csv";

    // loop through each entry in the csv file and calculate distance metric against the target value
    read_image_data_csv(csv_file, filenames, data, 0);

    for (int i = 0; i < data.size(); i++) {
        float distance;

        // calculate the square difference or intersection
        if (std::strcmp(metric, "squared_difference") == 0) {
            distance = get_ssd(target_flat_hist, data[i]);
        } else if (std::strcmp(metric, "intersection") == 0) {
            distance = intersection(target_flat_hist, data[i]);            
        }
        // save distance data into a vector for later sorting
        imagevec.push_back(std::make_pair(filenames[i], distance));
    }

    return 0;

}

// multi histogram match function - calculate the similirity of the images (combining the data in the top_histogram.csv 
// and down_histogram.csv with the given target image using intersection metric
int multi_histogram_match(cv::Mat &image, std::vector<std::pair<char *, float>> &imagevec, char *metric) {

    cv::Mat top_image;
    cv::Mat down_image;
    cv::Mat top_target_hist;
    cv::Mat down_target_hist;

    std::vector<float> top_target_flat_hist;
    std::vector<float> down_target_flat_hist;

    std::vector<char *> top_filenames;
    std::vector<char *> down_filenames;

    std::vector<std::vector<float>> top_data;
    std::vector<std::vector<float>> down_data;

    char top_csv_file[] = "../csv_files/top_histogram.csv";
    char down_csv_file[] = "../csv_files/down_histogram.csv";

    // split the target image
    half_image(image, top_image, down_image);

    // calculate target histogram top and down
    get_RGB_Hist(top_image, top_target_hist);
    get_RGB_Hist(down_image, down_target_hist);

    // flatten out the target histogram top and down
    convert_hist_to_1d(top_target_hist, top_target_flat_hist);
    convert_hist_to_1d(down_target_hist, down_target_flat_hist);

    // loop through each entry in the csv file and calculate distance metric against the target value
    read_image_data_csv(top_csv_file, top_filenames, top_data, 0);
    read_image_data_csv(down_csv_file, down_filenames, down_data, 0);

    for (int i = 0; i < top_data.size(); i++) {
        // make sure the the file name in the left is the same as on the right 
        int result = std::strcmp(top_filenames[i], down_filenames[i]);
        // checking if the top and down images and of the same image
        if (result != 0) {
            std::cout << "Something went wrong, please delete the left_histogram.csv and right_histogram,csv and re-build them." << std::endl;
            return (-1);
        }

        // initilize variabels
        float top_distance;
        float down_distance;
        float distance;

        // calculate the square difference or intersection
        if (std::strcmp(metric, "squared_difference") == 0) {
            top_distance = get_ssd(top_target_flat_hist, top_data[i]);
            down_distance = get_ssd(down_target_flat_hist, down_data[i]);
        } else if (std::strcmp(metric, "intersection") == 0) {
            top_distance = intersection(top_target_flat_hist, top_data[i]);
            down_distance = intersection(down_target_flat_hist, down_data[i]);            
        }

        distance = (top_distance * 0.5) + (down_distance * 0.5);
        imagevec.push_back(std::make_pair(top_filenames[i], distance));
    }
    return 0;

}

// multi histogram match function - calculate the similirity of the images (combining the data in the color_histogram.csv 
// and sobel_histogram.csv with the given target image using intersection metric
int color_and_texture_match(cv::Mat &image, std::vector<std::pair<char *, float>> &imagevec, char *metric) {

    // initilize the sobel of the target image
    cv::Mat magnitude;

    // initilize target histogram color and sobel
    cv::Mat color_target_hist;

    // initilize target vector color and sobel
    std::vector<float> color_target_flat_hist;
    std::vector<float> magnitude_target_vect;

    // initilize filenames vector to store the images names
    std::vector<char *> color_filenames;
    std::vector<char *> magnitude_filenames;

    // initilize data vector to store the 1d vector after histogram 
    std::vector<std::vector<float>> color_data;
    std::vector<std::vector<float>> magnitude_data;

    // color and sobel 2 csv files
    char color_csv_file[] = "../csv_files/color_histogram.csv";
    char magnitude_csv_file[] = "../csv_files/magnitude_histogram.csv";

    // calculate target histogram color and sobel
    get_RGB_Hist(image, color_target_hist);
    get_magnitude_hist(image, magnitude_target_vect);
    // flatten out the target histogram color and sobel
    convert_hist_to_1d(color_target_hist, color_target_flat_hist);

    // loop through each entry in the csv file and calculate distance metric against the target value
    read_image_data_csv(color_csv_file, color_filenames, color_data, 0);
    read_image_data_csv(magnitude_csv_file, magnitude_filenames, magnitude_data, 0);

    for (int i = 0; i < color_data.size(); i++) {
        // make sure the the file name in the left is the same as on the right 
        int result = std::strcmp(color_filenames[i], magnitude_filenames[i]);
        if (result != 0) {
            std::cout << "Something went wrong, please delete the left_histogram.csv and right_histogram,csv and re-build them." << std::endl;
            return (-1);
        }
        float color_distance;
        float magnitude_distance;
        float distance;

        // calculate the square difference or intersection
        if (std::strcmp(metric, "squared_difference") == 0) {
            color_distance = get_ssd(color_target_flat_hist, color_data[i]);
            magnitude_distance = get_ssd(magnitude_target_vect, magnitude_data[i]);
        } else if (std::strcmp(metric, "intersection") == 0) {
            color_distance = intersection(color_target_flat_hist, color_data[i]);
            magnitude_distance = intersection(magnitude_target_vect, magnitude_data[i]);         
        }

        // calculate the distance - combining the 2 calculated distances with assigned weight to make the best result
        distance = (0.1 * color_distance) + (0.9 * magnitude_distance);
        imagevec.push_back(std::make_pair(color_filenames[i], distance));
    }
    return 0;
}

// use hsv histogram to compare images to match the images that are similar to a human eye
int hsv_match(cv::Mat &image, std::vector<std::pair<char *, float>> &imagevec, char *metric) {

    cv::Mat target_hsv_hist;
    cv::Mat target_rgb_hist;
    std::vector<float> target_hsv_vect;
    std::vector<float> target_rgb_vect;

    // initilize filenames vector to store the images names
    std::vector<char *> color_filenames;
    std::vector<char *> hsv_filenames;

    // initilize data vector to store the 1d vector after histogram 
    std::vector<std::vector<float>> color_data;
    std::vector<std::vector<float>> hsv_data;

    // initilize csv files
    char color_csv_file[] = "../csv_files/histogram.csv";
    char hsv_csv_file[] = "../csv_files/hsv.csv";

    // calculate target hsv
    get_RGB_Hist(image, target_rgb_hist);
    get_hsv(image, target_hsv_hist);

    // histogram conversion
    convert_hist_to_1d(target_rgb_hist, target_rgb_vect);
    convert_hist_to_1d(target_hsv_hist, target_hsv_vect);

    // loop through each entry in the csv file and calculate distance metric against the target value
    read_image_data_csv(color_csv_file, color_filenames, color_data, 0);
    read_image_data_csv(hsv_csv_file, hsv_filenames, hsv_data, 0);

    for (int i = 0; i < color_data.size(); i++) {
        float distance;

        distance = 0.8 * (intersection(target_rgb_vect, color_data[i])) + 0.2 * (intersection(target_hsv_vect, hsv_data[i]));
        // call intersection and calculate the intersection between target and each entry in the csv file
        imagevec.push_back(std::make_pair(color_filenames[i], distance));
    }

    return 0;
}
