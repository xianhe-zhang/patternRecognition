#include <opencv2/opencv.hpp>
#include "../utils/csv_util.h" 
#include "../utils/utils.h"

// write feature vectors based on solely the center 9 x 9 pixles of images to baseline.csv file
int build_baseline (cv::Mat &src, char *image_filename) {
    char csvFile[] = "../csv_files/baseline.csv";
    std::vector<float> vect;
    int nrow = src.rows/2 - 4;
    int ncol = src.cols/2 - 4;
    for (int i = nrow; i < nrow + 9; i++) {
        cv::Vec3b *rowptr = src.ptr<cv::Vec3b>(i);
        for (int j = ncol; j < ncol + 9; j++) {
            for (int k = 0; k < 3; k++) {
                vect.push_back(rowptr[j][k]);
            }
        }
    }
    append_image_data_csv(csvFile, image_filename, vect, 0);
    return 0;
}

// get 3D hist -> convert into 1d -> save CSV
int build_histogram(cv::Mat &image, char *image_filename) {
    cv::Mat histogram;
    std::vector<float> flat_hist;
    char csvFile[] = "../csv_files/histogram.csv";

    get_RGB_Hist(image, histogram);
    convert_hist_to_1d(histogram, flat_hist);
    append_image_data_csv(csvFile, image_filename, flat_hist, 0);

    return 0;
}

// cut a image into 2 parts -> get Histogram for each -> save csv
int build_multi_histogram(cv::Mat &image, char *image_filename) {
    cv::Mat top_half;
    cv::Mat down_half;
    cv::Mat top_histogram;
    cv::Mat down_histogram;

    std::vector<float> top_flat_hist;
    std::vector<float> down_flat_hist;

    char top_csvFile[] = "../csv_files/top_histogram.csv";
    char down_csvFile[] = "../csv_files/down_histogram.csv";

    half_image(image, top_half, down_half);

    get_RGB_Hist(top_half, top_histogram);
    get_RGB_Hist(down_half, down_histogram);

    convert_hist_to_1d(top_histogram, top_flat_hist);
    convert_hist_to_1d(down_histogram, down_flat_hist);

    append_image_data_csv(top_csvFile, image_filename, top_flat_hist, 0);
    append_image_data_csv(down_csvFile, image_filename, down_flat_hist, 0);
    return 0;
}

// write color,sobel into CSV -> convert into 1D -> save csv
int build_texture_and_color(cv::Mat &image, char *image_filename) {

    cv::Mat magnitude;
    cv::Mat color_histogram;

    std::vector<float> color_flat_hist;
    std::vector<float> magnitude_vect;

    char color_csvFile[] = "../csv_files/color_histogram.csv";
    char magnitude_csvFile[] = "../csv_files/magnitude_histogram.csv";

    get_RGB_Hist(image, color_histogram);
    get_magnitude_hist(image, magnitude_vect);
    convert_hist_to_1d(color_histogram, color_flat_hist);

    append_image_data_csv(color_csvFile, image_filename, color_flat_hist, 0);
    append_image_data_csv(magnitude_csvFile, image_filename, magnitude_vect, 0);

    return 0;
}

// use hsv values to compare images and to match similar images
int build_hsv(cv::Mat &image, char *image_filename) {
    cv::Mat hsv_hist;
    std::vector<float> hsv_vect;
    char hsv_csvFile[] = "../csv_files/hsv.csv";
    get_hsv(image, hsv_hist);
    convert_hist_to_1d(hsv_hist, hsv_vect);
    append_image_data_csv(hsv_csvFile, image_filename, hsv_vect, 0);

    return 0;
}