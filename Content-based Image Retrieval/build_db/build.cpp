#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <typeinfo>

#include <opencv2/opencv.hpp>

#include "build_helper.h"


int main(int argc, char *argv[])
{
    char dirname[256];
    char buffer[256];
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;

    // intilize image mat
    cv::Mat image;

    // check for sufficient arguments
    if (argc < 2)
    {
        printf("usage: %s <directory path>\n", argv[0]);
        exit(-1);
    }

    // get the directory path
    strcpy(dirname, argv[1]);
    printf("Processing directory %s\n", dirname);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL)
    {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL)
    {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif"))
        {

            // build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            // read the images from a direcitonay and save their feature vectors into a csvfile
            // read image for baseline matching
            image = cv::imread(buffer, cv::IMREAD_ANYCOLOR);

            // check if image file can be opened
            if (image.empty()) {
                std::cout << "Could not open image" << std::endl;
                return -1;
            }

            // call function to constucte csv database
            build_baseline(image, dp->d_name);
            build_histogram(image, dp->d_name);
            build_multi_histogram(image, dp->d_name); 
            build_texture_and_color(image, dp->d_name);
            build_hsv(image, dp->d_name);
        }
    }
    printf("Terminating\n");
    return (0);
}