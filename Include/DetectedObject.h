#pragma once
#include <opencv2/core/core.hpp>
#include <vector>
#include <string>

//Class that holds an image with bounding box applied over the detected object, and the center coordinates of the box.
class DetectedObject
{
public:
    DetectedObject(cv::Mat image, cv::Point coords) : image(image), coords(coords) {};
    cv::Mat image;
    cv::Point coords;

    //Save image to specified path and write coordinates to Output/coords.txt. NB: if coordinates are (-1,-1) then the object was not found
    void save(const std::string & out_path);
};

