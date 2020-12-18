#pragma once
#include <DetectedObject.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>

//Class to detect a specified object from an image using yolov4 object detection
class ObjectDetector
{
public:
    ObjectDetector(const std::string &object_to_find);
    ObjectDetector(const std::string &object_to_find, const float confidence_threshold, const float nms_threshold);

    //Finds the target object from an image loaded from input_path, returning an image with the object marked, and object coordinates.
    //NB: if coordinates are(-1, -1) then the object was not found
    DetectedObject detect_object(const std::string &input_path);

private:
    cv::dnn::Net net;
    std::vector<std::string> classes;

    const std::string object_label;
    const float confidence_threshold;
    const float nms_threshold;

    void load_yolo_model();
    void load_class_names();
    
    std::vector<cv::Mat> process(const cv::Mat &blob);
    cv::Point find_and_display_object(cv::Mat &image, const std::vector<cv::Mat> &net_output);
    //Prunes boxes of low confidence and those belonging to different object classes
    void prune_boxes(cv::Mat &image, const std::vector<cv::Mat> &output, std::vector<int> &classIds,
                     std::vector<float> &confidences, std::vector<cv::Rect> &boxes);
    std::vector<std::string> get_output_names();
    void draw_bounding_box(const int classId, const float conf, const int left, const int top,
                           const int right, const int bottom, cv::Mat &image);
    //Returns the index of the box with the highest confidence
    int select_most_confident_box(const std::vector<int> &indices,
                                  const std::vector<float> &confidences);
    bool is_object_target_class(const int class_id);
    cv::Point get_box_center_point(const cv::Rect &boxes);
};
