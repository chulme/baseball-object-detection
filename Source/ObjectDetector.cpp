#include <ObjectDetector.h>
#include <DetectedObject.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <fstream>

const int NO_OBJECT_DETECTED = -1;
const cv::Point NO_OBJECT_COORD(-1, -1);

ObjectDetector::ObjectDetector(const std::string &object_to_find) : object_label(object_to_find), confidence_threshold(0.1F), nms_threshold(0.4F)
{
    load_yolo_model();
    load_class_names();
}

ObjectDetector::ObjectDetector(const std::string &object_to_find, const float confidence_threshold, const float nms_threshold)
    : object_label(object_to_find), confidence_threshold(confidence_threshold), nms_threshold(nms_threshold)
{
    load_yolo_model();
    load_class_names();
}

void ObjectDetector::load_yolo_model()
{
    const std::string config_path = "Data/Yolo/yolov4.cfg";
    const std::string weights_path = "Data/Yolo/yolov4.weights";

    net = cv::dnn::readNetFromDarknet(config_path, weights_path);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

void ObjectDetector::load_class_names()
{
    std::string classesFile = "Data/Yolo/coco.names";
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line))
    {
        classes.push_back(line);
    }
}

DetectedObject ObjectDetector::detect_object(const std::string &input_path)
{
    cv::Mat image = cv::imread(input_path);
    cv::Mat blob;

    const int input_width = 608, input_height = 608;

    cv::dnn::blobFromImage(image, blob, 1 / 255.0, cv::Size(input_width, input_height), cv::Scalar(0, 0, 0), true, false, CV_32F);

    std::vector<cv::Mat> output = process(blob);
    cv::Point coords = find_and_display_object(image, output);

    return {image, coords};
}

std::vector<cv::Mat> ObjectDetector::process(const cv::Mat &blob)
{
    net.setInput(blob);
    std::vector<cv::Mat> output;
    net.forward(output, get_output_names());
    return output;
}

std::vector<std::string> ObjectDetector::get_output_names()
{
    static std::vector<std::string> names;
    if (names.empty())
    {
        std::vector<int> output_layers = net.getUnconnectedOutLayers();

        std::vector<std::string> layer_names = net.getLayerNames();

        names.resize(output_layers.size());

        for (int i = 0; i < output_layers.size(); i++)
        {
            names.at(i) = layer_names.at(output_layers.at(i) - 1);
        }
    }
    return names;
}

cv::Point ObjectDetector::find_and_display_object(cv::Mat &image, const std::vector<cv::Mat> &net_output)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    prune_boxes(image, net_output, classIds, confidences, boxes);

    // Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold, indices);

    int best_box_index = select_most_confident_box(indices, confidences);

    if (best_box_index == NO_OBJECT_DETECTED)
    {
        return NO_OBJECT_COORD;
    }
    else
    {
        //Mark object with bounding box and center marker
        cv::Rect best_box = boxes.at(best_box_index);
        draw_bounding_box(classIds.at(best_box_index), confidences.at(best_box_index), best_box.x, best_box.y,
                          best_box.x + best_box.width, best_box.y + best_box.height, image);
        cv::Point center = get_box_center_point(best_box);
        cv::drawMarker(image, center, cv::Scalar(0, 4, 255), 0, 20, 2, 8);
        return center;
    }
}

void ObjectDetector::prune_boxes(cv::Mat &image, const std::vector<cv::Mat> &net_output,
                                 std::vector<int> &class_ids, std::vector<float> &confidences,
                                 std::vector<cv::Rect> &boxes)
{
    for (int i = 0; i < net_output.size(); i++)
    {
        float *data = (float *)net_output.at(i).data;
        for (int j = 0; j < net_output.at(i).rows; j++, data += net_output.at(i).cols)
        {
            cv::Mat scores = net_output.at(i).row(j).colRange(5, net_output.at(i).cols);
            cv::Point class_id_point;
            double confidence;

            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &class_id_point);

            //Only consider boxes that meet confidence threshold and are of the target object class
            if ((confidence > confidence_threshold) && is_object_target_class(class_id_point.x))
            {
                int centerX = (int)(data[0] * image.cols);
                int centerY = (int)(data[1] * image.rows);
                int width = (int)(data[2] * image.cols);
                int height = (int)(data[3] * image.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                class_ids.push_back(class_id_point.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
}

bool ObjectDetector::is_object_target_class(const int class_id)
{

    return classes.at(class_id).compare(object_label) == 0 ? true : false;
}

int ObjectDetector::select_most_confident_box(const std::vector<int> &indices, const std::vector<float> &confidences)
{
    int best_index = -1;
    float max_confidence = 0;
    for (int index : indices)
    {
        float confidence = confidences.at(index);
        if (confidence > max_confidence)
        {
            best_index = index;
            max_confidence = confidence;
        }
    }
    return best_index;
}

void ObjectDetector::draw_bounding_box(const int classId, const float conf, const int left, const int top,
                                       const int right, const int bottom, cv::Mat &image)
{
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes.at(classId) + ":" + label;
    }
    cv::Scalar colour = cv::Scalar(0, 4, 255);

    //Draw a rectangle displaying the bounding box
    cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), colour, 3);

    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int box_top = cv::max(top, labelSize.height);

    rectangle(image, cv::Point(left, box_top - round(1.5 * labelSize.height)),
              cv::Point(left + round(1.5 * labelSize.width), box_top + baseLine), colour, cv::FILLED);

    putText(image, label, cv::Point(left, box_top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
}

cv::Point ObjectDetector::get_box_center_point(const cv::Rect &box)
{
    return cv::Point(((2 * box.x + box.width - 1) / 2), ((2 * box.y + box.height - 1) / 2));
}
