// Main.cpp : This file contains a sample main method to demonstrate how to find coordinates and log the output
//            by searching for baseball bats within the provided images.

#include <ObjectDetector.h>
#include <Utilities.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <iostream>

void show_results(const DetectedObject & result) {
    std::cout << "Found object at " << result.coords.x << ", " << result.coords.y;
    
    cv::imshow("Result", result.image);
    cv::waitKey(0);
}

int main()
{
    std::string folder_path = "Data/InputImages/*.png";
    std::vector<std::string> input_image_paths;
    cv::glob(folder_path, input_image_paths);

    ObjectDetector baseball_detector("baseball bat"); //specify the target object to search for.
                                                      //NB: can tune confidence factor and nms threshold respectively with following constructor:
                                                      //    ObjectDetector baseball_detector("baseball bat", 0.4F, 0.2F); 
    //Loop over all provided images
    for (std::string input_image_path : input_image_paths)
    {
        DetectedObject result = baseball_detector.detect_object(input_image_path); //Detects object in provided image
                                                                                   //NB: using the returned object the edited image and coordinates can be accessed
                                                                                   //    as seen in the show_results(...) method

        result.save("Output/Images/" + get_filename(input_image_path)); //save image to a desired path and writes to internal logging file

        show_results(result);

    }
    return 0;
}
