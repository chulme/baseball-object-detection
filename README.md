# Yolo Object Detection

This program makes use of YOLOv4 and OpenCV to recognise baseball bats and find the respective coordinates from the provided images.
![](https://i.imgur.com/TR4PTKU.png)

### ObjectDetector Class

This is the primary class that holds key functionality of the program. It can find the baseball bat coordinates and marks the location on the image with a bounding box.

To improve the reusability of this class, it can be configured to search for different objects and different detection configurations, depending on how it is constructed, such as:
```cpp
ObjectDetector bat_detector("baseball bat"); //will search for a baseball bat
ObjectDetector cat_detector("cat", 0.4F, 0.2F); //will search for a cat with a confidence threshold and a NMS threshold of 0.4F and 0.2F respectively
```
Once constructed, the ObjectDetector instance can search for the target object in different images using the following:
```cpp
bat_detector.detect_object(<image_path>); //search for a baseball bat in image
```
This returns the DetectedObject class.

### DetectedObject Class

This is a secondary class which acts as the interface to the required output of the program, being the coordinates of the detected object, and image with the object marked. It can save the output to specified locations using:
```cpp
DetectedObject result = bat_detector.detect_object(<image_path>);
result.save(<image_save_location>);
```
### Results
 The program was able to successfully locate the baseball bat within 83% of the provided images. In cases where the baseball bat was found, a bounding box and coordinate marker was drawn on top of the image and saved.
