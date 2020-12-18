#include <DetectedObject.h>
#include <Utilities.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <fstream>

void DetectedObject::save(const std::string &out_path)
{
    cv::imwrite(out_path, image);

    std::string img_name = get_filename(out_path);
    std::fstream coords_file;
    coords_file.open("Output\\coords.txt", std::ios::app);
    if (!coords_file)
    {
        std::cout << "File not found.";
    }
    else
    {
        coords_file << img_name << "\t(" << coords.x << "," << coords.y << ")\n";
    }
    coords_file.close();
}
