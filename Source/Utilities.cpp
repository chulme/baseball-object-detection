#include <Utilities.h>

#include <string>

std::string get_filename(const std::string &path)
{
    std::string name = path.substr(path.find_last_of("/\\") + 1);
    size_t dot_i = path.find_last_of('.');
    return name.substr(0, dot_i);
}