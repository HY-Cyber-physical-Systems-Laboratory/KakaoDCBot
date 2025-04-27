#ifndef __UTILS_FILE_UTILS_HPP__
#define __UTILS_FILE_UTILS_HPP__

#include <string>
#include <namespace.hpp>

bool KakaoDCBOT::utils::file_utils::file_exists(const std::string& filename);
void KakaoDCBOT::utils::file_utils::create_directory(const std::string& path);
void KakaoDCBOT::utils::file_utils::delete_file(const std::string& filename);


#endif