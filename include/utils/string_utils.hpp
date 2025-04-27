#ifndef __UTILS_STRING_UTILS_HPP__
#define __UTILS_STRING_UTILS_HPP__

#include <string>
#include <vector>
#include <namespace.hpp>


std::string KakaoDCBOT::utils::string_utils::to_upper(const std::string& str)
{

}


std::string KakaoDCBOT::utils::string_utils::to_lower(const std::string& str)
{

}

std::string KakaoDCBOT::utils::string_utils::trim(const std::string& str)
{

}

std::vector<std::string> KakaoDCBOT::utils::string_utils::split(const std::string& str, char delimiter)
{
    
}


simdjson::dom::element KakaoDCBOT::utils::string_utils::parse_json(const std::string& json_string) {
    static simdjson::dom::parser parser;
    return parser.parse(json_string); // 안전하게 반환 가능!
}

#endif