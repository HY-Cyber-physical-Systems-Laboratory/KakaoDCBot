#ifndef __NAMESPACE_HPP__
#define __NAMESPACE_HPP__

#include <string>
#include <vector>

#include <simdjson.h>


namespace KakaoDCBOT
{
    namespace fastapi
    {
        namespace client
        {
            namespace api_request
            {
                std::string get_http_response(const std::string& host, const std::string& port, const std::string& target);
                std::string get_response_body(const std::string& host, const std::string& port, const std::string& target);
                std::string get_response_body_as_json(const std::string& host, const std::string& port, const std::string& target);

            }
        }

        namespace server
        {
            namespace api_main
            {

            }
        }
    }

    namespace utils
    {
        namespace string_utils
        {
            std::string to_upper(const std::string& str);
            std::string to_lower(const std::string& str);
            std::string trim(const std::string& str);
            std::vector<std::string> split(const std::string& str, char delimiter);
            
            simdjson::dom::element parse_json(const std::string& json_string);
        }

        namespace file_utils
        {
            bool file_exists(const std::string& filename);
            void create_directory(const std::string& path);
            void delete_file(const std::string& filename);
        }
    }
}



#endif