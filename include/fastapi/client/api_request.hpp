#ifndef __FASTAPI_CLIENT_API_REQUEST_HPP__
#define __FASTAPI_CLIENT_API_REQUEST_HPP__

#include <namespace.hpp>
#include <utils/string_utils.hpp>
#include <string>
#include <iostream>

#define ASIO_STANDALONE

#ifdef HTTP_REQUEST_BUFFER_LARGE
    #define BUFFER_SIZE 8192
#else
    #define BUFFER_SIZE 1024
#endif
#if defined(ASIO_STANDALONE) && !defined(BOOST_WINDOWS) && !defined(BOOST_POSIX_API)
    #define HTTP_CLIENT_WITHOUT_BOOST
    #include <asio.hpp>
#else
    #include <boost/beast/core.hpp>
    #include <boost/beast/http.hpp>
    #include <boost/beast/version.hpp>
    #include <boost/asio/connect.hpp>
    #include <boost/asio/ip/tcp.hpp>
    
    using namespace boost::beast;
    using tcp = boost::asio::ip::tcp; // from <boost/asio/ip/tcp.hpp>
    namespace net = boost::asio; // from <boost/asio.hpp>
    namespace http = beast::http; // from <boost/beast/http.hpp>
    namespace beast = boost::beast; // from <boost/beast.hpp>
    
#endif


std::string KakaoDCBOT::fastapi::client::api_request::get_http_response(const std::string& host, const std::string& port, const std::string& target);

std::string KakaoDCBOT::fastapi::client::api_request::get_response_body(const std::string& host, const std::string& port, const std::string& target);

std::string KakaoDCBOT::fastapi::client::api_request::get_response_body_as_json(const std::string& host, const std::string& port, const std::string& target);



#endif // __FASTAPI_CLIENT_API_REQUEST_HPP__