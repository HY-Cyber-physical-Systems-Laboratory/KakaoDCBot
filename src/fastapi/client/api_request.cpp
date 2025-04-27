#include "fastapi/client/api_request.hpp"



std::string KakaoDCBOT::fastapi::client::api_request::get_http_response(const std::string& host, const std::string& port, const std::string& target)
{

    try
    {

        #ifdef HTTP_CLIENT_WITHOUT_BOOST
            asio::io_context ioc;
            asio::ip::tcp::resolver resolver(ioc);
            asio::ip::tcp::socket socket(ioc);

            auto const result = resolver.resolve(host, port);
            
            asio::connect(socket, result.begin(), result.end());

            asio::ip::tcp::no_delay option(true);
            socket.set_option(option);

            // HTTP GET 요청
            std::string request = "GET " + target + " HTTP/1.1\r\n";
            request += "Host: " + host + "\r\n";
            request += "User-Agent: ASIO\r\n";
            request += "Connection: close\r\n\r\n";

            asio::write(socket, asio::buffer(request));


            std::string response;
            response.reserve(BUFFER_SIZE);

            char buffer[BUFFER_SIZE];
            int len = 0;

            while((len = asio::read(socket, asio::buffer(buffer))) > 0)
            {
                response.append(buffer, len);
            }

            return response;

        #else
                net::io_context ioc;

                // Resolver 생성
                tcp::resolver resolver(ioc);
                beast::tcp_stream stream(ioc);

                // 호스트와 포트로 연결
                auto const results = resolver.resolve(host, port);
                stream.connect(results);

                // HTTP GET 요청 만들기
                http::request<http::string_body> req{http::verb::get, target, 11};
                req.set(http::field::host, host);
                req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);

                // 요청 전송
                http::write(stream, req);

                // 응답 받기
                beast::flat_buffer buffer;
                http::response<http::dynamic_body> res;

                http::read(stream, buffer, res);

                // 응답 출력
                return beast::buffers_to_string(res.data());
        #endif
        
    }
    catch (std::exception const& e)
    {
        std::cerr << "에러: " << e.what() << std::endl;
    }

    return "";

}


std::string KakaoDCBOT::fastapi::client::api_request::get_response_body(const std::string& host, const std::string& port, const std::string& target)
{
    std::string response = get_http_response(host, port, target);

    size_t index = response.find("\r\n\r\n");
    
    if (index == std::string::npos)
    {
        return "";
    }

    return response.substr(index + 4);
}



std::string KakaoDCBOT::fastapi::client::api_request::get_response_body_as_json(const std::string& host, const std::string& port, const std::string& target)
{
    
}

