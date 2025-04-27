#include <crow.h>

int api_main()
{
    crow::SimpleApp app;

    CROW_ROUTE(app, "/api")
    ([] {
        return "Hello from the API!";
    });

    app.port(8080).multithreaded().run();
    return 0;
}