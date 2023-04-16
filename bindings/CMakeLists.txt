find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(pytuatara python.cpp)

target_link_libraries(pytuatara
    PRIVATE
        tuatara
)