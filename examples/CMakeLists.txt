# Add the examples executable
add_executable(basic
    basic.cpp
)

# Link the tuatara library to the examples executable
target_link_libraries(basic
    PRIVATE
        tuatara
)
