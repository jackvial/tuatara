add_executable(resume
    resume.cpp
)

add_executable(table
    table.cpp
)


# Link the tuatara library to the examples executable
target_link_libraries(resume
    PRIVATE
        tuatara
)

target_link_libraries(table
    PRIVATE
        tuatara
)
