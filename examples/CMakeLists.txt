add_executable(resume
    resume.cpp
)

add_executable(table
    table.cpp
)

target_link_libraries(resume
    PRIVATE
        tuatara
)

target_link_libraries(table
    PRIVATE
        tuatara
)
