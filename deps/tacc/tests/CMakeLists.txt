add_executable(test_basics test_basics.cpp)
target_include_directories(test_basics PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_basics PRIVATE tacc)