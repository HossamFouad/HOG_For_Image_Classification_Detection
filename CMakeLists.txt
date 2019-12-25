# CMakeList.txt : CMake project for TDCV2, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("TDCV2")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package( OpenCV REQUIRED )

# list all the headers of the module
include_directories("${PROJECT_SOURCE_DIR}/include")

set(INC_DIR "${PROJECT_SOURCE_DIR}/include")
set(MODULE_HEADERS 
"${INC_DIR}/task1.h" 
"${INC_DIR}/task2.h" 
"${INC_DIR}/task3.h")
message("${MODULE_HEADERS}")
# list all the code files of the module
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(MODULE_SOURCES 
"${SRC_DIR}/task1.cpp" 
"${SRC_DIR}/task2.cpp" 
"${SRC_DIR}/task3.cpp")

message("${OpenCV_INCLUDE_DIRS}")
# Add source to this project's executable.
add_executable (TDCV2 "main.cpp" "${MODULE_SOURCES}" "${MODULE_HEADERS}")
target_link_libraries( TDCV2 ${OpenCV_LIBS} )

# TODO: Add tests and install targets if needed.
