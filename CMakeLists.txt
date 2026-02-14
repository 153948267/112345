cmake_minimum_required(VERSION 3.11)



project(YOLO_TRY)



set(CMAKE_CXX_STANDARD 17)



find_package(OpenCV REQUIRED)



include_directories(${OpenCV_INCLUDE_DIRS})



add_executable(YOLO_TRY src/main.cpp)



target_link_libraries(YOLO_TRY ${OpenCV_LIBS})



message("项目配置完成！")