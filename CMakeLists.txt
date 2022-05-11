cmake_minimum_required(VERSION 3.16)

project (IntroToDetectors)

add_executable(app src/Main.cpp)

FIND_PACKAGE( OpenCV REQUIRED )
TARGET_LINK_LIBRARIES (app ${OpenCV_LIBS})    