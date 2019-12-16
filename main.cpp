// TDCV2.cpp : Defines the entry point for the application.
//
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "task1.h"
#include<filesystem>
int main()
{
	std::string imgpath = (std::filesystem::current_path() /"../../../data/task1").string();
	auto p = new HOGDescriptor(imgpath) ;
	p->PrintPath();
	p->imgList();
	p->PrintImgList();
	return 0;
}
