// TDCV2.cpp : Defines the entry point for the application.
//
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "task1.h"
#include<filesystem>
int main()
{   // HOG Descriptor Parameters
	cv::Size WinSize = cv::Size(500, 500);
	cv::Size BlockSize = cv::Size(50, 50);
	cv::Size BlockStride = cv::Size(5, 5);
	cv::Size CellSize = cv::Size(25, 25);
	int Bins = 8;
	// Resize
	float xfactor = 2;
	float yfactor = 2;
	// Rotate
	float angle = 60;
	float scale = 1;
	//Flip
	float flipX = 0;
	//Pad
	float borders = 4;
	std::string imgpath = (std::filesystem::current_path() /".."/".."/".."/"data"/"task1").string();
	auto p = new HOG(imgpath) ;
	p->imgList();
	p->loadImgs();
	p->PadOrigin(0, 500);
	p->visualizeImg(0,"I");
	//GrayScale
	p->GrayScale(0);
	p->visualizeImg(0);
	p->HOGExtractor(WinSize,BlockSize,BlockStride,CellSize,Bins);
	p->VisHOG(0);
	p->clearManVec();

	//Rotated
	p->Rotated(0, angle, scale);
	p->visualizeImg(0);
	p->HOGExtractor(WinSize, BlockSize, BlockStride, CellSize, Bins);
	p->VisHOG(0);
	p->clearManVec();
	//Flip
	p->Flip(0, flipX);
	p->visualizeImg(0);
	p->HOGExtractor(WinSize, BlockSize, BlockStride, CellSize, Bins);
	p->VisHOG(0);
	p->clearManVec();
	//Origin
	p->setToIdentity(0);
	p->visualizeImg(0);
	p->HOGExtractor(WinSize, BlockSize, BlockStride, CellSize, Bins);
	p->VisHOG(0);
	p->clearManVec();
	return 0;
}
