// TDCV2.cpp : Defines the entry point for the application.
//
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "task1.h"
#include "task2.h"
#include<filesystem>
#include "Config.h"


int main()
{
	
	/* Task1
	// HOG Descriptor Parameters
	cv::Size imgSize = cv::Size(384,384);
	cv::Size WinSize = cv::Size(384, 384);
	cv::Size BlockSize = cv::Size(64, 64);
	cv::Size BlockStride = cv::Size(32, 32);
	cv::Size CellSize = cv::Size(32, 32);
	int padType = cv::BORDER_CONSTANT;
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
	auto p = new HOG(imgSize,WinSize, BlockSize, BlockStride, CellSize, Bins,padType) ;
	p->imgList(imgpath,0.0);
	p->loadImgs();
	p->visualizeImg(0,"I");
	//Origin
	p->setToIdentity(0);
	p->visualizeImg(0, "M");
	p->HOGExtractor();
	p->VisHOG(0);
	p->clearManVec();
	//Rotated
	p->Rotated(0, angle, scale);
	p->visualizeImg(0, "M");
	p->HOGExtractor();
	p->VisHOG(0);
	p->clearManVec();
	//Flip
	p->Flip(0, flipX);
	p->visualizeImg(0, "M");
	p->HOGExtractor();
	p->VisHOG(0);
	p->clearManVec();
	// Rotate and Flip
	p->RotatedAndFlip(0, angle, scale, flipX);
	p->visualizeImg(0, "M");
	p->HOGExtractor();
	p->VisHOG(0);
	p->clearManVec();
	*/

	/* Task2
	struct Config* config = new Config();
	std::string imgpath = (std::filesystem::current_path() / ".." / ".." / ".." / "data" / "task2").string();
	auto p = new Classifier(imgpath,config);
	if (!config->trained) {
		p->loadTrainImgs();
		p->imgsPreprocessing();
		p->trainRandomForest();
		//p->MultiAugTrain();

	}
	
	

	p->loadTestImgs();
	p->testRandomForest();
	delete config;

	*/
	
	return 0;
}
