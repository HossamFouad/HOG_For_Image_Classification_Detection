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
	struct Config* config = new Config();
	
	// HOG Descriptor Parameters
	/*cv::Size imgSize = cv::Size(384,384);
	cv::Size WinSize = cv::Size(384, 384);
	cv::Size BlockSize = cv::Size(64, 64);
	cv::Size BlockStride = cv::Size(32, 32);
	cv::Size CellSize = cv::Size(32, 32);
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
	auto p = new HOG(imgSize,WinSize, BlockSize, BlockStride, CellSize, Bins) ;
	p->imgList(imgpath,0.0);
	p->loadImgs();
	p->visualizeImg(0,"I");
	//p->RotatedAndFlip(0, angle, 1.2,flipX);
	p->visualizeImg(0, "M");
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
	
	float randomSampleRatio = 0.90;
	int num_trees = 10;
	int cv_folds = 1;
	int max_depth = 25;
	int categoriesNum = 6;
	int min_sample_count = 40;
	std::vector<std::string> folder{ "train","test","models","predictions" };
	std::vector<std::string> classesFolder{ "Teeth","Motor","Black Item","Background" };
	std::string imgpath = (std::filesystem::current_path() / ".." / ".." / ".." / "data" / "task2").string();
	auto p = new Classifier(imgpath, randomSampleRatio, num_trees, cv_folds, max_depth, min_sample_count, 
		folder,imgSize, WinSize, BlockSize, BlockStride, CellSize, Bins, categoriesNum, classesFolder);

	
	bool Manipulation = true;
	int NumManPerImg = 32;
	bool loaded = true;
	bool trained = true;
	*/
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

	return 0;
}
