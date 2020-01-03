// TDCV2.cpp : Defines the entry point for the application.
//
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "task1.h"
#include "task2.h"
#include "task3.h"
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

	// Task2
	/*struct Config* config = new Config();
	std::string imgpath = (std::filesystem::current_path() / ".." / ".." / ".." / "data" / "task2").string();
	auto p = new Classifier(imgpath,config);
	config->loaded = false;
	config->trained = false;
	//config->ImgConfig.NumManPerImg = 5;
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
	
	
	 //Task3
	struct DetectionConfig* config = new DetectionConfig();
	config->classifier.loaded = true;
	config->classifier.trained = false;
	config->classifier.RandomForest.randomSampleRatio = 0.95;
	config->classifier.ImgConfig.NumManPerImg = 40;
	//config->classifier.ImgConfig.FixedPad = false;
	//config->classifier.ImgConfig.weighted = true;
	if (config->classifier.ImgConfig.FixedPad&& config->classifier.ImgConfig.weighted) {
		config->classifier.ImgConfig.classesCount= std::vector<int>{ 3,2,3,1 };
	}
	else if(!config->classifier.ImgConfig.FixedPad){
		config->classifier.ImgConfig.classesCount = std::vector<int>{ 3,2,3, int(config->classifier.ImgConfig.PadImgNum)};
	}
	config->classifier.ImgConfig.scale[1] = 1.3;
	config->classifier.ImgConfig.scale[0] = 0.6;
	config->classifier.RandomForest.categoriesNum = 4;
	config->classifier.RandomForest.max_depth = 30;
	config->classifier.RandomForest.num_trees = 6;
	//config->classifier.ImgConfig.angle[0] = -45;
	//config->classifier.ImgConfig.angle[1] = 45;
	std::string imgpath = (std::filesystem::current_path() / ".." / ".." / ".." / "data" / "task3").string();
	auto p = new Detection(imgpath,config);
	p->LoadInferenceImgs();
	p->LoadGTBoundingBox();
	//p->SaveBoundingBoxGT();
	if(!config->classifier.trained)p->TrainClassifier();
	//p->ImgDetection(0);
	for (int i=0;i<43;i++)p->ImgDetection(i);
	p->PRCalculate();
	delete config;
	
	
	return 0;
}
