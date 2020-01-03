#pragma once

#include <opencv2/opencv.hpp>

struct RF {
	float randomSampleRatio = 0.90;
	int num_trees = 10;
	int cv_folds = 1;
	int max_depth = 25;
	int categoriesNum = 6;
	int min_sample_count = 40;
};

struct HOGDesc {
	cv::Size imgSize = cv::Size(384, 384);
	cv::Size WinSize = cv::Size(384, 384);
	cv::Size BlockSize = cv::Size(64, 64);
	cv::Size BlockStride = cv::Size(32, 32);
	cv::Size CellSize = cv::Size(32, 32);
	int Bins = 8;
};

struct ImgProc {
	int PadImgNum = 3;
	bool FixedPad = true;
	std::vector<int> classesCount{ 1,1,1,1 };
	bool weighted = false;
	bool Manipulation = true; // enable manipulation
	int NumManPerImg =32; // number of augmentations per image
	float angle[4] = { -30.0,30.0 ,80,110};
	float scale[2] = { 0.9,1.2 };
	int padBorder = cv::BORDER_CONSTANT;
};

struct Config {
	bool loaded = true;// Loading HOG features
	bool trained = true; // loading trained random forest
	std::vector<std::string> folders{ "train","test","models","predictions" };
	struct RF RandomForest;
	struct HOGDesc HOG;
	struct ImgProc ImgConfig;


	};

struct DetectionConfig {
	std::vector<std::string> classesFolder{ "Teeth","Motor","Black Item","Background" };	
	std::vector<cv::Scalar> classesColors{ cv::Scalar(0, 255, 0),cv::Scalar(255, 0, 0),cv::Scalar(0, 0, 255)};
	std::vector<std::string> Detectionfolders{ "gt","boundingBoxGT","boundingBoxInference","boundingBoxExperiment" };
	std::vector<int> boundingbox{ 90,180 };//{ 64,80,96,112,128,144,160,176,192,208,224 };// , 150, 200, 250};
	std::vector<float> IOUthreshold{ 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
	float CondifentThres = 0.7;
	float OverlapThres = 0.0;
	float StepSlide = 0.3;
	struct Config classifier;
};


