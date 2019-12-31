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
	std::vector<std::string> classesFolder{ "Teeth","Motor","Black Item","Background" };
	struct RF RandomForest;
	struct HOGDesc HOG;
	struct ImgProc ImgConfig;


	};


