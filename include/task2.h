#pragma once
#pragma once
#include<iostream>
#include "task1.h"
#include "RandomForest.h"
#include <opencv2/opencv.hpp>
#include<filesystem>
class HOG;
#include "Config.h"
class Classifier {
public:
	Classifier(std::string, struct Config*);
	Classifier(std::string, float, int, int, int, int, const std::vector<std::string>&, cv::Size, cv::Size, cv::Size, cv::Size, cv::Size, int, int,  std::vector<std::string>&);
	~Classifier();
	void loadTrainImgs();
	void loadTestImgs();
	void imgsPreprocessing();
	void MultiAugTrain();
	void trainRandomForest();
	void testRandomForest();
	cv::Mat inference(std::vector<cv::Mat>&,bool);

	
protected:
	std::unique_ptr<HOG> _trainHOG;
	std::unique_ptr<HOG> _testHOG;
	std::unique_ptr<RandomForest> _RF;
private:
	float Classifier::random_float(float, float);
	std::string _imgpath;
	int imgProcessingCount;
	//int _imgPadSize;
	//cv::Size _WinSize;
	//cv::Size _BlockSize;
	//cv::Size _BlockStride;
	//cv::Size _CellSize;
	//int _Bins;
	//float _randomSampleRatio;
	//int _num_trees;
	//int _cv_folds;
	//int _max_depth;
	//int _min_sample_count;
	std::vector<std::string> loadFolders(std::string);
	//int max_categories;
	//const std::vector<std::string>& folders;
	//cv::Size imgSize;
	//cv::Size w;
	//cv::Size bs;
	//cv::Size bstride;
	//cv::Size cz;
	//int nb;
	//std::vector<std::string> _className;
	int inferenceCount;
	struct Config* configPtr;
};