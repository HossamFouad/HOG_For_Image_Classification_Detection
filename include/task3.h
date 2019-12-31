#pragma once
#include "task2.h"
#include <opencv2/opencv.hpp>
#include "Config.h"
class Detection:public Classifier{
public:
	Detection(std::string, struct DetectionConfig*);
	~Detection();
	void LoadInferenceImgs();
	void LoadGTBoundingBox();
	void SaveBoundingBoxGT();
	void TrainClassifier();



protected:

private:
	struct DetectionConfig* _config;
	std::vector<cv::Mat> _BoundingBoxImgs;
	std::vector<cv::Mat> _BoundingBoxGrayImgs;
	std::vector<std::string> _BoundingBoxImgsNames;
	std::vector<cv::Mat> _BoundingBoxGT;
	std::vector<cv::Mat> _BoundingBoxEst;
	std::string _imgPath;
};