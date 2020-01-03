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
	void ImgDetection(int);
	void PRCalculate();


protected:

private:
	void GenerateBoundingBox(int);
	void SaveBoundingBoxEst(int);
	struct DetectionConfig* _config;
	std::vector<cv::Mat> _BoundingBoxImgs;
	std::vector<cv::Mat> _GenerateBoundingBoxImgs;
	std::vector<cv::Mat> _GenerateBoundingBox;
	std::vector<cv::Mat> _BoundingBoxGrayImgs;
	std::vector<std::string> _BoundingBoxImgsNames;
	std::vector<cv::Mat> _BoundingBoxGT;
	std::vector<cv::Mat> _BoundingBoxEst;
	std::vector<cv::Mat> _BoundingBoxEstRect;
	std::vector<cv::Mat> _BoundingBoxEstImgs;
	std::vector<float> _BoundingBoxEstConfidence;
	std::vector<int> _BoundingBoxEstLabel;
	void sort();
	void NMS();
	cv::Mat precisionRecall(float threshold);
	float Overlap(const cv::Mat&, const cv::Mat&);
	float IOU(cv::Mat, cv::Mat);
	std::string _imgPath;
};