#pragma once
#include<iostream>
#include <opencv2/opencv.hpp>
class HOGDescriptor {
public:
	HOGDescriptor(std::string);
	~HOGDescriptor();
	void loadImgs();
	void imgList();
	void HOGExtractor();
	void VisualizeHOG();
	void PrintPath();
	void PrintImgList();
	unsigned imgNum();
protected:
	std::vector<cv::Mat> imgsVec;
	std::vector<cv::Mat> HOGVec;
private:
	std::string imgPath_;
	std::vector<std::string> imgListVec_;
};