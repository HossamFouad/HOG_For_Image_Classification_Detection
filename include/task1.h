#pragma once
#include<iostream>
#include <opencv2/opencv.hpp>

class HOG {
public:
	HOG(std::string);
	~HOG();
	void loadImgs();
	void imgList();
	void HOGExtractor(cv::Size,cv::Size,cv::Size,cv::Size, int);
	void VisHOG(int, int scale_factor=1);
	void PrintPath();
	void PrintImgList();
	unsigned imgNum();
	void visualizeImg(int,const std::string& s = "M");
	void GrayScale(int);
	void Resized(int, float, float);
	void Rotated(int, double, double scale = 1);
	void Flip(int, int);
	void PadOrigin(int, int);
	void clearManVec();
	void setToIdentity(int);
	std::unique_ptr<cv::Mat>& GetManImg(int);
	std::unique_ptr<cv::Mat>& GetImage(int);
	
protected:
	void logImgSize(int index,const std::string& s = "M");
	std::vector<std::unique_ptr<cv::Mat>> imgsVec;
	std::vector<std::unique_ptr<cv::Mat>> ManimgsVec;
	std::vector < std::unique_ptr < std::vector < float >> > featsVec;
	std::vector<cv::Mat> HOGVec;
private:
	cv::HOGDescriptor hog;
	std::string imgPath_;
	std::vector<std::string> imgListVec_;
};