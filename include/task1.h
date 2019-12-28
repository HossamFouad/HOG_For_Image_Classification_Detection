#pragma once
#include<iostream>
#include <opencv2/opencv.hpp>
#include "task2.h"
class HOG {
public:
	friend class Classifier;
	HOG(cv::Size, cv::Size, cv::Size, cv::Size, cv::Size, int);
	virtual ~HOG();
	void loadImgs(bool M = true);
	void imgList(std::string,float);
	void HOGExtractor(std::string p);
	void HOGLoad(std::string p);
	void VisHOG(int, int scale_factor=1);
	void PrintImgList();
	unsigned imgNum();
	void visualizeImg(int,const std::string& s = "M");
	void GrayScale(int);
	void Resized(int, float, float);
	void Rotated(int, double, double scale = 1);
	void RotatedAndFlip(int index, double angle, double scale, int x);
	void Flip(int, int);
	void PadOrigin(int);
	void clearManVec();
	void setToIdentity(int);
	void PrintImglabels();
	std::unique_ptr<cv::Mat>& GetManImg(int);
	std::unique_ptr<cv::Mat>& GetImage(int);
	int GetManImgNum();
	int GetImgNum();
	int xmax = 0, ymax = 0;
protected:
	void logImgSize(int index,const std::string&);
	std::vector<std::unique_ptr<cv::Mat>> imgsVec;
	std::vector<std::unique_ptr<cv::Mat>> ManimgsVec;
	cv::Mat featsImg;
	cv::Mat labels;
	cv::Mat featslabel;
	std::vector<cv::Mat> HOGVec;
private:
	cv::Size image_size;
	cv::Size _wsize;
	cv::Size _blockSize;
	cv::Size _blockStride;
	cv::Size _cellSize;
	int _nbins;
	
	
	cv::HOGDescriptor hog;
	std::vector<std::string> ManimgListVec_;
	std::vector<std::string> imgListVec_;
};