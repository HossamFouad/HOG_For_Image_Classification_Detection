#include "task1.h"
#include <filesystem>
#include <string>
#include "hog_visualization.cpp"
#include <iostream>
#include <fstream>
#include <cstdio>
namespace fs = std::filesystem;

HOG::HOG(cv::Size p, cv::Size w, cv::Size bs, cv::Size bstride, cv::Size cz, int nb, int padType):
	image_size(p),
_wsize(w),
_blockSize(bs),
_blockStride(bstride),
_cellSize(cz),
_nbins(nb),
_padType(padType)
{}

HOG::~HOG() {
	imgsVec.clear();
	HOGVec.clear();
	imgListVec_.clear();
}
void HOG::loadImgs(bool M) {
	for (auto& imgName : imgListVec_) {
		imgsVec.push_back(std::make_unique<cv::Mat>(cv::imread(imgName)));
	}
	std::cout << "Num of images Loaded =" << imgsVec.size() << std::endl;
	for (int i = 0; i < imgListVec_.size();i++) {
		PadOrigin(i);
		if (M)GrayScale(i);

	}
}
void HOG::loadImgs(const std::vector<cv::Mat>& extImgVec) {
	imgsVec.clear();
	for (auto& img : extImgVec) {
		imgsVec.push_back(std::make_unique<cv::Mat>(img.clone()));
	}
	std::cout << "Num of images Loaded =" << imgsVec.size() << std::endl;
	for (int i = 0; i < imgsVec.size(); i++) {
		PadOrigin(i);
	}
}

void HOG::imgList(std::string p,float label) {
	std::vector<cv::String> fn;
	cv::glob(p+"/*.jpg", fn, false);

	for (auto& entry : fn) {
			imgListVec_.push_back(entry);
			labels.push_back(label);	
	}
		
}
unsigned HOG::imgNum() { return imgListVec_.size(); }

void HOG::logImgSize(int index, const std::string& s) {
	if (s == "I") {
		std::cout << "I" << index << ": Img size = " << imgsVec[index]->size() << " , Channels = " << imgsVec[index]->channels() << std::endl;

	}
	else {
		std::cout << "M" << index << ": Img size = " << ManimgsVec[index]->size() << " , Channels = " << ManimgsVec[index]->channels() << std::endl;
	}
}
void HOG::HOGExtractor(std::string p) {
	hog.winSize = _wsize;
	hog.blockSize = _blockSize;
	hog.blockStride = _blockStride;
	hog.cellSize = _cellSize;
	hog.nbins = _nbins;
	std::remove((p + std::string("Feats.csv")).c_str());
	std::remove((p + "Labels.csv").c_str());
	std::vector<float> feats;
	int i = 0;
	for(auto& imgptr: ManimgsVec){
		hog.compute(*imgptr, feats, cv::Size(8, 8), cv::Size(0, 0));
	std::cout<<"Extract "<< feats.size()<< " features from image "<<i++<<std::endl;
	cv::Mat f = static_cast<cv::Mat>(feats).reshape(1, 1); // flatten to a single row
	f.convertTo(f, CV_32F);     // ml needs float data
	featsImg.push_back(f);
	
	}

	if (!(p == "")) {
		std::ofstream outputFeats(p + "Feats.csv");
		outputFeats << format(featsImg, cv::Formatter::FMT_CSV) << std::endl;
		outputFeats.close();
		if (!labels.empty())
		{
			std::ofstream outputLabels(p + "Labels.csv");
			outputLabels << format(featslabel, cv::Formatter::FMT_CSV) << std::endl;
			outputLabels.close();
		}
	}
	
}
void HOG::HOGLoad(std::string p) {
	cv::Ptr<cv::ml::TrainData> feats_data = cv::ml::TrainData::loadFromCSV(p + "Feats.csv", 0, -2, 0);
	cv::Mat Fdata = feats_data->getSamples();
	Fdata.convertTo(featsImg, CV_32F);
	if (!labels.empty())
	{
		cv::Ptr<cv::ml::TrainData> labels_data = cv::ml::TrainData::loadFromCSV(p + "Labels.csv", 0, -2, 0);
		cv::Mat Ldata = labels_data->getSamples();
		Ldata.convertTo(featslabel, CV_32F);
	}


}

void HOG::visualizeImg(int i, const std::string&s ) {
	if (s == "I") {
		cv::namedWindow("Image= " + std::to_string(i)+", Class= "+ std::to_string(int(labels.at<float>(i)))+", path= "+ imgListVec_[i], cv::WINDOW_AUTOSIZE); // Create a window for display.
		cv::imshow("Image= " + std::to_string(i) + ", Class= " + std::to_string(int(labels.at<float>(i))) + ", path= " + imgListVec_[i], *imgsVec[i]);                // Show our image inside it.
		cv::waitKey(0); // Wait for a keystroke in the window

	}
	else {
		cv::namedWindow("ManipulatedImage= " + std::to_string(i) + ", Class= " + std::to_string(int(labels.at<float>(i))) + ", Manipulation= " + ManStr[i] + ", path= " + ", path= " + ManimgListVec_[i], cv::WINDOW_AUTOSIZE); // Create a window for display.
		cv::imshow("ManipulatedImage= " + std::to_string(i) + ", Class= " + std::to_string(int(labels.at<float>(i))) + ", Manipulation= " + ManStr[i]+ ", path= " + ManimgListVec_[i], *ManimgsVec[i]);                // Show our image inside it.
		cv::waitKey(0); // Wait for a keystroke in the window

	}
}
void HOG::setToIdentity(int index) {
	ManimgsVec.push_back(std::make_unique<cv::Mat>(imgsVec[index]->rows, imgsVec[index]->cols, imgsVec[index]->depth()));
	*(ManimgsVec.back()) = imgsVec[index]->clone();
	if (!labels.empty())featslabel.push_back(labels.at<float>(index));
	ManimgListVec_.push_back(imgListVec_[index]);
	ManStr.push_back("Identity");

}


void HOG::clearManVec() {
	ManimgsVec.clear();
	featsImg.release();
	featslabel.release();
	ManStr.clear();
	ManimgListVec_.clear();
}
void HOG::GrayScale(int index) {
	cv::cvtColor(*imgsVec[index], *imgsVec[index], cv::COLOR_BGR2GRAY);
	
}


void HOG::Resized(int index, float xfactor, float yfactor) {
	ManimgsVec.push_back(std::make_unique<cv::Mat>(imgsVec[index]->rows, imgsVec[index]->cols, imgsVec[index]->depth()));
	cv::resize(*imgsVec[index], *(ManimgsVec.back()), cv::Size(imgsVec[index]->cols * xfactor, imgsVec[index]->rows * yfactor), 0, 0);
	featslabel.push_back(labels.at<float>(index));
	ManimgListVec_.push_back(imgListVec_[index]);

}
void HOG::Rotated(int index, double angle, double scale) {
	ManimgsVec.push_back(std::make_unique<cv::Mat>(imgsVec[index]->rows, imgsVec[index]->cols, imgsVec[index]->depth()));
	cv::Mat rot_mat(2, 3, CV_32FC1);
	cv::Point center = cv::Point(imgsVec[index]->cols / 2, imgsVec[index]->rows / 2);
	rot_mat = cv::getRotationMatrix2D(center, angle, scale);
	warpAffine(*imgsVec[index], *(ManimgsVec.back()), rot_mat, imgsVec[index]->size());
	featslabel.push_back(labels.at<float>(index));
	ManimgListVec_.push_back(imgListVec_[index]);
	ManStr.push_back("Rotated(" + std::to_string(angle) + "," + std::to_string(scale) + ")");
}
void HOG::RotatedAndFlip(int index, double angle, double scale,int x) {
	ManimgsVec.push_back(std::make_unique<cv::Mat>(imgsVec[index]->rows, imgsVec[index]->cols, imgsVec[index]->depth()));
	cv::Mat rot_mat(2, 3, CV_32FC1);
	cv::Point center = cv::Point(imgsVec[index]->cols / 2, imgsVec[index]->rows / 2);
	rot_mat = cv::getRotationMatrix2D(center, angle, scale);
	warpAffine(*imgsVec[index], *(ManimgsVec.back()), rot_mat, imgsVec[index]->size());
	flip(*(ManimgsVec.back()), *(ManimgsVec.back()), x);
	featslabel.push_back(labels.at<float>(index));
	ManimgListVec_.push_back(imgListVec_[index]);
	ManStr.push_back("RotatedAndFlip(" + std::to_string(angle) + "," + std::to_string(scale) + "," + std::to_string(x) + ")");
}
void HOG::Flip(int index, int x) {
	ManimgsVec.push_back(std::make_unique<cv::Mat>(imgsVec[index]->rows, imgsVec[index]->cols, imgsVec[index]->depth()));
	flip(*imgsVec[index], *(ManimgsVec.back()), x);
	featslabel.push_back(labels.at<float>(index));
	ManimgListVec_.push_back(imgListVec_[index]);
	ManStr.push_back("Flip(" + std::to_string(x) + ")");

}
void HOG::PadOrigin(int index) {
	if (xmax < imgsVec[index]->size[0])xmax = imgsVec[index]->size[0];
	if (ymax < imgsVec[index]->size[1])ymax = imgsVec[index]->size[1];
	int left = int((image_size.height - imgsVec[index]->cols) / 2);
	int right = image_size.height - imgsVec[index]->cols - left;
	int top = int((image_size.width - imgsVec[index]->rows) / 2);
	int bottom = image_size.width - imgsVec[index]->rows - top;

	if (left < 0 || right < 0 || top < 0 || bottom < 0) {
		std::cout << "Stupid" << std::endl;
	}
	std::unique_ptr<cv::Mat>  padded= std::make_unique<cv::Mat>(image_size.height, image_size.width, imgsVec[index]->depth());
	
	cv::copyMakeBorder(*imgsVec[index], *padded, top, bottom, left, right, _padType);
	*imgsVec[index] = padded->clone();

}



std::unique_ptr<cv::Mat>& HOG::GetImage(int index) {
	return imgsVec[index];
}

std::unique_ptr<cv::Mat>& HOG::GetManImg(int index) {
	return ManimgsVec[index];
}


cv::Mat HOG::GetGroundTruth() {
	cv::Mat predMat = featslabel.clone();
	predMat.convertTo(predMat, 4);
	return predMat;
}
int HOG::GetManImgNum() {
	return ManimgListVec_.size();
}

int HOG::GetImgNum() {
	return imgListVec_.size();
}

void HOG::VisHOG(int index,int scale_factor) {
 	visualizeHOG(*ManimgsVec[index], static_cast<std::vector<float>>(featsImg.row(index)), hog, scale_factor);
}

void HOG::PrintImgList() {
	for (const auto& f : imgListVec_)std::cout << f << std::endl;
}
void HOG::PrintImglabels() {
	std::cout << labels << std::endl;
}
