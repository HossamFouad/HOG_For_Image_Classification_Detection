#include "task1.h"
#include <filesystem>
#include <string>
#include "hog_visualization.cpp"

HOG::HOG(std::string p):imgPath_(p){

}

HOG::~HOG() {
	imgsVec.clear();
	HOGVec.clear();
	imgListVec_.clear();
}
void HOG::loadImgs() {
	for (auto& imgName : imgListVec_) {
		imgsVec.push_back(std::make_unique<cv::Mat>(cv::imread(imgName)));
	}
	std::cout << "Num of images Loaded =" << imgsVec.size() << std::endl;

}

void HOG::imgList() {
	namespace fs = std::filesystem;
	for (const auto& entry : fs::directory_iterator(imgPath_)) {
		auto str = entry.path().string();
		if (str.find(".jpg") != std::string::npos|| str.find(".JPG") != std::string::npos) {
			imgListVec_.push_back(str);
		}
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
void HOG::HOGExtractor(cv::Size _wsize, cv::Size _blockSize, cv::Size _blockStride, cv::Size _cellSize, int _nbins) {
	hog.winSize = _wsize;
	hog.blockSize = _blockSize;
	hog.blockStride = _blockStride;
	hog.cellSize = _cellSize;
	hog.nbins = _nbins;
	std::vector<float> feats;
	for(auto& imgptr: ManimgsVec){
		hog.compute(*imgptr, feats, cv::Size(8, 8), cv::Size(0, 0));
	std::cout<<feats.size()<<std::endl;
	featsVec.push_back(std::make_unique<std::vector<float>>(feats));
	}
}
void HOG::visualizeImg(int i, const std::string& s) {
	if (s == "I") {
		cv::namedWindow("Image " + std::to_string(i), cv::WINDOW_AUTOSIZE); // Create a window for display.
		cv::imshow("Image " + std::to_string(i), *imgsVec[i]);                // Show our image inside it.
		cv::waitKey(0); // Wait for a keystroke in the window

	}
	else {
		cv::namedWindow("ManipulatedImage " + std::to_string(i), cv::WINDOW_AUTOSIZE); // Create a window for display.
		cv::imshow("ManipulatedImage " + std::to_string(i), *ManimgsVec[i]);                // Show our image inside it.
		cv::waitKey(0); // Wait for a keystroke in the window

	}
}
void HOG::setToIdentity(int index) {
	ManimgsVec.push_back(std::make_unique<cv::Mat>(imgsVec[index]->rows, imgsVec[index]->cols, imgsVec[index]->depth()));
	*(ManimgsVec.back()) = imgsVec[index]->clone();
}
void HOG::clearManVec() {
	ManimgsVec.clear();
	featsVec.clear();
}
void HOG::GrayScale(int index) {
	ManimgsVec.push_back(std::make_unique<cv::Mat>(imgsVec[index]->rows, imgsVec[index]->cols, imgsVec[index]->depth()));
	cv::cvtColor(*imgsVec[index], *(ManimgsVec.back()), cv::COLOR_BGR2GRAY);
	logImgSize(0);
}


void HOG::Resized(int index, float xfactor, float yfactor) {
	ManimgsVec.push_back(std::make_unique<cv::Mat>(imgsVec[index]->rows, imgsVec[index]->cols, imgsVec[index]->depth()));
	cv::resize(*imgsVec[index], *(ManimgsVec.back()), cv::Size(imgsVec[index]->cols * xfactor, imgsVec[index]->rows * yfactor), 0, 0);
}
void HOG::Rotated(int index, double angle, double scale) {
	ManimgsVec.push_back(std::make_unique<cv::Mat>(imgsVec[index]->rows, imgsVec[index]->cols, imgsVec[index]->depth()));
	cv::Mat rot_mat(2, 3, CV_32FC1);
	cv::Point center = cv::Point(imgsVec[index]->cols / 2, imgsVec[index]->rows / 2);
	rot_mat = cv::getRotationMatrix2D(center, angle, scale);
	warpAffine(*imgsVec[index], *(ManimgsVec.back()), rot_mat, imgsVec[index]->size());

}
void HOG::Flip(int index, int x) {
	ManimgsVec.push_back(std::make_unique<cv::Mat>(imgsVec[index]->rows, imgsVec[index]->cols, imgsVec[index]->depth()));
	flip(*imgsVec[index], *(ManimgsVec.back()), x);
}
void HOG::PadOrigin(int index, int image_size) {
	cv::Mat image_padded(image_size, image_size, imgsVec[index]->depth());
	int left = int((image_size - imgsVec[index]->cols) / 2);
	int right = image_size - imgsVec[index]->cols - left;
	int top = int((image_size - imgsVec[index]->rows) / 2);
	int bottom = image_size - imgsVec[index]->rows - top;

	if (left < 0 || right < 0 || top < 0 || bottom < 0) {
		std::cout << "Stupid" << std::endl;
	}
	std::unique_ptr<cv::Mat>  padded= std::make_unique<cv::Mat>(image_size, image_size, imgsVec[index]->depth());
	cv::copyMakeBorder(*imgsVec[index], *padded, top, bottom, left, right, cv::BORDER_CONSTANT);
	*imgsVec[index] = padded->clone();

}



std::unique_ptr<cv::Mat>& HOG::GetImage(int index) {
	return imgsVec[index];
}
std::unique_ptr<cv::Mat>& HOG::GetManImg(int index) {
	return ManimgsVec[index];
}
void HOG::VisHOG(int index,int scale_factor) {
 	visualizeHOG(*ManimgsVec[index], *featsVec[index], hog, scale_factor);
}

void HOG::PrintPath() {
	std::cout << imgPath_ << std::endl;
	
}

void HOG::PrintImgList() {
	for (const auto& f : imgListVec_)std::cout << f << std::endl;
}
