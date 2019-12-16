#include "task1.h"
#include <filesystem>


HOGDescriptor::HOGDescriptor(std::string p):imgPath_(p){

}

HOGDescriptor::~HOGDescriptor() {
	imgsVec.clear();
	HOGVec.clear();
	imgListVec_.clear();
}
void HOGDescriptor::loadImgs(){

}

void HOGDescriptor::imgList() {
	namespace fs = std::filesystem;
	for (const auto& entry : fs::directory_iterator(imgPath_)) {
		auto str = entry.path().string();
		if (str.find(".jpg") != std::string::npos|| str.find(".JPG") != std::string::npos) {
			imgListVec_.push_back(str);
		}
	}
		
}
unsigned HOGDescriptor::imgNum() { return imgListVec_.size(); }

void HOGDescriptor::HOGExtractor() {

}

void HOGDescriptor::VisualizeHOG() {

}

void HOGDescriptor::PrintPath() {
	std::cout << imgPath_ << std::endl;
	
}

void HOGDescriptor::PrintImgList() {
	for (const auto& f : imgListVec_)std::cout << f << std::endl;
}
