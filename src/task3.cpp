#include "task3.h"
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

Detection::Detection(std::string p,struct DetectionConfig* conf):
	Classifier(p,&(conf->classifier)),
	_imgPath(p),
	_config(conf){}


void Detection::LoadInferenceImgs() {
	std::vector<std::string> fn;
	cv::glob(_imgPath +"/"+_config->classifier.folders[1]+ "/*.jpg", fn, false);
	for (auto& img : fn) {
		size_t found1 = img.rfind("/") > img.rfind("\\") ? img.rfind("/") : img.rfind("\\");
		size_t found2 = img.rfind(".");
		_BoundingBoxImgsNames.push_back(img.substr(found1 + 1, found2 - found1 - 1));
		_BoundingBoxImgs.push_back(cv::imread(img));
		_BoundingBoxGrayImgs.push_back(cv::imread(img,cv::COLOR_RGB2GRAY));
	}
 }
void Detection::LoadGTBoundingBox() {
	std::ifstream inFile;
	int x;
	cv::Mat tmp;
	for (auto& BBimgName : _BoundingBoxImgsNames) {
		inFile.open((fs::path(_imgPath) / _config->Detectionfolders[0] / (BBimgName+ ".gt.txt")).string());
		if (!inFile) {
			std::cout << "Unable to open file";
			exit(1); // terminate with error
		}

		while (inFile >> x) {
			tmp.push_back(x);
		}
		tmp = tmp.reshape(5);
		_BoundingBoxGT.push_back(tmp.clone());
		tmp.release();
		inFile.close();
	}
	}
	
void Detection::SaveBoundingBoxGT() {
	int max{ 0 }, min{ _BoundingBoxImgs[0].rows };
	for (int i = 0; i < _BoundingBoxImgs.size(); i++) {
		cv::Mat img;
		img = _BoundingBoxImgs[i].clone();
		for (int j = 0; j<_BoundingBoxGT[i].rows; j++) {
			cv::Mat f = _BoundingBoxGT[i].row(j);
			cv::Point left{ f.at<int>(0,1),f.at<int>(0,2) };
			cv::Point right{ f.at<int>(0,3),f.at<int>(0,4) };
			for (int k = 0; k < _config->classesColors.size(); k++) {
				if (f.at<int>(0, 0) == k)
				{
					cv::putText(img, "Class "+ std::to_string(k),
						cv::Point(left.x, left.y-4), // Coordinates
						cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
						.5, // Scale. 2.0 = 2x bigger
						_config->classesColors[k], // BGR Color
						1); // Anti-alias (Optional)
					cv::rectangle(img, left, right, _config->classesColors[k]);
				}
			}
			if (max < f.at<int>(0, 3) - f.at<int>(0, 1))max = f.at<int>(0, 3) - f.at<int>(0, 1);
			if (min > f.at<int>(0, 3) - f.at<int>(0, 1))min = f.at<int>(0, 3) - f.at<int>(0, 1);
		}
		
		cv::imwrite((fs::path(_imgPath) / _config->Detectionfolders[1] / (_BoundingBoxImgsNames[i] + ".jpg")).string(), img);
		
	}
	std::cout << "min window size= "<<min<<",max window size= "<<max << std::endl;


}
void Detection::TrainClassifier(){
	loadTrainImgs();
	imgsPreprocessing();
	trainRandomForest();

}