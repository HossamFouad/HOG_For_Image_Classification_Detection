#include "task3.h"
#include <filesystem>
#include <fstream>
#include <opencv2/plot.hpp>
namespace fs = std::filesystem;

Detection::Detection(std::string p,struct DetectionConfig* conf):
	Classifier(p,&(conf->classifier)),
	_imgPath(p),
	_config(conf){
	fs::remove_all((fs::path(_imgPath) / _config->Detectionfolders[3]).string().c_str());
	fs::create_directory((fs::path(_imgPath) / _config->Detectionfolders[3]).string().c_str());

}


void Detection::LoadInferenceImgs() {
	std::vector<std::string> fn;
	cv::glob(_imgPath +"/"+_config->classifier.folders[1]+ "/*.jpg", fn, false);
	for (auto& img : fn) {
		size_t found1 = img.rfind("/") > img.rfind("\\") ? img.rfind("/") : img.rfind("\\");
		size_t found2 = img.rfind(".");
		_BoundingBoxImgsNames.push_back(img.substr(found1 + 1, found2 - found1 - 1));
		cv::Mat imgMatrix = cv::imread(img);
		_BoundingBoxImgs.push_back(imgMatrix);
		cv::cvtColor(imgMatrix, imgMatrix, cv::COLOR_BGR2GRAY);
		_BoundingBoxGrayImgs.push_back(imgMatrix);
		imgMatrix.release();
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

void Detection::SaveBoundingBoxEst(int index) {
		cv::Mat img;
		img = _BoundingBoxImgs[index].clone();
		for (int j = 0; j < _BoundingBoxEstRect.size(); j++) {
			cv::Mat f = _BoundingBoxEstRect[j];
			cv::Point left{ f.at<int>(0),f.at<int>(1) };
			cv::Point right{ f.at<int>(2),f.at<int>(3) };
			for (int k = 0; k < _config->classesColors.size(); k++) {
				if (int(_BoundingBoxEstLabel[j]) == k)
				{
					cv::putText(img, "Class " + std::to_string(k)+ " (" + std::to_string(_BoundingBoxEstConfidence[j]).substr(0, 4) + ")",
						cv::Point(left.x, left.y - 4), // Coordinates
						cv::FONT_HERSHEY_PLAIN, // Font
						.5, // Scale. 2.0 = 2x bigger
						_config->classesColors[k], // BGR Color
						1); // Anti-alias (Optional)
					cv::rectangle(img, left, right, _config->classesColors[k]);
				}
			}
		}

		cv::imwrite((fs::path(_imgPath) / _config->Detectionfolders[2] / (std::to_string(index) + ".jpg")).string(), img);

	}
	


void Detection::TrainClassifier(){
	loadTrainImgs();
	imgsPreprocessing();
	trainRandomForest();

}

void Detection::ImgDetection(int index) {
	 _BoundingBoxEstRect.clear();
	_BoundingBoxEstConfidence.clear();
	_BoundingBoxEstLabel.clear();
	_BoundingBoxEstImgs.clear();
	
	GenerateBoundingBox(index);
	auto vec = inference(_GenerateBoundingBoxImgs);
	//std::cout << vec[0] << std::endl;
	//std::cout << vec[1] << std::endl;
	for (int i = 0; i < _GenerateBoundingBoxImgs.size(); i++) {
		if (vec[0].at<int>(i) == 3|| vec[1].at<float>(i) <_config->CondifentThres ) { continue; }
		//std::cout << vec[0].at<int>(i)<<","<< vec[1].at<float>(i) << std::endl;
		cv::Mat img = _GenerateBoundingBoxImgs[i].clone();
		_BoundingBoxEstRect.push_back(_GenerateBoundingBox[i].clone());
		_BoundingBoxEstConfidence.push_back(vec[1].at<float>(i));
		_BoundingBoxEstLabel.push_back(vec[0].at<int>(i));
		_BoundingBoxEstImgs.push_back(_GenerateBoundingBoxImgs[i].clone());
		cv::putText(img, "Class = " + std::to_string(vec[0].at<int>(i)),
			cv::Point(0, 10), // Coordinates
			cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
			.4, // Scale. 2.0 = 2x bigger
			cv::Scalar(255, 0, 255), // BGR Color
			1); // Anti-alias (Optional)
		cv::putText(img,"Belief = " + (std::to_string(vec[1].at<float>(i)).substr(0,4)),
			cv::Point(0, 20), // Coordinates
			cv::FONT_HERSHEY_TRIPLEX, // Font
			.4, // Scale. 2.0 = 2x bigger
			cv::Scalar(255, 0, 255), // BGR Color
			1); // Anti-alias (Optional)

		cv::imwrite((fs::path(_imgPath) / _config->Detectionfolders[3] / (_BoundingBoxImgsNames[index] + "_" + std::to_string(i) + ".jpg")).string(), img);
	}
	sort();
	NMS();
	SaveBoundingBoxEst(index);
	cv::Mat Rectimg;
	for (int i = 0; i < _BoundingBoxEstConfidence.size(); i++) {
		cv::Mat tmp;
		tmp.push_back(_BoundingBoxEstLabel[i]);
		tmp.push_back(_BoundingBoxEstRect[i].at<int>(0));
		tmp.push_back(_BoundingBoxEstRect[i].at<int>(1));
		tmp.push_back(_BoundingBoxEstRect[i].at<int>(2));
		tmp.push_back(_BoundingBoxEstRect[i].at<int>(3));
		Rectimg.push_back(tmp.reshape(1, 1).clone());
		tmp.release();
	}
	_BoundingBoxEst.push_back(Rectimg);
	//std::cout << _BoundingBoxEst[index] << std::endl;
	//std::cout << _BoundingBoxGT[index] << std::endl;
	//for (int i = 0; i < _BoundingBoxEstConfidence.size();i++)std::cout << _BoundingBoxEstConfidence[i] << "," << _BoundingBoxEstLabel[i] << std::endl;
}

void Detection::sort() {
	float tmpf;
	for (int i = 0; i < _BoundingBoxEstConfidence.size(); i++) {
		tmpf = _BoundingBoxEstConfidence[i];
		for (int j = i+1; j < _BoundingBoxEstConfidence.size(); j++) {
			if (tmpf < _BoundingBoxEstConfidence[j]) {
				tmpf = _BoundingBoxEstConfidence[j];
				std::swap(_BoundingBoxEstConfidence[i], _BoundingBoxEstConfidence[j]);
				std::swap(_BoundingBoxEstLabel[i], _BoundingBoxEstLabel[j]);
					std::swap(_BoundingBoxEstRect[i], _BoundingBoxEstRect[j]);
				std::swap(_BoundingBoxEstImgs[i], _BoundingBoxEstImgs[j]);
			}
		}
	}
}

void Detection::PRCalculate() {
	cv::Mat PR;
	for (auto thresh : _config->IOUthreshold) {
		auto PRtmp = precisionRecall(thresh);
		cv::Mat tmp;
		tmp.push_back(thresh);
		tmp.push_back(PRtmp.at<float>(0));
		tmp.push_back(PRtmp.at<float>(1));
		PR.push_back(tmp.reshape(1, 1));
	}
	//std::cout << PR<< std::endl;
	
	/*
	std::cout << PR.col(1) << std::endl;
	std::cout << PR.col(2) << std::endl;

	cv::Mat plot_result;
	cv::Mat dataX, dataY;
	PR.col(1).convertTo(dataX, CV_64F);
	PR.col(2).convertTo(dataY, CV_64F);
	cv::Ptr<cv::plot::Plot2d> plot;
	plot= cv::plot::Plot2d::create(dataX, dataY);
	plot->setPlotBackgroundColor(cv::Scalar(255, 255, 255));

	//Set plot line color
	plot->setPlotLineColor(cv::Scalar(50, 50, 255));
	plot->setShowText(true);
	plot->setInvertOrientation(true);

	plot->setPlotAxisColor(cv::Scalar(255, 0, 0));
	plot->render(plot_result);

	imshow("plot PR Curve", plot_result);
	cv::waitKey(0);
	*/
	//std::cout << _imgPath + "/PR.csv" << std::endl;
	std::ofstream outputFeats(_imgPath + "/PR.csv");
	outputFeats << format(PR, cv::Formatter::FMT_CSV) << std::endl;
	outputFeats.close();
}

cv::Mat Detection::precisionRecall(float threshold) {
	int sumEst{ 0 };
	int sumGT{ 0 };
	int sumPred{ 0 };
	cv::Mat tmp;
	for (int i = 0; i < _BoundingBoxEst.size(); i++) {
		//std::cout << _BoundingBoxEst[i] << std::endl;
			for (int j = 0; j < _BoundingBoxEst[i].rows; j++) {
			sumEst++;
			cv::Mat f = _BoundingBoxEst[i].row(j);
			sumPred+=IOU(f, _BoundingBoxGT[i].row(int(f.at<int>(0))))>threshold?1:0;
		}
		sumGT += _BoundingBoxGT[0].rows;
	}
	float precision = float(sumPred) / sumEst;
	float recall = float(sumPred) / sumGT;
	tmp.push_back(precision);
	tmp.push_back(recall);
	return tmp;
}
void Detection::NMS() {
	bool M = true;
	int j{ 0 };
	int size = _BoundingBoxEstConfidence.size();
	while (j<=size-1) {
		for (int i = j + 1; i < size; ){
			//std::cout << _BoundingBoxEstRect[j] << std::endl;
			//std::cout << _BoundingBoxEstRect[i] << std::endl;
			if (Overlap(_BoundingBoxEstRect[j], _BoundingBoxEstRect[i]) > _config->OverlapThres) {
				_BoundingBoxEstConfidence.erase(_BoundingBoxEstConfidence.begin() + i);
				_BoundingBoxEstLabel.erase(_BoundingBoxEstLabel.begin() + i);
				_BoundingBoxEstRect.erase(_BoundingBoxEstRect.begin() + i);
				_BoundingBoxEstImgs.erase(_BoundingBoxEstImgs.begin() + i);
				size--;
			}
			else {
				i++;
			}
		}
		j++;
	}
}
float Detection::Overlap(const cv::Mat& r1, const cv::Mat& r2){
	int xx1 = std::max(r1.at<int>(0), r2.at<int>(0));
	int yy1 = std::max(r1.at<int>(1), r2.at<int>(1));
	int xx2 = std::min(r1.at<int>(2), r2.at<int>(2));
	int yy2 = std::min(r1.at<int>(3), r2.at<int>(3));
	int area = (r2.at<int>(2) - r2.at<int>(0) + 1) * (r2.at<int>(3) - r2.at<int>(1) + 1);
	
	// compute the widthand height of the bounding box
	int w = std::max(0, xx2 - xx1 + 1);
	int h = std::max(0, yy2 - yy1 + 1);

	//compute the ratio of overlap between the computed
    // bounding box and the bounding box in the area list
	//std::cout << float(w * h) / area << std::endl;
	return  float(w * h) / area;

}

float Detection::IOU(cv::Mat r1,cv::Mat r2) {
	//std::cout << r1.at<int>(0, 1) << std::endl;
	//std::cout << r2.at<int>(0,1) << std::endl;
	int xx1 = std::max(r1.at<int>(0, 1), r2.at<int>(0, 1));
	int yy1 = std::max(r1.at<int>(0,2), r2.at<int>(0,2));
	int xx2 = std::min(r1.at<int>(0,3), r2.at<int>(0,3));
	int yy2 = std::min(r1.at<int>(0,4), r2.at<int>(0,4));
	
	
	// compute the widthand height of the bounding box
	int w = std::max(0, xx2 - xx1 + 1);
	int h = std::max(0, yy2 - yy1 + 1);

	int interArea = w * h;
	
	int boxAArea = (r1.at<int>(0,3) - r1.at<int>(0,1) + 1) * (r1.at<int>(0,4) - r1.at<int>(0,2) + 1);
	int boxBArea = (r2.at<int>(0,3) - r2.at<int>(0,1) + 1) * (r2.at<int>(0,4) - r2.at<int>(0,2) + 1);
	//compute the ratio of overlap between the computed
	// bounding box and the bounding box in the area list
	//std::cout << interArea / float(boxAArea + boxBArea - interArea) << std::endl;

	return  interArea / float(boxAArea + boxBArea - interArea);


}

void Detection::GenerateBoundingBox(int index) {
	_GenerateBoundingBoxImgs.clear();
	_GenerateBoundingBox.clear();
	cv::Mat img = _BoundingBoxGrayImgs[index];
	for (auto BBsize : _config->boundingbox) {
		cv::Mat DrawResultHere = img.clone();
		for (int row = 0; row <= img.rows - BBsize; row += int(BBsize*_config->StepSlide))
		{
			// Cycle col step
			for (int col = 0; col <= img.cols - BBsize; col += int(BBsize * _config->StepSlide))
			{
				cv::Mat tmp;
				cv::Rect windows(col, row, BBsize, BBsize);
				_GenerateBoundingBoxImgs.push_back(img(windows).clone());
				tmp.push_back(col);
				tmp.push_back(row);
				tmp.push_back(col+BBsize);
				tmp.push_back(row+BBsize);
				_GenerateBoundingBox.push_back(tmp.clone());
				tmp.release();
				// Draw only rectangle
				//cv::rectangle(DrawResultHere, windows, cv::Scalar(255), 1, 8, 0);
			}
		}
	}
}
