#include "task2.h"
#include <algorithm>
#include <random>
#include <string>

namespace fs = std::filesystem;

Classifier::Classifier(std::string path, struct Config* config):
_imgpath(path),
configPtr(config) , 
_trainHOG(nullptr),
imgProcessingCount(0)
{

	_testHOG = std::make_unique<HOG>(configPtr);
	_RF = std::make_unique<RandomForest>(configPtr->RandomForest.randomSampleRatio, configPtr->RandomForest.num_trees, configPtr->RandomForest.cv_folds, configPtr->RandomForest.categoriesNum, configPtr->RandomForest.max_depth, configPtr->RandomForest.min_sample_count);

}

/*Classifier::Classifier(std::string path, float randomSampleRatio, int num_trees, int cv_folds, int max_depth, int min_sample_count,
	 const std::vector<std::string>& folder, cv::Size imageSize, cv::Size WinSize, cv::Size BlockSize, cv::Size BlockStride, cv::Size CellSize, int Bins, int categoriesNum, std::vector<std::string>& className):
	_imgpath(path),
	_randomSampleRatio(randomSampleRatio),
	_num_trees(num_trees),
	_cv_folds(cv_folds),
	_max_depth(max_depth),
	_min_sample_count(min_sample_count),
	folders(folder),
	imgSize(imageSize),
	w(WinSize),
	bs(BlockSize),
	bstride(BlockStride),
	cz(CellSize),
	nb(Bins),
	_trainHOG(nullptr),
	_testHOG(nullptr),
	_RF(nullptr),
	imgProcessingCount(0),
	inferenceCount(0),
	max_categories(categoriesNum),
	_className(className)

{

	_testHOG = std::make_unique<HOG>(imgSize, w, bs, bstride, cz, nb);
	_RF = std::make_unique<RandomForest>(_randomSampleRatio, _num_trees, _cv_folds, max_categories, _max_depth, _min_sample_count);

}*/

void Classifier::loadTrainImgs()
{
	std::vector<std::string> f = loadFolders(configPtr->folders[0]);
	_trainHOG = std::make_unique<HOG>(configPtr);
	for (const auto& folder : f) {
		_trainHOG->imgList((fs::path(_imgpath) / configPtr->folders[0]/folder).string() ,static_cast<float>(folder[folder.length()-1]) - 48.0);
	}
	_trainHOG->loadImgs(configPtr->ImgConfig.FixedPad, configPtr->ImgConfig.PadImgNum);
	//for(int i =0;i<_trainHOG->GetImgNum();i++){
		//cv::imwrite((fs::path(_imgpath) / std::string("boundingBoxExperiment") / (std::to_string(i) + ".jpg")).string(), *(_trainHOG->imgsVec[i]));
	//}
		//_trainHOG->visualizeImg(i, "I");
	std::cout << "TrainMaxSize = " << _trainHOG->xmax << " X " << _trainHOG->ymax << std::endl;
}


void Classifier::loadTestImgs()
{
	_testHOG->clearManVec();
	std::vector<std::string> f = loadFolders(configPtr->folders[1]);
	for (const auto& folder : f) {
		_testHOG->imgList((fs::path(_imgpath) / configPtr->folders[1] / folder).string(), static_cast<float>(folder[folder.length() - 1]) - 48.0);
	}
	_testHOG->loadImgs();
	for (int i = 0; i < _testHOG->GetImgNum(); i++) {
		_testHOG->setToIdentity(i);
	}	
	std::cout << "TestMaxSize = " << _testHOG->xmax << " X " << _testHOG->ymax << std::endl;
}

Classifier::~Classifier() {}

std::vector<std::string> Classifier::loadFolders(std::string p) {
	std::vector<cv::String> fn;
	std::string path = (fs::path(_imgpath) / p).string();
	for (const auto& entry : fs::directory_iterator(path)) {
		auto str = entry.path().string();
		fn.push_back(str.substr(str.length() - 2));
	}
	return fn;
}

void Classifier::trainRandomForest(){
	_RF->createRF();
	std::cout << "---------------Training---------------" << std::endl;
	_RF->train(_trainHOG->featsImg,_trainHOG->featslabel);
	_RF->save((fs::path(_imgpath)/ configPtr->folders[2]).string());
}
void Classifier::testRandomForest() {
	if (configPtr->loaded) {
		std::cout << "---------------test features Loading---------------" << std::endl;
		_testHOG->HOGLoad((fs::path(_imgpath) / configPtr->folders[1]).string());
	}
	else {
		
		std::cout << "---------------test features Extraction---------------" << std::endl;
		_testHOG->HOGExtractor((fs::path(_imgpath) / configPtr->folders[1]).string());
	}
	
	if (configPtr->trained) {
		_RF->load((fs::path(_imgpath) / configPtr->folders[2]).string());
	}
	std::cout << "---------------Prediction---------------" << std::endl;
	_RF->Prediction(_testHOG->featsImg);
	std::cout << "---------------Accuracy---------------" << std::endl;
	_RF->accuracy(_testHOG->featslabel);
	
	auto pred = _RF->GetPrediction();
	auto GT = _testHOG->GetGroundTruth();
	for (int i = 0; i < _testHOG->ManimgsVec.size();i++) {
		std::cout << "save img " << i << std::endl;
		cv::Mat img = _testHOG->ManimgsVec[i]->clone();
		cv::putText(img,"GT = "+std::to_string(GT.at<int>(i))+ ", Predicted = " + std::to_string(pred.at<int>(i)),
			cv::Point(10, 10), // Coordinates
			cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
			.5, // Scale. 2.0 = 2x bigger
			cv::Scalar(255, 0, 255), // BGR Color
			1); // Anti-alias (Optional)
		cv::imwrite((fs::path(_imgpath) / configPtr->folders[3] / (std::to_string(i) + ".jpg")).string(), img);
	}

}

std::vector<cv::Mat> Classifier::inference(std::vector<cv::Mat>& imgVec) {
	_testHOG->clearManVec();
	std::vector<cv::Mat> predVec;
	_testHOG->loadImgs(imgVec);
	for (int i = 0; i < _testHOG->GetImgNum(); i++) {
		_testHOG->setToIdentity(i);
	}
	//for (int i = 0; i < _testHOG->ManimgsVec.size(); i++)cv::imwrite((fs::path(_imgpath) / std::string("boundingBoxExperiment") / (std::to_string(i) + ".jpg")).string(), *(_testHOG->ManimgsVec[i]));

	_testHOG->HOGExtractor();
	_RF->load((fs::path(_imgpath) / configPtr->folders[2]).string());
	_RF->Prediction(_testHOG->featsImg);
	predVec.push_back(_RF->GetPrediction());
	predVec.push_back(_RF->GetBelief());
	
	
	return predVec;
}
void Classifier::MultiAugTrain() {
	_RF->createRF();
	for (int i = 0; i < configPtr->RandomForest.num_trees; i++) {
		imgsPreprocessing();
		_RF->train(i,_trainHOG->featsImg, _trainHOG->featslabel);
		
	}
	_RF->save((fs::path(_imgpath) / configPtr->folders[2]).string());

}

void Classifier::imgsPreprocessing() {
	_trainHOG->clearManVec();
	if (configPtr->loaded) {
		std::cout << "---------------train features Loading---------------" << std::endl;
		_trainHOG->HOGLoad((fs::path(_imgpath) / (configPtr->folders[0]+std::to_string(imgProcessingCount++))).string());
		
	}
	
	else {
		std::cout << "---------------Images Preprocessing---------------" << std::endl;
		if (configPtr->ImgConfig.Manipulation) {
			srand(time(NULL));
			std::vector<int> flipAxis{ -1,0,1 };
			
			if (!configPtr->ImgConfig.weighted) {
				configPtr->ImgConfig.classesCount.clear();
				for(int k=0; k<configPtr->RandomForest.categoriesNum;k++)
				{
					configPtr->ImgConfig.classesCount.push_back(1);

				}
			}
			
			for (int index = 0; index < _trainHOG->imgsVec.size(); index++) {
				int count = configPtr->ImgConfig.classesCount[int(_trainHOG->labels.at<float>(index))];
				for (int j = 0; j < configPtr->ImgConfig.NumManPerImg*count; j++) {
						int randMan = rand() % 2;
						if ((randMan == 0)) {
							//Rotation
							std::cout << "Image:" << index << " >>Rotated " << std::endl;
							float Rot = random_float(configPtr->ImgConfig.angle[0], configPtr->ImgConfig.angle[1]);
							float scale = random_float(configPtr->ImgConfig.scale[0], configPtr->ImgConfig.scale[1]);
							_trainHOG->Rotated(index, Rot, scale);
						}
						else {
							//Rotation And Flip
							std::cout << "Image:" << index << " >>RotatedAndFlip " << std::endl;
							float Rot = random_float(configPtr->ImgConfig.angle[0], configPtr->ImgConfig.angle[1]);
							float scale = random_float(configPtr->ImgConfig.scale[0], configPtr->ImgConfig.scale[1]);
							_trainHOG->RotatedAndFlip(index, Rot, scale, flipAxis[rand() % 3]);

						}
					}
				}
			}

		
		else {
			for (int i = 0; i < _trainHOG->GetImgNum(); i++) {
				_trainHOG->setToIdentity(i);
			}
		}
		//std::cout << _trainHOG->featslabel << std::endl;
		//for (int i = 0; i < _trainHOG->ManimgsVec.size(); i++)_trainHOG->visualizeImg(i, "M");
		/*for(int i =0;i<_trainHOG->ManimgsVec.size();i++){
			
			auto img = _trainHOG->ManimgsVec[i]->clone();
			cv::putText(img, std::to_string(int(_trainHOG->featslabel.at<float>(i))),
				cv::Point(128, 128), // Coordinates
				cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
				5, // Scale. 2.0 = 2x bigger
				cv::Scalar(255, 0, 255), // BGR Color
				1); // Anti-alias (Optional)
		cv::imwrite((fs::path(_imgpath) / std::string("boundingBoxExperiment") / (std::to_string(i) + ".jpg")).string(),img);
	}*/
		std::cout << "---------------train features Extraction---------------" << std::endl;
		_trainHOG->HOGExtractor((fs::path(_imgpath) / (configPtr->folders[0] + std::to_string(imgProcessingCount++))).string());
		

	}
	

}

float Classifier::random_float(float min, float max) {

	return ((float)rand() / RAND_MAX) * (max - min) + min;

}