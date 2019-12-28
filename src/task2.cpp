#include "task2.h"
#include <algorithm>
#include <random>

namespace fs = std::filesystem;

Classifier::Classifier(std::string path, float randomSampleRatio, int num_trees, int cv_folds, int max_depth, int min_sample_count,
	 const std::vector<std::string>& folder, cv::Size imageSize, cv::Size WinSize, cv::Size BlockSize, cv::Size BlockStride, cv::Size CellSize, int Bins):
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
	_RF(nullptr)

{}

void Classifier::loadTrainImgs()
{
	std::vector<std::string> f = loadFolders(folders[0]);
	max_categories = f.size();
	_trainHOG = std::make_unique<HOG>(imgSize,w,bs,bstride,cz,nb);
	for (const auto& folder : f) {
		_trainHOG->imgList((fs::path(_imgpath) / folders[0]/folder).string() ,static_cast<float>(folder[folder.length()-1]) - 48.0);
	}
	_trainHOG->loadImgs();
	std::cout << "TrainMaxSize = " << _trainHOG->xmax << " X " << _trainHOG->ymax << std::endl;
}


void Classifier::loadTestImgs()
{
	std::vector<std::string> f = loadFolders(folders[1]);
	max_categories = f.size();
	_testHOG = std::make_unique<HOG>(imgSize, w, bs, bstride, cz, nb);
	for (const auto& folder : f) {
		_testHOG->imgList((fs::path(_imgpath) / folders[1] / folder).string(), static_cast<float>(folder[folder.length() - 1]) - 48.0);
	}
	_testHOG->loadImgs();
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
	_RF = std::make_unique<RandomForest>(_randomSampleRatio, _num_trees, _cv_folds, max_categories, _max_depth, _min_sample_count);
	_RF->createRF();
	std::cout << "---------------Training---------------" << std::endl;
	_RF->train(_trainHOG->featsImg,_trainHOG->featslabel);
	_RF->save((fs::path(_imgpath)/folders[2]).string());
}
void Classifier::testRandomForest(bool trained,bool loaded) {
	_testHOG->clearManVec();
	if (loaded) {
		std::cout << "---------------test features Loading---------------" << std::endl;
		_testHOG->HOGLoad((fs::path(_imgpath) / folders[1]).string());
	}
	else {
		for (int i = 0; i < _testHOG->GetImgNum(); i++) {
			_testHOG->setToIdentity(i);
		}
		std::cout << "---------------test features Extraction---------------" << std::endl;
		_testHOG->HOGExtractor((fs::path(_imgpath) / folders[1]).string());
	}
	
	if (trained) {
		_RF = std::make_unique<RandomForest>(_randomSampleRatio, _num_trees, _cv_folds, max_categories, _max_depth, _min_sample_count);
		_RF->load((fs::path(_imgpath) / folders[2]).string());
	}
	std::cout << "---------------Prediction---------------" << std::endl;
	_RF->Prediction(_testHOG->featsImg);
	std::cout << "---------------Accuracy---------------" << std::endl;
	_RF->accuracy(_testHOG->featslabel);
}

void Classifier::imgsPreprocessing(bool M,bool loaded, int NumManPerImg) {
	_trainHOG->clearManVec();
	if (loaded) {
		std::cout << "---------------train features Loading---------------" << std::endl;
		_trainHOG->HOGLoad((fs::path(_imgpath) / folders[0]).string());
		
	}
	
	else {
		std::cout << "---------------Images Preprocessing---------------" << std::endl;
		if (M) {
			srand(time(NULL));
			std::vector<int> num;
			std::vector<int> flipAxis{ -1,0,1 };
			for (int i = 0; i < _trainHOG->GetImgNum(); i++)num.push_back(i);
			auto rng = std::default_random_engine{};
			for (int j = 0; j < NumManPerImg; j++) {
				std::shuffle(std::begin(num), std::end(num), rng);
				for (auto& index : num)
				{
					int randMan = rand() % 3;
					if ((randMan == 0)) {
						//Rotation
						std::cout << "Image:" << index << " >>Rotated " << std::endl;
						float Rot = random_float(-180.0f, 180.0f);
						float scale = random_float(-0.9f, 2.0f);
						_trainHOG->Rotated(index, Rot, scale);
					}
					else if (randMan == 1) {
						//Flip
						std::cout << "Image:" << index << " >>Flip" << std::endl;
						_trainHOG->Flip(index, flipAxis[rand() % 3]);
					}
					else {
						//Rotation And Flip
						std::cout << "Image:" << index << " >>RotatedAndFlip " << std::endl;
						float Rot = random_float(-180.0f, 180.0f);
						float scale = random_float(-0.9f, 2.0f);
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
		
		std::cout << "---------------train features Extraction---------------" << std::endl;
		_trainHOG->HOGExtractor((fs::path(_imgpath) / folders[0]).string());
		

	}
	
}

float Classifier::random_float(float min, float max) {

	return ((float)rand() / RAND_MAX) * (max - min) + min;

}