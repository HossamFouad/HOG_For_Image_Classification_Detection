#include "RandomForest.h"
#include <string>
#include <filesystem>
#include<opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cstdio>
namespace fs = std::filesystem;


RandomForest::RandomForest(float randomSampleRatio, int num_trees, int cv_folds, int max_categories, int max_depth, int min_sample_count):
    _ratio(randomSampleRatio),
    _num_trees(num_trees),
_cv_folds (cv_folds),
_max_categories (max_categories),
_max_depth (max_depth),
_min_sample_count (min_sample_count)
{

}

void RandomForest::createRF() {
        for (int t = 0; t < _num_trees; ++t) {
            cv::Ptr<cv::ml::DTrees> tree = cv::ml::DTrees::create();
            tree->setCVFolds(_cv_folds);
            tree->setMaxCategories(_max_categories);
            tree->setMaxDepth(_max_depth);
            tree->setMinSampleCount(_min_sample_count);
            _forest.push_back(tree);
            
        }
    }

void RandomForest::save(const std::string& p){
    std::vector<cv::String> fn;
    cv::glob(p, fn, false);
    for (auto& treeName : fn) {
        std::remove(treeName.c_str());
    }
    
    for (int i = 0; i < _forest.size();i++) {
        _forest[i]->save((fs::path(p) / std::to_string(i)).string());
}
}

void RandomForest::load(const std::string& p) {
    std::vector<cv::String> fn;
    cv::glob(p, fn, false);
    for (auto& treeName:fn) {
        cv::Ptr<cv::ml::DTrees> tree = cv::ml::DTrees::load(treeName);
        _forest.push_back(tree);
    }
}
void RandomForest::train(cv::Mat& featsImg, cv::Mat& featslabel) {
    cv::Ptr<cv::ml::TrainData> bootstrap = cv::ml::TrainData::create(featsImg, cv::ml::ROW_SAMPLE, featslabel);
    for (int t = 0; t < _num_trees; ++t) {
        std::cout << "Train tree no:" << t << std::endl;
        bootstrap->setTrainTestSplitRatio(_ratio, true);
        bootstrap->shuffleTrainTest();
        _forest[t]->train(bootstrap);
    }

}
     
void RandomForest::Prediction(cv::Mat& featsImg) {
    cv::Mat f;
    for (int i = 0; i < _num_trees; i++) {
        _forest[i]->predict(featsImg, f);
        predM.push_back(f.reshape(1, 1));
    }
    predM.convertTo(predM, 4);
    std::cout << predM << std::endl;
    for (int i = 0; i < predM.cols; i++) {
        f = predM.col(i);
        pred.push_back(float(MajorityVote(f)));

    }
   


}
int RandomForest::MajorityVote(cv::Mat& predCol) {
     int size = predCol.rows;
     cv::Mat count = cv::Mat::zeros(cv::Size(1, _max_categories), 4);
          for (int i = 0; i < size; i++) {
              count.at<int>(predCol.at<int>(i)) += 1;

          }
          double min, max;
          cv::Point min_loc, max_loc;
          minMaxLoc(count, &min, &max, &min_loc, &max_loc);
          return max_loc.y;
      }
    
void RandomForest::accuracy(cv::Mat& featslabel) {
    std::cout << pred << std::endl;
    std::cout << featslabel << std::endl;

          acc = cv::sum((pred == featslabel))[0] / 255.0;
          acc /= featslabel.rows;
          acc *= 100;
          std::cout << "Prediction Accuracy= " << acc << "%" << std::endl;
      }