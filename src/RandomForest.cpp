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
    
  
    for (int t = 0; t < _num_trees; ++t) {
        std::cout << "Train tree no:" << t << std::endl;
        cv::Mat accVal;
        //cv::Mat test;
        cv::Ptr<cv::ml::TrainData> bootstrap = cv::ml::TrainData::create(featsImg, cv::ml::ROW_SAMPLE, featslabel);
        bootstrap->setTrainTestSplitRatio(_ratio, true);
        //std::cout << bootstrap->getTrainSampleIdx().size() << std::endl;
        //cv::transpose(bootstrap->getTestResponses(), test);
        //std::cout << test << std::endl;
        //bootstrap->shuffleTrainTest();
        _forest[t]->train(bootstrap);
        _forest[t]->predict(bootstrap->getTestSamples(), accVal);
        //cv::transpose(accVal , test);
        //std::cout << test << std::endl;
        validate(bootstrap->getTestResponses(), accVal);
        accVal.release();
    }

}
void RandomForest::train(int index, cv::Mat& featsImg, cv::Mat& featslabel) {
        std::cout << "Train tree no:" << index << std::endl;
        cv::Mat accVal;
        cv::Ptr<cv::ml::TrainData> bootstrap = cv::ml::TrainData::create(featsImg, cv::ml::ROW_SAMPLE, featslabel);
        bootstrap->setTrainTestSplitRatio(_ratio, true);
        _forest[index]->train(bootstrap);
        _forest[index]->predict(bootstrap->getTestSamples(), accVal);
        validate(bootstrap->getTestResponses(), accVal);

}



void RandomForest::Prediction(const cv::Mat& featsImg) {
    predM.release();
    pred.release();
    belief.release();
    cv::Mat f;
    for (int i = 0; i < _num_trees; i++) {
        _forest[i]->predict(featsImg, f);
        predM.push_back(f.reshape(1, 1));
    }
    predM.convertTo(predM, 4);
    
    for (int i = 0; i < predM.cols; i++) {
        f = predM.col(i);
        //std::cout << f << std::endl;
       
        auto vec = MajorityVote(f);
        pred.push_back(vec[0]);
        belief.push_back(vec[1]);
    }  
}

cv::Mat RandomForest::GetPrediction() {
    cv::Mat predMat = pred.clone();
    predMat.convertTo(predMat, 4);
    return predMat;
}
cv::Mat RandomForest::GetBelief() {
    cv::Mat BeliefMat = belief.clone();
    return BeliefMat;
}
std::vector<float> RandomForest::MajorityVote(cv::Mat& predCol) {
     int size = predCol.rows;
     cv::Mat count = cv::Mat::zeros(cv::Size(1, _max_categories), 4);
          for (int i = 0; i < size; i++) {
              count.at<int>(predCol.at<int>(i)) += 1;

          }
          double min, max;
          cv::Point min_loc, max_loc;
          
          minMaxLoc(count, &min, &max, &min_loc, &max_loc);
          std::vector<float> vec{ float(max_loc.y),float(max / size) };
          return vec ;
      }
    
void RandomForest::accuracy(cv::Mat& featslabel) {
    //std::cout << pred << std::endl;
    //std::cout << featslabel << std::endl;

          acc = cv::sum((pred == featslabel))[0] / 255.0;
          acc /= featslabel.rows;
          acc *= 100;
          std::cout << "Total Prediction Accuracy= " << acc << "%" << std::endl;
          std::cout << "<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>" << std::endl;
          accuracyforDT(featslabel);
}

void RandomForest::validate(cv::Mat& featslabel, cv::Mat& val) {
    //std::cout << pred << std::endl;
    //std::cout << featslabel << std::endl;
    //std::cout << val << std::endl;
    float valAcc;
    val = val.reshape(1, 1);
    val.convertTo(val, 4);
    val.convertTo(val, CV_32F);
    cv::transpose(val, val);
    //std::cout << val << std::endl;
    //std::cout << featslabel<< std::endl;

    valAcc = cv::sum((val == featslabel))[0] / 255.0;
    valAcc /= featslabel.rows;
    valAcc *= 100;
    std::cout << "validation Prediction Accuracy= " << valAcc << "%" << std::endl;
    std::cout << "<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>" << std::endl;
    
}

void RandomForest::accuracyforDT(cv::Mat& featslabel) {
    //std::cout << pred << std::endl;
    //std::cout << featslabel << std::endl;
    cv::Mat predDT;
    float f;
    for (int i = 0; i < predM.rows; i++) {
        predDT = predM.row(i);
        std::cout << predDT << std::endl;
        cv::transpose(predDT, predDT);
        predDT.convertTo(predDT, CV_32F);
        f = cv::sum((predDT== featslabel))[0] / 255.0;
        f /= featslabel.rows;
        f *= 100;
        std::cout << "test Accuracy for tree no."<<i<<"  = " << f << "%" << std::endl;
        accVec.push_back(f);
    }
    
}