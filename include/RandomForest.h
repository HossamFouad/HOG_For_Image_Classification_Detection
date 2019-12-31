#pragma once
#include <opencv2/opencv.hpp>
class RandomForest {
public:
    RandomForest(float,int, int, int, int, int);
    void createRF();
    void train(cv::Mat&, cv::Mat&);
    void RandomForest::train(int, cv::Mat&, cv::Mat&);
    void Prediction(const cv::Mat&);
    void accuracy(cv::Mat&);
    void save(const std::string&); 
    void load(const std::string&);
    void accuracyforDT(cv::Mat& featslabel);
    cv::Mat GetPrediction();
    cv::Mat GetBelief();
protected:
   cv::Mat pred,belief, predM;
    cv::Mat float_labels;
    float acc;
    std::vector<float> accVec;

private:
    void validate(cv::Mat&, cv::Mat&);
    std::vector<float> MajorityVote(cv::Mat&);
    float _ratio;
    int _num_trees;
    int _cv_folds;
    int _max_categories;
    int _max_depth;
    int _min_sample_count;
    std::vector< cv::Ptr<cv::ml::DTrees> > _forest;
};
