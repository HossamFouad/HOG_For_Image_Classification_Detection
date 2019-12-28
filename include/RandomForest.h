#pragma once
#include <opencv2/opencv.hpp>
class RandomForest {
public:
    RandomForest(float,int, int, int, int, int);
    void createRF();
    void train(cv::Mat&, cv::Mat&);
    void Prediction(cv::Mat&);
    void accuracy(cv::Mat&);
    void save(const std::string&); 
    void load(const std::string&);

protected:
   cv::Mat pred, predM;
    cv::Mat float_labels;
    float acc;
private:
    int MajorityVote(cv::Mat&);
    float _ratio;
    int _num_trees;
    int _cv_folds;
    int _max_categories;
    int _max_depth;
    int _min_sample_count;
    std::vector< cv::Ptr<cv::ml::DTrees> > _forest;
};
