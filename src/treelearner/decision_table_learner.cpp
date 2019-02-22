#include "decision_table_learner.h"

#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>

#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/common.h>

#include <algorithm>
#include <vector>
#include <queue>

namespace LightGBM {

DecisionTableLearner::DecisionTableLearner(const Config* config){
  //TODO: Make this a config parameter
  tree_depth_ = 5;
  config_ = config;
}

DecisionTableLearner::~DecisionTableLearner() {
}

void DecisionTableLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  auto num_leaves = 1 << tree_depth_;
  //TODO: Do we bother with the cache-size config?
  histogram_pool_.DynamicChangeSize(train_data_, config_, num_leaves, num_leaves);
  leaf_splits_.reset(new LeafSplits(num_data_));

  //TODO: Copypasta is fun, get on it!
  // get ordered bin
  train_data_->CreateOrderedBins(&ordered_bins_);

  // check existing for ordered bin
  for (int i = 0; i < static_cast<int>(ordered_bins_.size()); ++i) {
    if (ordered_bins_[i] != nullptr) {
      has_ordered_bin_ = true;
      break;
    }
  }
  // initialize ordered gradients and hessians
  ordered_gradients_.resize(num_data_);
  ordered_hessians_.resize(num_data_);
  //TODO: ordered_bin_indices copy-pasta.
}

void DecisionTableLearner::ResetTrainingData(const Dataset* train_data) {
  throw std::runtime_error("Resetting training data is not implemented yet for decision tables.");
}

void DecisionTableLearner::ResetConfig(const Config* config) {
  throw std::runtime_error("Resetting config is not implemented yet for decision tables.");
}

void DecisionTableLearner::ConstructHistograms(const std::vector<int8_t>& is_feature_used, const int num_leaves, const score_t* gradients, const score_t* hessians){
  for(int i = 0; i < num_leaves; ++i){
    FeatureHistogram* histogram_array;
    auto whatIsThisAnyway = histogram_pool_.Get(i,&histogram_array);
    HistogramBinEntry* ptr_leaf_hist_data = histogram_array[0].RawData() - 1;

    //TODO: Get enough shit copied in here to make this compile.
    // train_data_->ConstructHistograms(is_feature_used,
    // 				     smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
    // 				     smaller_leaf_splits_->LeafIndex(),
    // 				     ordered_bins_, gradients_, hessians_,
    // 				     ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
    // 				     ptr_smaller_leaf_hist_data);    
  }
}

Tree* DecisionTableLearner::Train(const score_t* gradients, const score_t *hessians, bool is_constant_hessian, Json& forced_split_json) {
  auto num_leaves = 1 << tree_depth_;
  auto tree = std::unique_ptr<Tree>(new Tree(num_leaves));
  leaf_splits_->Init(gradients, hessians);  
  for(int i = 0; i < tree_depth_ - 1; ++i){
    std::cout << "DEBUG SHOULD DO ITER " << i << std::endl;
  }
  return tree.release();
}

Tree* DecisionTableLearner::FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t *hessians) const {
  return nullptr;
}

Tree* DecisionTableLearner::FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred, const score_t* gradients, const score_t *hessians) {
  return nullptr;
}

void DecisionTableLearner::RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, const double* prediction,
                                        data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt) const {
}

void DecisionTableLearner::RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, double prediction,
  data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt) const {
}

}  // namespace LightGBM
