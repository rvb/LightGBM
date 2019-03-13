#ifndef LIGHTGBM_TREELEARNER_DECISION_TABLE_LEARNER_H_
#define LIGHTGBM_TREELEARNER_DECISION_TABLE_LEARNER_H_
#include <LightGBM/tree_learner.h>

#include <LightGBM/utils/random.h>
#include <LightGBM/utils/array_args.h>

#include <LightGBM/dataset.h>
#include <LightGBM/tree.h>

#include "feature_histogram.hpp"
#include "split_info.hpp"
#include "data_partition.hpp"
#include "leaf_splits.hpp"

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <memory>
#ifdef USE_GPU
// Use 4KBytes aligned allocator for ordered gradients and ordered hessians when GPU is enabled.
// This is necessary to pin the two arrays in memory and make transferring faster.
#include <boost/align/aligned_allocator.hpp>
#endif

using namespace json11;

namespace LightGBM {

/*!
* \brief Used for learning a tree by single machine
*/
class DecisionTableLearner: public TreeLearner {
 public:
  explicit DecisionTableLearner(const Config* config);

  ~DecisionTableLearner();

  void Init(const Dataset* train_data, bool is_constant_hessian) override;

  void ResetTrainingData(const Dataset* train_data) override;

  void ResetConfig(const Config* config) override;

  Tree* Train(const score_t* gradients, const score_t *hessians, bool is_constant_hessian,
              Json& forced_split_json) override;

  Tree* FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t* hessians) const override;

  Tree* FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                          const score_t* gradients, const score_t* hessians) override;

  void SetBaggingData(const data_size_t* used_indices, data_size_t num_data) override {
    data_partition_->SetUsedDataIndices(used_indices, num_data);
  }

  void AddPredictionToScore(const Tree* tree, double* out_score) const override {
    if (tree->num_leaves() <= 1) { return; }
    CHECK(tree->num_leaves() <= data_partition_->num_leaves());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < tree->num_leaves(); ++i) {
      double output = static_cast<double>(tree->LeafOutput(i));
      data_size_t cnt_leaf_data = 0;
      auto tmp_idx = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
      for (data_size_t j = 0; j < cnt_leaf_data; ++j) {
        out_score[tmp_idx[j]] += output;
      }
    }
  };

  void RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, const double* prediction,
                       data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt) const override;

  void RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, double prediction,
                       data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt) const override;

 protected:
  void ConstructHistogram(const std::vector<int8_t>& is_feature_used, const score_t* gradients, const score_t* hessians, int leaf_idx);  
  FeatureSplits FindBestSplit(const std::vector<int8_t>& is_feature_used, const int num_leaves);
  FeatureSplits FindBestFeatureSplit(const int num_leaves, const double min_gain_shift, const std::vector<double>& gain_shifts, const std::vector<FeatureHistogram*>& histogram_arrs, const int feature_idx);
  FeatureSplits FindBestFeatureSplitNumerical(const int num_leaves, const double min_gain_shift, const std::vector<FeatureHistogram*>& histogram_arrs, const int feature_idx);
  FeatureSplits FindBestFeatureSplitCategorical(const int num_leaves, const double min_gain_shift, const std::vector<FeatureHistogram*>& histogram_arrs, const int feature_idx);
  void FindBestThresholdSequence(const int num_leaves, const double min_gain_shift, const std::vector<FeatureHistogram*>& histogram_arrs, const int feature_idx, FeatureSplits& output, const int dir, const bool skip_default_bin, const bool use_na_as_missing);
  void Split(const std::vector<int8_t>& is_feature_used, Tree* tree, const FeatureSplits& split, const score_t* gradients, const score_t* hessians);
  void PerformSplit(const int left_leaf, const int right_leaf, const score_t* gradients, const score_t* hessians, const std::vector<int8_t>& is_feature_used);
  int32_t ForceSplits(Tree* tree, Json& forced_split_json, int32_t* cur_depth, const score_t* gradients, const score_t* hessians);
  void SampleFeatures(std::vector<int8_t>& is_feature_used);
  void InitOrderedBin();

  const Dataset* train_data_;
  HistogramPool histogram_pool_;
  const Config* config_;
  std::unique_ptr<DataPartition> data_partition_;
  std::vector<std::unique_ptr<LeafSplits>> leaf_splits_;
  data_size_t num_data_;
  int tree_depth_;

  //Copied pasta.
  std::vector<std::unique_ptr<OrderedBin>> ordered_bins_;
  bool has_ordered_bin_ = false;
  std::vector<int> ordered_bin_indices_;
  /*! \brief gradients of current iteration, ordered for cache optimized */
  std::vector<score_t> ordered_gradients_;
  /*! \brief hessians of current iteration, ordered for cache optimized */
  std::vector<score_t> ordered_hessians_;
  /*! \brief  is_data_in_leaf_[i] != 0 means i-th data is marked */
  std::vector<char> is_data_in_leaf_;
  /*! \brief used for generate used features */
  Random random_;  

  bool is_constant_hessian_;

  std::vector<bool> feature_used_;
  std::vector<int> valid_feature_indices_;
};
}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_DECISION_TABLE_LEARNER_H_
