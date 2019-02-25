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
  data_partition_.reset(new DataPartition(num_data_, num_leaves));  
  //TODO: Do we bother with the cache-size config?
  histogram_pool_.DynamicChangeSize(train_data_, config_, num_leaves, num_leaves);
  leaf_splits_.resize(num_leaves);
  for(int i = 0; i < num_leaves; ++i){
    leaf_splits_[i].reset(new LeafSplits(num_data_));
  }

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

  is_constant_hessian_ = is_constant_hessian;
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
    histogram_pool_.Get(i,&histogram_array);
    HistogramBinEntry* ptr_leaf_hist_data = histogram_array[0].RawData() - 1;
    train_data_->ConstructHistograms(is_feature_used,
    				     leaf_splits_[i]->data_indices(), leaf_splits_[i]->num_data_in_leaf(),
    				     leaf_splits_[i]->LeafIndex(),
    				     ordered_bins_, gradients, hessians,
    				     ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
    				     ptr_leaf_hist_data);
  }
}

void DecisionTableLearner::FindBestThresholdSequence(const int num_leaves, const double min_gain_shift, const std::vector<FeatureHistogram*>& histogram_arrs, const int feature_idx, FeatureSplits& output, const int dir, const bool skip_default_bin, const bool use_na_as_missing){
  std::vector<HistogramBinEntry*> leaf_histograms(num_leaves);
  bool is_splittable = false;
  for(int i = 0; i < num_leaves; i++){
    leaf_histograms[i] = histogram_arrs[i][feature_idx].RawData();
  }
  int8_t bias;
  if(train_data_->FeatureBinMapper(feature_idx)->GetDefaultBin() == 0){
    bias = 1;
  } else {
    bias = 0;
  }

  auto num_bin = train_data_->FeatureNumBin(feature_idx);
  auto default_bin = train_data_->FeatureBinMapper(feature_idx)->GetDefaultBin();
  //TODO: Vectorise all the gradient sums.
  std::vector<double> best_sum_left_gradient(num_leaves, NAN);
  std::vector<double> best_sum_left_hessian(num_leaves, NAN);
  double best_gain = kMinScore;
  std::vector<data_size_t> best_left_count(num_leaves,0);
  uint32_t best_threshold = static_cast<uint32_t>(num_bin);

  if (dir == -1) {
    std::vector<double> sum_right_gradient(num_leaves, 0.0f);
    std::vector<double> sum_right_hessian(num_leaves, kEpsilon);
    std::vector<data_size_t> right_count(num_leaves, 0);
    std::vector<double> sum_left_gradient(num_leaves);
    std::vector<double> sum_left_hessian(num_leaves);
    std::vector<data_size_t> left_count(num_leaves, 0);

    int t = num_bin - 1 - bias - use_na_as_missing;
    const int t_end = 1 - bias;

    // from right to left, and we don't need data in bin0
    for (; t >= t_end; --t) {
      // need to skip default bin
      if (skip_default_bin && (t + bias) == static_cast<int>(default_bin)) { continue; }

      for(int i = 0; i < num_leaves; ++i){
	sum_right_gradient[i] += leaf_histograms[i][t].sum_gradients;
	sum_right_hessian[i] += leaf_histograms[i][t].sum_hessians;
	right_count[i] += leaf_histograms[i][t].cnt;
      }

      //Check conditions for each leaf.
      bool should_skip = false, should_break = false;
      for(int i = 0; i < num_leaves; ++i){
	if(right_count[i] < config_->min_data_in_leaf
	   || sum_right_hessian[i] < config_->min_sum_hessian_in_leaf){
	  should_skip = true;
	  break;
	}
	left_count[i] = leaf_splits_[i]->num_data_in_leaf() - right_count[i];
	if(left_count[i] < config_->min_data_in_leaf){
	  should_break = true;
	  break;
	}
	sum_left_hessian[i] = leaf_splits_[i]->sum_hessians() - sum_right_hessian[i];
	if(sum_left_hessian[i] < config_->min_sum_hessian_in_leaf){
	  should_break = true;
	  break;
	}
      }
      if(should_skip){
	continue;
      }
      if(should_break){
	break;
      }

      double current_gain = 0.0;
      for(int i = 0; i < num_leaves; ++i){
	sum_left_gradient[i] = leaf_splits_[i]->sum_gradients() - sum_right_gradient[i];
	double this_leaf = FeatureHistogram::GetSplitGains(sum_left_gradient[i], sum_left_hessian[i], sum_right_gradient[i], sum_right_hessian[i],
							   config_->lambda_l1, config_->lambda_l2, config_->max_delta_step,
							   leaf_splits_[i]->min_constraint(), leaf_splits_[i]->max_constraint(), train_data_->FeatureMonotone(feature_idx));
	current_gain += this_leaf;
      }
      // gain with split is worse than without split
      if (current_gain <= min_gain_shift) continue;

      // mark to is splittable
      is_splittable = true;
      
      // better split point
      if (current_gain > best_gain) {
	for(int i = 0; i < num_leaves; ++i){
	  best_left_count[i] = left_count[i];
	  best_sum_left_gradient[i] = sum_left_gradient[i];
	  best_sum_left_hessian[i] = sum_left_hessian[i];
	}
	// left is <= threshold, right is > threshold.  so this is t-1
	best_threshold = static_cast<uint32_t>(t - 1 + bias);
	best_gain = current_gain;
      }
    }
  } else {
    std::vector<double> sum_left_gradient(num_leaves, 0.0f);
    std::vector<double> sum_left_hessian(num_leaves, kEpsilon);
    std::vector<data_size_t> left_count(num_leaves, 0);
    std::vector<double> sum_right_gradient(num_leaves);
    std::vector<double> sum_right_hessian(num_leaves);
    std::vector<data_size_t> right_count(num_leaves);    

    int t = 0;
    const int t_end = num_bin - 2 - bias;

    if (use_na_as_missing && bias == 1) {
      for(int j = 0; j < num_leaves; ++j){
	sum_left_gradient[j] = leaf_splits_[j]->sum_gradients();
	sum_left_hessian[j] = leaf_splits_[j]->sum_hessians() - kEpsilon;
	left_count[j] = leaf_splits_[j]->num_data_in_leaf();
	auto data = leaf_histograms[j];
	for (int i = 0; i < num_bin - bias; ++i) {
	  sum_left_gradient[j] -= data[i].sum_gradients;
	  sum_left_hessian[j] -= data[i].sum_hessians;
	  left_count[j] -= data[i].cnt;
	}
      }
      t = -1;
    }

    for (; t <= t_end; ++t) {
      // need to skip default bin
      if (skip_default_bin && (t + bias) == static_cast<int>(default_bin)) { continue; }
      if (t >= 0) {
	for(int i = 0; i < num_leaves; i++){
	  sum_left_gradient[i] += leaf_histograms[i][t].sum_gradients;
	  sum_left_hessian[i] += leaf_histograms[i][t].sum_hessians;
	  left_count[i] += leaf_histograms[i][t].cnt;
	}
      }
      bool should_skip = false, should_break = false;

      for(int i = 0; i < num_leaves; ++i){
	// if data not enough, or sum hessian too small
        if (left_count[i] < config_->min_data_in_leaf
            || sum_left_hessian[i] < config_->min_sum_hessian_in_leaf){
	  should_skip = true;
	  break;
	}

	right_count[i] = leaf_splits_[i]->num_data_in_leaf() - left_count[i];
	// if data not enough
	if (right_count[i] < config_->min_data_in_leaf){
	  should_break = true;
	  break;
	}

	sum_right_hessian[i] = leaf_splits_[i]->sum_hessians() - sum_left_hessian[i];
	// if sum hessian too small
	if (sum_right_hessian[i] < config_->min_sum_hessian_in_leaf){
	  should_break = true;
	  break;
	}
	
      }
      if(should_skip){
	continue;
      }
      if(should_break){
	break;
      }

      double current_gain = 0.0;
      for(int i = 0; i < num_leaves; ++i){
	sum_right_gradient[i] = leaf_splits_[i]->sum_gradients() - sum_left_gradient[i];
        // current split gain
        double this_leaf = FeatureHistogram::GetSplitGains(sum_left_gradient[i], sum_left_hessian[i], sum_right_gradient[i], sum_right_hessian[i],
							   config_->lambda_l1, config_->lambda_l2, config_->max_delta_step,
							   leaf_splits_[i]->min_constraint(), leaf_splits_[i]->max_constraint(), train_data_->FeatureMonotone(feature_idx));
	current_gain += this_leaf;
      }

      // gain with split is worse than without split
      if (current_gain <= min_gain_shift) continue;

      // mark to is splittable
      is_splittable = true;
      // better split point
      if (current_gain > best_gain) {
	for(int i = 0; i < num_leaves; ++i){
	  best_left_count[i] = left_count[i];
	  best_sum_left_gradient[i] = sum_left_gradient[i];
	  best_sum_left_hessian[i] = sum_left_hessian[i];
	}
	best_threshold = static_cast<uint32_t>(t + bias);
	best_gain = current_gain;
      }
    }
  }
  if (is_splittable && best_gain > output.gain) {
    for(int i = 0; i < num_leaves; ++i){
      // update split information
      output.leaf_splits[i].threshold = best_threshold;
      output.leaf_splits[i].left_output = FeatureHistogram::CalculateSplittedLeafOutput(best_sum_left_gradient[i], best_sum_left_hessian[i],
											config_->lambda_l1, config_->lambda_l2, config_->max_delta_step,
											leaf_splits_[i]->min_constraint(), leaf_splits_[i]->max_constraint());
      output.leaf_splits[i].left_count = best_left_count[i];
      output.leaf_splits[i].left_sum_gradient = best_sum_left_gradient[i];
      output.leaf_splits[i].left_sum_hessian = best_sum_left_hessian[i] - kEpsilon;
      output.leaf_splits[i].right_output = FeatureHistogram::CalculateSplittedLeafOutput(leaf_splits_[i]->sum_gradients() - best_sum_left_gradient[i],
											 leaf_splits_[i]->sum_hessians() - best_sum_left_hessian[i],
											 config_->lambda_l1, config_->lambda_l2, config_->max_delta_step,
											 leaf_splits_[i]->min_constraint(), leaf_splits_[i]->max_constraint());
      output.leaf_splits[i].right_count = leaf_splits_[i]->num_data_in_leaf() - best_left_count[i];
      output.leaf_splits[i].right_sum_gradient = leaf_splits_[i]->sum_gradients() - best_sum_left_gradient[i];
      output.leaf_splits[i].right_sum_hessian = leaf_splits_[i]->sum_hessians() - best_sum_left_hessian[i] - kEpsilon;
      output.leaf_splits[i].gain = 0.0; //TODO: Fill in.
      output.leaf_splits[i].default_left = dir == -1;
    }
    output.gain = best_gain;
  }
}

FeatureSplits DecisionTableLearner::FindBestFeatureSplitNumerical(const int num_leaves, const double min_gain_shift, const std::vector<FeatureHistogram*>& histogram_arrs, const int feature_idx){
  auto num_bin = train_data_->FeatureNumBin(feature_idx);
  auto missing_type = train_data_->FeatureBinMapper(feature_idx)->missing_type();
  FeatureSplits output(num_leaves);
  if(num_bin > 2 && missing_type != MissingType::None){
    if(missing_type == MissingType::Zero){
      FindBestThresholdSequence(num_leaves, min_gain_shift, histogram_arrs, feature_idx, output, -1, true, false);
      FindBestThresholdSequence(num_leaves, min_gain_shift, histogram_arrs, feature_idx, output, 1, true, false);
    } else {
      FindBestThresholdSequence(num_leaves, min_gain_shift, histogram_arrs, feature_idx, output, -1, false, true);
      FindBestThresholdSequence(num_leaves, min_gain_shift, histogram_arrs, feature_idx, output, 1, false, true);
    }
  } else {
    FindBestThresholdSequence(num_leaves, min_gain_shift, histogram_arrs, feature_idx, output, -1, false, false);
    // fix the direction error when only have 2 bins
    if (missing_type == MissingType::NaN) {
      for(auto& split : output.leaf_splits){
	split.default_left = false;
      }
    }    
  }
  output.gain -= min_gain_shift;
  int real_fidx = train_data_->RealFeatureIndex(feature_idx);
  for(int i = 0; i < num_leaves; ++i){
    output.leaf_splits[i].feature = real_fidx;
  }
  return output;
}

FeatureSplits DecisionTableLearner::FindBestFeatureSplit(const int num_leaves, const double min_gain_shift, const std::vector<FeatureHistogram*>& histogram_arrs, const int feature_idx){
  if(train_data_->FeatureBinMapper(feature_idx)->bin_type() == BinType::NumericalBin){
    return FindBestFeatureSplitNumerical(num_leaves, min_gain_shift, histogram_arrs, feature_idx);
  } else {
    throw std::runtime_error("Decision table learner does not currently support categorical features.");
  }
}

FeatureSplits DecisionTableLearner::FindBestSplit(const std::vector<int8_t>& is_feature_used, const int num_leaves, const score_t* gradients, const score_t* hessians){
  FeatureSplits ret(num_leaves);
  std::vector<FeatureHistogram*> histogram_arrs(num_leaves);
  for(int leaf_idx = 0; leaf_idx < num_leaves; ++leaf_idx){
    histogram_pool_.Get(leaf_idx, &histogram_arrs[leaf_idx]);
  }  
  for(int feature_idx = 0; feature_idx < train_data_->num_features(); ++feature_idx){
    if(is_feature_used[feature_idx]){
      double min_shift_gain = config_->min_gain_to_split * num_leaves;
      for(int leaf_idx = 0; leaf_idx < num_leaves; ++leaf_idx){
	//Step 0: Fix histogram (TODO: This is a copy-pasta from the tree learner)
	//TODO: I vaguely recall this being related to feature bundling, is that the case???
	train_data_->FixHistogram(feature_idx,
				  leaf_splits_[leaf_idx]->sum_gradients(), leaf_splits_[leaf_idx]->sum_hessians(),
				  leaf_splits_[leaf_idx]->num_data_in_leaf(),
				  histogram_arrs[leaf_idx][feature_idx].RawData());
	//Step 1: Compute minimum gain to overcome.
	double shift_gain =
	  histogram_arrs[leaf_idx][feature_idx].GetLeafSplitGain(leaf_splits_[leaf_idx]->sum_gradients(),
								 leaf_splits_[leaf_idx]->sum_hessians(),
								 config_->lambda_l1,
								 config_->lambda_l2,
								 config_->max_delta_step);
	min_shift_gain += shift_gain;
      }
      //Find best split for this feature.
      auto split = FindBestFeatureSplit(num_leaves, min_shift_gain, histogram_arrs, feature_idx);
      if(split.gain > ret.gain){
	ret = split;
      }
    }
  }
  return ret;
}

void DecisionTableLearner::Split(Tree* tree, const FeatureSplits& split, const score_t* gradients, const score_t* hessians){
  const int inner_feature_index = train_data_->InnerFeatureIndex(split.leaf_splits[0].feature);
  bool is_numerical_split = train_data_->FeatureBinMapper(inner_feature_index)->bin_type() == BinType::NumericalBin;
  if(is_numerical_split){
    for(int left_leaf = 0; left_leaf < split.leaf_splits.size(); left_leaf++){
      std::cout << "DEBUG: Splitting leaf " << left_leaf << " left count : " << split.leaf_splits[left_leaf].left_count << " , right count : " << split.leaf_splits[left_leaf].right_count << std::endl;
      auto threshold_double = train_data_->RealThreshold(inner_feature_index, split.leaf_splits[left_leaf].threshold);
      // split tree, will return right leaf
      auto right_leaf = tree->Split(left_leaf,
				    inner_feature_index,
				    split.leaf_splits[left_leaf].feature,
				    split.leaf_splits[left_leaf].threshold,
				    threshold_double,
				    static_cast<double>(split.leaf_splits[left_leaf].left_output),
				    static_cast<double>(split.leaf_splits[left_leaf].right_output),
				    static_cast<data_size_t>(split.leaf_splits[left_leaf].left_count),
				    static_cast<data_size_t>(split.leaf_splits[left_leaf].right_count),
				    static_cast<float>(split.leaf_splits[left_leaf].gain),
				    train_data_->FeatureBinMapper(inner_feature_index)->missing_type(),
				    split.leaf_splits[left_leaf].default_left);
      data_partition_->Split(left_leaf, train_data_, inner_feature_index,
			     &split.leaf_splits[left_leaf].threshold, 1, split.leaf_splits[left_leaf].default_left, right_leaf);
      leaf_splits_[left_leaf]->Init(left_leaf, data_partition_.get(), gradients, hessians);
      leaf_splits_[right_leaf]->Init(right_leaf, data_partition_.get(), gradients, hessians);
    }
  } else {
    throw std::runtime_error("Decision table learner does not support categorical splits yet.");
  }
}

Tree* DecisionTableLearner::Train(const score_t* gradients, const score_t *hessians, bool is_constant_hessian, Json& forced_split_json) {
  auto num_leaves = 1 << tree_depth_;
  auto tree = std::unique_ptr<Tree>(new Tree(num_leaves));
  //Puts all data in the leaf.
  data_partition_->Init();
  leaf_splits_[0]->Init(0, data_partition_.get(), gradients, hessians);
  //TODO: Feature bagging etc.
  std::vector<int8_t> is_feature_used(train_data_->num_features(), true);
  for(int i = 0; i < tree_depth_ - 1; ++i){
    ConstructHistograms(is_feature_used, 1 << i, gradients, hessians);
    auto split = FindBestSplit(is_feature_used, 1 << i, gradients, hessians);
    if(split.gain <= 0.0){
      Log::Warning("No further splits with positive gain, best gain: %f", split.gain);
      break;
    }
    Split(tree.get(), split, gradients, hessians);
    std::cout << "DEBUG: Iter " << i << " split gain " << split.gain << " feature " << split.leaf_splits[0].feature << " threshold " << split.leaf_splits[0].threshold << std::endl;
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
