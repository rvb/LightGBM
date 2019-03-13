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
  config_ = config;
  random_ = Random(config_->feature_fraction_seed);  
  if(config_->max_depth > 0){
    tree_depth_ = config_->max_depth;
  } else {
    Log::Warning("Max depth is not set, defaulting to 5.");
    tree_depth_ = 5;
  }
}

DecisionTableLearner::~DecisionTableLearner() {
}

void DecisionTableLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  auto num_leaves = 1 << tree_depth_;
  data_partition_.reset(new DataPartition(num_data_, num_leaves));  
  histogram_pool_.DynamicChangeSize(train_data_, config_, num_leaves, num_leaves);
  leaf_splits_.resize(num_leaves);
  for(int i = 0; i < num_leaves; ++i){
    leaf_splits_[i].reset(new LeafSplits(num_data_));
  }

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

  //Initialise ordered_bin_indices, pointing to the indices of each feature that has
  //ordered bins.
  if (has_ordered_bin_) {
    is_data_in_leaf_.resize(num_data_);
    std::fill(is_data_in_leaf_.begin(), is_data_in_leaf_.end(), static_cast<char>(0));
    ordered_bin_indices_.clear();
    for (int i = 0; i < static_cast<int>(ordered_bins_.size()); i++) {
      if (ordered_bins_[i] != nullptr) {
        ordered_bin_indices_.push_back(i);
      }
    }
  }

  is_constant_hessian_ = is_constant_hessian;
  feature_used_.resize(train_data_->num_features(), false);
  valid_feature_indices_ = train_data_->ValidFeatureIndices();
}

void DecisionTableLearner::ResetTrainingData(const Dataset* train_data) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  
  // get ordered bin
  train_data_->CreateOrderedBins(&ordered_bins_);

  for(int i = 0; i < (1 << tree_depth_); ++i){
    leaf_splits_[i]->ResetNumData(num_data_);
  }

  data_partition_->ResetNumData(num_data_);
  
  ordered_gradients_.resize(num_data_);
  ordered_hessians_.resize(num_data_);

  if (has_ordered_bin_) {
    is_data_in_leaf_.resize(num_data_);
    std::fill(is_data_in_leaf_.begin(), is_data_in_leaf_.end(), static_cast<char>(0));
  }
}

void DecisionTableLearner::ResetConfig(const Config* config) {
  config_ = config;
  int new_depth;
  if(config_->max_depth > 0){
    new_depth = config_->max_depth;
  } else {
    Log::Warning("Max depth is not set, defaulting to 5.");
    new_depth = 5;
  }
  if(new_depth != tree_depth_){
    auto num_leaves = 1 << new_depth;
    histogram_pool_.DynamicChangeSize(train_data_, config_, num_leaves, num_leaves);
    data_partition_->ResetLeaves(num_leaves);
    leaf_splits_.resize(num_leaves);
    for(int i = 0; i < num_leaves; ++i){
      leaf_splits_[i].reset(new LeafSplits(num_data_));
    }
    tree_depth_ = new_depth;
  }
}

void DecisionTableLearner::ConstructHistogram(const std::vector<int8_t>& is_feature_used, const score_t* gradients, const score_t* hessians, int leaf_idx){
  FeatureHistogram* histogram_array;
  histogram_pool_.Get(leaf_idx,&histogram_array);
  HistogramBinEntry* ptr_leaf_hist_data = histogram_array[0].RawData() - 1;
  train_data_->ConstructHistograms(is_feature_used,
				   leaf_splits_[leaf_idx]->data_indices(), leaf_splits_[leaf_idx]->num_data_in_leaf(),
				   leaf_splits_[leaf_idx]->LeafIndex(),
				   ordered_bins_, gradients, hessians,
				   ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
				   ptr_leaf_hist_data);
  for(int feature_idx = 0; feature_idx < is_feature_used.size(); ++feature_idx){
    if(is_feature_used[feature_idx]){
      train_data_->FixHistogram(feature_idx,
				leaf_splits_[leaf_idx]->sum_gradients(), leaf_splits_[leaf_idx]->sum_hessians(),
				leaf_splits_[leaf_idx]->num_data_in_leaf(),
				histogram_array[feature_idx].RawData());
    }
  }
}

void DecisionTableLearner::ConstructHistograms(const std::vector<int8_t>& is_feature_used, const int num_leaves, const score_t* gradients, const score_t* hessians){
  for(int i = 0; i < num_leaves; ++i){
    ConstructHistogram(is_feature_used, gradients, hessians, i);
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
  std::vector<double> best_sum_left_gradient(num_leaves, NAN);
  std::vector<double> best_sum_left_hessian(num_leaves, NAN);
  double best_gain = kMinScore;
  std::vector<double> best_gain_per_node(num_leaves, kMinScore);
  std::vector<double> gain_per_node(num_leaves, kMinScore);
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
	gain_per_node[i] = FeatureHistogram::GetSplitGains(sum_left_gradient[i], sum_left_hessian[i], sum_right_gradient[i], sum_right_hessian[i],
							   config_->lambda_l1, config_->lambda_l2, config_->max_delta_step,
							   leaf_splits_[i]->min_constraint(), leaf_splits_[i]->max_constraint(), train_data_->FeatureMonotone(feature_idx));
	current_gain += gain_per_node[i];
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
	  best_gain_per_node[i] = gain_per_node[i];
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
        gain_per_node[i] = FeatureHistogram::GetSplitGains(sum_left_gradient[i], sum_left_hessian[i], sum_right_gradient[i], sum_right_hessian[i],
							   config_->lambda_l1, config_->lambda_l2, config_->max_delta_step,
							   leaf_splits_[i]->min_constraint(), leaf_splits_[i]->max_constraint(), train_data_->FeatureMonotone(feature_idx));
	current_gain += gain_per_node[i];
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
	  best_gain_per_node[i] = gain_per_node[i];
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
      output.leaf_splits[i].gain = best_gain_per_node[i];
      output.leaf_splits[i].default_left = dir == -1;
    }
    output.gain = best_gain;
  }
}

FeatureSplits DecisionTableLearner::FindBestFeatureSplitNumerical(const int num_leaves, const double min_gain_shift, const std::vector<double>& gain_shifts, const std::vector<FeatureHistogram*>& histogram_arrs, const int feature_idx){
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
  return output;
}

FeatureSplits DecisionTableLearner::FindBestFeatureSplitCategorical(const int num_leaves, const double min_gain_shift, const std::vector<double>& gain_shifts, const std::vector<FeatureHistogram*>& histogram_arrs, const int feature_idx){
  FeatureSplits output(num_leaves);
  auto bin_mapper = train_data_->FeatureBinMapper(feature_idx);
  int num_bin = bin_mapper->num_bin();
  bool is_full_categorical = bin_mapper->missing_type() == MissingType::None;
  int used_bin = num_bin - 1 + is_full_categorical;
  bool use_onehot = num_bin <= config_->max_cat_to_onehot;
  double l2 = config_->lambda_l2;
  if(!use_onehot){
    l2 += config_->cat_l2;
  }

  std::vector<HistogramBinEntry*> leaf_histograms(num_leaves);
  for(int i = 0; i < num_leaves; i++){
    leaf_histograms[i] = histogram_arrs[i][feature_idx].RawData();
  }
  double best_gain = kMinScore;
  std::vector<double> best_gain_per_node(num_leaves, kMinScore);
  std::vector<double> best_sum_left_gradient(num_leaves, 0.0);
  std::vector<double> best_sum_left_hessian(num_leaves, 0.0);
  std::vector<data_size_t> best_left_count(num_leaves, 0);
  std::vector<double> gain_per_node(num_leaves, kMinScore);
  //Per-leaf, because multiple-category splits differ between nodes.
  std::vector<uint32_t> best_threshold(num_leaves, 0);
  std::vector<std::vector<std::pair<int, double>>> leaf_values;
  int best_direction = 0;
  bool is_splittable = false;
  if(use_onehot){
    for (int t = 0; t < used_bin; ++t) {

      bool t_is_splittable = true;
      for(int i = 0; i < num_leaves; ++i){
	// if data not enough, or sum hessian too small
	if (leaf_histograms[i][t].cnt < config_->min_data_in_leaf
	    || leaf_histograms[i][t].sum_hessians < config_->min_sum_hessian_in_leaf) {
	  t_is_splittable = false;
	  break;
	}

	data_size_t other_count = leaf_splits_[i]->num_data_in_leaf() - leaf_histograms[i][t].cnt;
	// if data not enough
	if (other_count < config_->min_data_in_leaf){
	  t_is_splittable = false;
	  break;
	}

	double sum_other_hessian = leaf_splits_[i]->sum_hessians() - leaf_histograms[i][t].sum_hessians - kEpsilon;
	// if sum hessian too small
	if (sum_other_hessian < config_->min_sum_hessian_in_leaf){
	  t_is_splittable = false;
	  break;
	}
      }
      if(!t_is_splittable) continue;

      double current_gain = 0.0;
      for(int i = 0; i < num_leaves; ++i){
	double sum_other_gradient = leaf_splits_[i]->sum_gradients() - leaf_histograms[i][t].sum_gradients;	
	double sum_other_hessian = leaf_splits_[i]->sum_hessians() - leaf_histograms[i][t].sum_hessians - kEpsilon;	
	// current split gain
	gain_per_node[i] = FeatureHistogram::GetSplitGains(sum_other_gradient, sum_other_hessian, leaf_histograms[i][t].sum_gradients, leaf_histograms[i][t].sum_hessians + kEpsilon,
							   config_->lambda_l1, l2, config_->max_delta_step,
							   leaf_splits_[i]->min_constraint(), leaf_splits_[i]->max_constraint(), 0);
	current_gain += gain_per_node[i];
      }

      // gain with split is worse than without split
      if (current_gain <= min_gain_shift) continue;

      // mark to is splittable
      is_splittable = true;
      // better split point
      if (current_gain > best_gain) {
	std::fill(best_threshold.begin(), best_threshold.end(), t);
      	best_gain = current_gain;
	for(int i = 0; i < num_leaves; ++i){
	  best_gain_per_node[i] = gain_per_node[i];
	  best_sum_left_gradient[i] = leaf_histograms[i][t].sum_gradients;
	  best_sum_left_hessian[i] = leaf_histograms[i][t].sum_hessians + kEpsilon;
	  best_left_count[i] = leaf_histograms[i][t].cnt;
	}
      }
    }
  } else {
    std::vector<int> used_bins;
    std::vector<int> index_by_leaf(num_leaves,0);
    double best_leaf_value_threshold;

    //Find categorical values that have enough data points to be worth including.
    for (int i = 0; i < used_bin; ++i) {
      data_size_t total_cnt = 0;
      for(int j = 0; j < num_leaves; j++){
	total_cnt += leaf_histograms[j][i].cnt;
      }
      if (total_cnt >= config_->cat_smooth) {
	used_bins.push_back(i);
      }
    }
    auto ctr_fun = [this](double sum_grad, double sum_hess) {
		     return (sum_grad) / (sum_hess + config_->cat_smooth);
    };
    used_bin = static_cast<int>(used_bins.size());

    //Sort, for each leaf, the per-category output values.
    leaf_values.resize(num_leaves);
    for(int i = 0; i < num_leaves; ++i){
      leaf_values[i].resize(used_bin);
      for(int j = 0; j < used_bin; ++j){
	auto& bin = leaf_histograms[i][used_bins[j]];
	leaf_values[i][j].first = used_bins[j];
	leaf_values[i][j].second = ctr_fun(bin.sum_gradients, bin.sum_hessians);
      }
      std::sort(leaf_values[i].begin(), leaf_values[i].end(),
		[](std::pair<int,double> i, std::pair<int, double> j) {
		  return i.second < j.second;
		});
    }

    std::vector<int> directions = {1, -1};

    for(auto dir : directions){
      //Initialisation step:
      for(int i = 0; i < num_leaves; ++i){
	if(dir == 1){
	  index_by_leaf[i] = 0;
	} else {
	  index_by_leaf[i] = used_bin - 1;
	}
      }

      //Find the current threshold value as the largest/smallest of the per-leaf values.
      //This is the minimum that defines a meaningful split in every leaf.
      double threshold = leaf_values[0][index_by_leaf[0]].second;
      for(int i = 1; i < num_leaves; ++i){
	if((dir == 1 && leaf_values[i][index_by_leaf[i]].second > threshold) ||
	   (dir == -1 && leaf_values[i][index_by_leaf[i]].second < threshold)){
	  threshold = leaf_values[i][index_by_leaf[i]].second;
	}
      }

      int max_index = std::min(std::min(config_->max_cat_threshold - 1, (used_bin+1)/2), used_bin-2);

      bool unsplittable = false;
      //Ensure each leaf index is the last value below threshold.
      for(int i = 0; i < num_leaves; ++i){
	if(dir == 1){
	  while(index_by_leaf[i] < used_bin - 1 && leaf_values[i][index_by_leaf[i]+1].second <= threshold){
	    index_by_leaf[i]++;
	  }
	  //Cannot split this leaf node, as even the smallest feasible threshold is too far out.
	  if(index_by_leaf[i] > max_index)
	    unsplittable = true;
	} else {
	  while(index_by_leaf[i] > 0 && leaf_values[i][index_by_leaf[i]-1].second >= threshold){
	    index_by_leaf[i]--;
	  }
	  //Cannot split this leaf node, as even the smallest feasible threshold is too far out.
	  if(used_bin - 1 - index_by_leaf[i] > max_index)
	    unsplittable = true;
	}
      }

      if(!unsplittable){
	//Statistics on the left-hand side of the split, by leaf.
	std::vector<double> sum_left_gradient(num_leaves, 0.0);
	std::vector<double> sum_left_hessian(num_leaves, 0.0);
	std::vector<data_size_t> left_count(num_leaves, 0);
	//Gains for each leaf node.
	//These are incrementally updated as we move left-to-right through
	//the thresholds.
	std::vector<double> gain_per_node(num_leaves, 0.0);
	double current_gain = 0.0;

	for(int i = 0; i < num_leaves; ++i){
	  if(dir == 1){
	    for(int j = 0; j <= index_by_leaf[i]; ++j){
	      int bin = leaf_values[i][j].first;
	      sum_left_gradient[i] += leaf_histograms[i][bin].sum_gradients;
	      sum_left_hessian[i] += leaf_histograms[i][bin].sum_hessians;
	      left_count[i] += leaf_histograms[i][bin].cnt;
	    }
	  } else {
	    for(int j = used_bin - 1; j >= index_by_leaf[i]; --j){
	      int bin = leaf_values[i][j].first;
	      sum_left_gradient[i] += leaf_histograms[i][bin].sum_gradients;
	      sum_left_hessian[i] += leaf_histograms[i][bin].sum_hessians;
	      left_count[i] += leaf_histograms[i][bin].cnt;
	    }
	  }
	  double sum_right_gradient = leaf_splits_[i]->sum_gradients()-sum_left_gradient[i];
	  double sum_right_hessian = leaf_splits_[i]->sum_hessians()-sum_left_hessian[i];
	  gain_per_node[i] = FeatureHistogram::GetSplitGains(sum_left_gradient[i], sum_left_hessian[i], sum_right_gradient, sum_right_hessian,
							     config_->lambda_l1, l2, config_->max_delta_step,
							     leaf_splits_[i]->min_constraint(), leaf_splits_[i]->max_constraint(), 0);
	  current_gain += gain_per_node[i];
	}

	while(true){
	  bool split_acceptable = true;
	  bool no_further_splits = false;
	  for(int i = 0; i < num_leaves; ++i){
	    if(left_count[i] < config_->min_data_in_leaf
	       || left_count[i] < config_->min_data_per_group //This seems saner than the original version (bug?)
	       || sum_left_hessian[i] < config_->min_sum_hessian_in_leaf){
	      split_acceptable = false;
	      break;
	    }
	    data_size_t right_count = leaf_splits_[i]->num_data_in_leaf() - left_count[i];
	    if (right_count < config_->min_data_in_leaf || right_count < config_->min_data_per_group){
	      no_further_splits = true;
	      break;
	    }

	    double sum_right_hessian = leaf_splits_[i]->sum_hessians() - sum_left_hessian[i];
	    if (sum_right_hessian < config_->min_sum_hessian_in_leaf){
	      no_further_splits = true;
	      break;
	    }
	  }
	  if(no_further_splits)
	    break;
	  if(split_acceptable && current_gain > min_gain_shift){
	    is_splittable = true;
	    if(current_gain > best_gain){
	      for(int i = 0; i < num_leaves; ++i){
		best_left_count[i] = left_count[i];
		best_sum_left_gradient[i] = sum_left_gradient[i];
		best_sum_left_hessian[i] = sum_left_hessian[i];
		best_threshold[i] = index_by_leaf[i];
		best_gain_per_node[i] = gain_per_node[i];
	      }
	      best_gain = current_gain;
	      best_direction = dir;
	      best_leaf_value_threshold = threshold;
	    }
	  }
	  //Advance to the next split point.
	  int min_idx = 0;
	  for(int i = 1; i < num_leaves; i++){
	    if((dir == 1 && leaf_values[i][index_by_leaf[i]+1].second < leaf_values[min_idx][index_by_leaf[min_idx]+1].second)
	       || (dir == -1 && leaf_values[i][index_by_leaf[i]-1].second > leaf_values[min_idx][index_by_leaf[min_idx]-1].second)){
	      min_idx = i;
	    }
	  }
	  index_by_leaf[min_idx]+=dir;
	  if((dir == 1 && index_by_leaf[min_idx] > max_index) || (dir == -1 && used_bin - 1 - index_by_leaf[min_idx] > max_index))
	    break;
	  int min_idx_bin = leaf_values[min_idx][index_by_leaf[min_idx]].first;
	  sum_left_gradient[min_idx] += leaf_histograms[min_idx][min_idx_bin].sum_gradients;
	  sum_left_hessian[min_idx] += leaf_histograms[min_idx][min_idx_bin].sum_hessians;
	  left_count[min_idx] += leaf_histograms[min_idx][min_idx_bin].cnt;
	  double sum_right_gradient = leaf_splits_[min_idx]->sum_gradients()-sum_left_gradient[min_idx];
	  double sum_right_hessian = leaf_splits_[min_idx]->sum_hessians()-sum_left_hessian[min_idx];
	  current_gain -= gain_per_node[min_idx];
	  gain_per_node[min_idx] = FeatureHistogram::GetSplitGains(sum_left_gradient[min_idx], sum_left_hessian[min_idx], sum_right_gradient, sum_right_hessian,
								   config_->lambda_l1, l2, config_->max_delta_step,
								   leaf_splits_[min_idx]->min_constraint(), leaf_splits_[min_idx]->max_constraint(), 0);
	  current_gain += gain_per_node[min_idx];
	  threshold = leaf_values[min_idx][index_by_leaf[min_idx]].second;
	}
      }
    }
    if(best_gain > kMinScore){
      for(int i = 0; i < num_leaves; ++i){
	if(best_direction >= 0){
	  for(int j = 0; j <= best_threshold[i]; ++j){
	    if(leaf_values[i][j].second > best_leaf_value_threshold){
	      throw std::runtime_error("Threshold should be the largest included value in leaf, it is not.");
	    }
	  }
	} else {
	  for(int j = best_threshold[i]; j <= used_bin-1; ++j){
	    if(leaf_values[i][j].second < best_leaf_value_threshold){
	      throw std::runtime_error("Threshold should be the smallest included value in leaf, it is not.");
	    }
	  }
	}
      }
    }
  }

  if (is_splittable) {
    output.gain = best_gain;
    for(int i = 0; i < num_leaves; ++i){
      output.leaf_splits[i].left_output = FeatureHistogram::CalculateSplittedLeafOutput(best_sum_left_gradient[i], best_sum_left_hessian[i],
											config_->lambda_l1, l2, config_->max_delta_step,
											leaf_splits_[i]->min_constraint(), leaf_splits_[i]->max_constraint());
      output.leaf_splits[i].left_count = best_left_count[i];
      output.leaf_splits[i].left_sum_gradient = best_sum_left_gradient[i];
      output.leaf_splits[i].left_sum_hessian = best_sum_left_hessian[i] - kEpsilon;
      output.leaf_splits[i].right_output = FeatureHistogram::CalculateSplittedLeafOutput(leaf_splits_[i]->sum_gradients() - best_sum_left_gradient[i],
											 leaf_splits_[i]->sum_hessians() - best_sum_left_hessian[i],
											 config_->lambda_l1, l2, config_->max_delta_step,
											 leaf_splits_[i]->min_constraint(), leaf_splits_[i]->max_constraint());
      output.leaf_splits[i].right_count = leaf_splits_[i]->num_data_in_leaf() - best_left_count[i];
      output.leaf_splits[i].right_sum_gradient = leaf_splits_[i]->sum_gradients() - best_sum_left_gradient[i];
      output.leaf_splits[i].right_sum_hessian = leaf_splits_[i]->sum_hessians() - best_sum_left_hessian[i] - kEpsilon;
      output.leaf_splits[i].gain = best_gain_per_node[i];
      if (use_onehot) {
        output.leaf_splits[i].num_cat_threshold = 1;
        output.leaf_splits[i].cat_threshold = std::vector<uint32_t>(1, static_cast<uint32_t>(best_threshold[i]));
      } else {
        output.leaf_splits[i].num_cat_threshold = best_threshold[i] + 1;
        output.leaf_splits[i].cat_threshold = std::vector<uint32_t>(output.leaf_splits[i].num_cat_threshold);
	if(best_direction == 1){
	  for (int j = 0; j < output.leaf_splits[i].num_cat_threshold; ++j) {
	    auto t = leaf_values[i][j].first;
	    output.leaf_splits[i].cat_threshold[j] = t;
	  }
	} else {
	  for (int j = 0; j < output.leaf_splits[i].num_cat_threshold; ++j) {
	    auto t = leaf_values[i][used_bin - 1 - j].first;
	    output.leaf_splits[i].cat_threshold[j] = t;
	  }
	}
      }
      output.leaf_splits[i].monotone_type = 0;
      output.leaf_splits[i].min_constraint = leaf_splits_[i]->min_constraint();
      output.leaf_splits[i].max_constraint = leaf_splits_[i]->max_constraint();
    }
  }
  return output;
}

FeatureSplits DecisionTableLearner::FindBestFeatureSplit(const int num_leaves, const double min_gain_shift, const std::vector<double>& gain_shifts, const std::vector<FeatureHistogram*>& histogram_arrs, const int feature_idx){
  FeatureSplits output(num_leaves);
  if(train_data_->FeatureBinMapper(feature_idx)->bin_type() == BinType::NumericalBin){
    output = FindBestFeatureSplitNumerical(num_leaves, min_gain_shift, gain_shifts, histogram_arrs, feature_idx);
  } else {
    output = FindBestFeatureSplitCategorical(num_leaves, min_gain_shift, gain_shifts, histogram_arrs, feature_idx);
  }
  int real_fidx = train_data_->RealFeatureIndex(feature_idx);
  output.gain -= min_gain_shift;
  for(int i = 0; i < num_leaves; ++i){
    output.leaf_splits[i].feature = real_fidx;
    output.leaf_splits[i].gain -= gain_shifts[i];
  }
  return output;
}

FeatureSplits DecisionTableLearner::FindBestSplit(const std::vector<int8_t>& is_feature_used, const int num_leaves, const score_t* gradients, const score_t* hessians){
  FeatureSplits ret(num_leaves);
  std::vector<FeatureHistogram*> histogram_arrs(num_leaves);
  for(int leaf_idx = 0; leaf_idx < num_leaves; ++leaf_idx){
    histogram_pool_.Get(leaf_idx, &histogram_arrs[leaf_idx]);
  }
  std::vector<double> gain_shifts(num_leaves, kMinScore);
  for(int feature_idx = 0; feature_idx < train_data_->num_features(); ++feature_idx){
    if(is_feature_used[feature_idx]){
      double min_shift_gain = 0.0;
      double additional_cost_per_node = config_->cegb_penalty_split;
      if(!feature_used_[feature_idx] && !config_->cegb_penalty_feature_coupled.empty()){
	additional_cost_per_node = config_->cegb_penalty_feature_coupled[feature_idx]/num_leaves;
      }
      additional_cost_per_node *= config_->cegb_tradeoff;
      for(int leaf_idx = 0; leaf_idx < num_leaves; ++leaf_idx){
	//Step 1: Compute minimum gain to overcome.
	gain_shifts[leaf_idx] =
	  histogram_arrs[leaf_idx][feature_idx].GetLeafSplitGain(leaf_splits_[leaf_idx]->sum_gradients(),
								 leaf_splits_[leaf_idx]->sum_hessians(),
								 config_->lambda_l1,
								 config_->lambda_l2,
								 config_->max_delta_step);
	gain_shifts[leaf_idx] += config_->min_gain_to_split + additional_cost_per_node;
	if(!feature_used_[feature_idx] && !config_->cegb_penalty_feature_lazy.empty()){
	  gain_shifts[leaf_idx] +=
	    config_->cegb_tradeoff*config_->cegb_penalty_feature_lazy[feature_idx]*leaf_splits_[leaf_idx]->num_data_in_leaf();
	}
	min_shift_gain += gain_shifts[leaf_idx];
      }
      //Find best split for this feature.
      auto split = FindBestFeatureSplit(num_leaves, min_shift_gain, gain_shifts, histogram_arrs, feature_idx);
      if(split.gain > ret.gain){
	ret = split;
      }
    }
  }
  return ret;
}

void DecisionTableLearner::Split(const std::vector<int8_t>& is_feature_used, Tree* tree, const FeatureSplits& split, const score_t* gradients, const score_t* hessians){
  const int inner_feature_index = train_data_->InnerFeatureIndex(split.leaf_splits[0].feature);
  feature_used_[inner_feature_index] = true;
  bool is_numerical_split = train_data_->FeatureBinMapper(inner_feature_index)->bin_type() == BinType::NumericalBin;
  for(int left_leaf = 0; left_leaf < split.leaf_splits.size(); left_leaf++){
    int right_leaf;
    auto split_info = split.leaf_splits[left_leaf];
    if(is_numerical_split){
      auto threshold_double = train_data_->RealThreshold(inner_feature_index, split.leaf_splits[left_leaf].threshold);
      // split tree, will return right leaf
      right_leaf = tree->Split(left_leaf,
			       inner_feature_index,
			       split_info.feature,
			       split_info.threshold,
			       threshold_double,
			       static_cast<double>(split_info.left_output),
			       static_cast<double>(split_info.right_output),
			       static_cast<data_size_t>(split_info.left_count),
			       static_cast<data_size_t>(split_info.right_count),
			       static_cast<float>(split_info.gain),
			       train_data_->FeatureBinMapper(inner_feature_index)->missing_type(),
			       split_info.default_left);
      data_partition_->Split(left_leaf, train_data_, inner_feature_index,
			     &split_info.threshold, 1, split_info.default_left, right_leaf);
    } else {
      std::vector<uint32_t> cat_bitset_inner = Common::ConstructBitset(split_info.cat_threshold.data(), split_info.num_cat_threshold);
      std::vector<int> threshold_int(split_info.num_cat_threshold);
      for (int i = 0; i < split_info.num_cat_threshold; ++i) {
	threshold_int[i] = static_cast<int>(train_data_->RealThreshold(inner_feature_index, split_info.cat_threshold[i]));
      }
      std::vector<uint32_t> cat_bitset = Common::ConstructBitset(threshold_int.data(), split_info.num_cat_threshold);
      right_leaf = tree->SplitCategorical(left_leaf,
					  inner_feature_index,
					  split_info.feature,
					  cat_bitset_inner.data(),
					  static_cast<int>(cat_bitset_inner.size()),
					  cat_bitset.data(),
					  static_cast<int>(cat_bitset.size()),
					  static_cast<double>(split_info.left_output),
					  static_cast<double>(split_info.right_output),
					  static_cast<data_size_t>(split_info.left_count),
					  static_cast<data_size_t>(split_info.right_count),
					  static_cast<float>(split_info.gain),
					  train_data_->FeatureBinMapper(inner_feature_index)->missing_type());
      data_partition_->Split(left_leaf, train_data_, inner_feature_index,
			     cat_bitset_inner.data(), static_cast<int>(cat_bitset_inner.size()), split_info.default_left, right_leaf);
    }
    PerformSplit(left_leaf, right_leaf, gradients, hessians, is_feature_used);
  }
}

void DecisionTableLearner::PerformSplit(const int left_leaf, const int right_leaf, const score_t* gradients, const score_t* hessians, const std::vector<int8_t>& is_feature_used){
  leaf_splits_[left_leaf]->Init(left_leaf, data_partition_.get(), gradients, hessians);
  leaf_splits_[right_leaf]->Init(right_leaf, data_partition_.get(), gradients, hessians);

  if (has_ordered_bin_) {
    // mark data that at left-leaf
    const data_size_t* indices = data_partition_->indices();
    const auto left_cnt = data_partition_->leaf_count(left_leaf);
    char mark = 1;
    data_size_t begin = data_partition_->leaf_begin(left_leaf);
    data_size_t end = begin + left_cnt;
#pragma omp parallel for schedule(static, 512) if (end - begin >= 1024)
    for (data_size_t i = begin; i < end; ++i) {
      is_data_in_leaf_[indices[i]] = 1;
    }
    OMP_INIT_EX();
    // split the ordered bin
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(ordered_bin_indices_.size()); ++i) {
      OMP_LOOP_EX_BEGIN();
      ordered_bins_[ordered_bin_indices_[i]]->Split(left_leaf, right_leaf, is_data_in_leaf_.data(), mark);
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
#pragma omp parallel for schedule(static, 512) if (end - begin >= 1024)
    for (data_size_t i = begin; i < end; ++i) {
      is_data_in_leaf_[indices[i]] = 0;
    }
  }

  //Left node is smaller, move the parent histogram (now in left leaf) into the right leaf.
  //Then compute left histogram and subtract from right.
  if(leaf_splits_[left_leaf]->num_data_in_leaf() < leaf_splits_[right_leaf]->num_data_in_leaf()){
    FeatureHistogram* left_arr, *right_arr;
    histogram_pool_.Move(left_leaf, right_leaf);
    histogram_pool_.Get(right_leaf, &right_arr);
    histogram_pool_.Get(left_leaf, &left_arr);
    ConstructHistogram(is_feature_used, gradients, hessians, left_leaf);
    for(int i = 0; i < is_feature_used.size(); ++i){
      if(!is_feature_used[i]){ continue;}
      right_arr[i].Subtract(left_arr[i]);
    }
  //Right node is smaller, construct its histogram and subtract from left. (This path appears fine, trigger the above unconditionally as a debugging exercise)
  } else {
    FeatureHistogram* left_arr, *right_arr;
    ConstructHistogram(is_feature_used, gradients, hessians, right_leaf);
    histogram_pool_.Get(right_leaf, &right_arr);
    histogram_pool_.Get(left_leaf, &left_arr);
    for(int i = 0; i < is_feature_used.size(); ++i){
      if(!is_feature_used[i]){ continue;}
      left_arr[i].Subtract(right_arr[i]);
    }    
  }
}

void DecisionTableLearner::SampleFeatures(std::vector<int8_t>& is_feature_used){
  int num_features = is_feature_used.size();  
  if (config_->feature_fraction < 1) {
    int used_feature_cnt = static_cast<int>(valid_feature_indices_.size()*config_->feature_fraction);
    // at least use one feature
    used_feature_cnt = std::max(used_feature_cnt, 1);
    // initialize used features
    std::memset(is_feature_used.data(), 0, sizeof(int8_t) * num_features);
    // Get used feature at current tree
    auto sampled_indices = random_.Sample(static_cast<int>(valid_feature_indices_.size()), used_feature_cnt);
    int omp_loop_size = static_cast<int>(sampled_indices.size());
    #pragma omp parallel for schedule(static, 512) if (omp_loop_size >= 1024)
    for (int i = 0; i < omp_loop_size; ++i) {
      int used_feature = valid_feature_indices_[sampled_indices[i]];
      int inner_feature_index = train_data_->InnerFeatureIndex(used_feature);
      CHECK(inner_feature_index >= 0);
      is_feature_used[inner_feature_index] = 1;
    }
  } else {
    #pragma omp parallel for schedule(static, 512) if (num_features >= 1024)
    for (int i = 0; i < num_features; ++i) {
      is_feature_used[i] = 1;
    }
  }
}

void DecisionTableLearner::InitOrderedBin(){
  // if has ordered bin, need to initialize the ordered bin
  if (has_ordered_bin_) {
    if (data_partition_->leaf_count(0) == num_data_) {
      // use all data, pass nullptr
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < static_cast<int>(ordered_bin_indices_.size()); ++i) {
        OMP_LOOP_EX_BEGIN();
        ordered_bins_[ordered_bin_indices_[i]]->Init(nullptr, config_->num_leaves);
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
    } else {
      // bagging, only use part of data

      // mark used data
      const data_size_t* indices = data_partition_->indices();
      data_size_t begin = data_partition_->leaf_begin(0);
      data_size_t end = begin + data_partition_->leaf_count(0);
      #pragma omp parallel for schedule(static, 512) if (end - begin >= 1024)
      for (data_size_t i = begin; i < end; ++i) {
        is_data_in_leaf_[indices[i]] = 1;
      }
      OMP_INIT_EX();
      // initialize ordered bin
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < static_cast<int>(ordered_bin_indices_.size()); ++i) {
        OMP_LOOP_EX_BEGIN();
        ordered_bins_[ordered_bin_indices_[i]]->Init(is_data_in_leaf_.data(), config_->num_leaves);
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
      #pragma omp parallel for schedule(static, 512) if (end - begin >= 1024)
      for (data_size_t i = begin; i < end; ++i) {
        is_data_in_leaf_[indices[i]] = 0;
      }
    }
  }
}

Tree* DecisionTableLearner::Train(const score_t* gradients, const score_t *hessians, bool is_constant_hessian, Json& forced_split_json) {
  auto num_leaves = 1 << tree_depth_;
  auto tree = std::unique_ptr<Tree>(new Tree(num_leaves));
  std::vector<int8_t> is_feature_used(train_data_->num_features());
  SampleFeatures(is_feature_used);  
  //Puts all data in the leaf.
  data_partition_->Init();
  leaf_splits_[0]->Init(0, data_partition_.get(), gradients, hessians);
  InitOrderedBin();
  int32_t cur_depth = 1;
  ConstructHistogram(is_feature_used, gradients, hessians, 0);
  if (!forced_split_json.is_null()) {
    ForceSplits(tree.get(), forced_split_json, &cur_depth, gradients, hessians);
  }
  for(int i = cur_depth - 1; i < tree_depth_ - 1; ++i){
    auto split = FindBestSplit(is_feature_used, tree->num_leaves(), gradients, hessians);
    if(split.gain <= 0.0){
      Log::Warning("No further splits with positive gain, best gain: %f", split.gain);
      break;
    }
    Split(is_feature_used, tree.get(), split, gradients, hessians);
  }
  return tree.release();
}

Tree* DecisionTableLearner::FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t *hessians) const {
  auto tree = std::unique_ptr<Tree>(new Tree(*old_tree));
  CHECK(data_partition_->num_leaves() >= tree->num_leaves());
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < tree->num_leaves(); ++i) {
    OMP_LOOP_EX_BEGIN();
    data_size_t cnt_leaf_data = 0;
    auto tmp_idx = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
    double sum_grad = 0.0f;
    double sum_hess = kEpsilon;
    for (data_size_t j = 0; j < cnt_leaf_data; ++j) {
      auto idx = tmp_idx[j];
      sum_grad += gradients[idx];
      sum_hess += hessians[idx];
    }
    double output = FeatureHistogram::CalculateSplittedLeafOutput(sum_grad, sum_hess,
                                                                  config_->lambda_l1, config_->lambda_l2, config_->max_delta_step);
    auto old_leaf_output = tree->LeafOutput(i);
    auto new_leaf_output = output * tree->shrinkage();
    tree->SetLeafOutput(i, config_->refit_decay_rate * old_leaf_output + (1.0 - config_->refit_decay_rate) * new_leaf_output);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  return tree.release();
}

Tree* DecisionTableLearner::FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred, const score_t* gradients, const score_t *hessians) {
  data_partition_->ResetByLeafPred(leaf_pred, old_tree->num_leaves());
  return FitByExistingTree(old_tree, gradients, hessians);  
}

void DecisionTableLearner::RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, const double* prediction,
                                        data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt) const {
  if (obj != nullptr && obj->IsRenewTreeOutput()) {
    CHECK(tree->num_leaves() <= data_partition_->num_leaves());
    const data_size_t* bag_mapper = nullptr;
    if (total_num_data != num_data_) {
      CHECK(bag_cnt == num_data_);
      bag_mapper = bag_indices;
    }
    std::vector<int> n_nozeroworker_perleaf(tree->num_leaves(), 1);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < tree->num_leaves(); ++i) {
      const double output = static_cast<double>(tree->LeafOutput(i));
      data_size_t cnt_leaf_data = 0;
      auto index_mapper = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
      if (cnt_leaf_data > 0) {
        // bag_mapper[index_mapper[i]]
        const double new_output = obj->RenewTreeOutput(output, prediction, index_mapper, bag_mapper, cnt_leaf_data);
        tree->SetLeafOutput(i, new_output);
      } else {
	throw std::runtime_error("All leaves should contain data in decision table learner.");
      }
    }
  }
}

void DecisionTableLearner::RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, double prediction,
  data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt) const {
  if (obj != nullptr && obj->IsRenewTreeOutput()) {
    CHECK(tree->num_leaves() <= data_partition_->num_leaves());
    const data_size_t* bag_mapper = nullptr;
    if (total_num_data != num_data_) {
      CHECK(bag_cnt == num_data_);
      bag_mapper = bag_indices;
    }
    std::vector<int> n_nozeroworker_perleaf(tree->num_leaves(), 1);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < tree->num_leaves(); ++i) {
      const double output = static_cast<double>(tree->LeafOutput(i));
      data_size_t cnt_leaf_data = 0;
      auto index_mapper = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
      if (cnt_leaf_data > 0) {
        // bag_mapper[index_mapper[i]]
        const double new_output = obj->RenewTreeOutput(output, prediction, index_mapper, bag_mapper, cnt_leaf_data);
        tree->SetLeafOutput(i, new_output);
      } else {
	throw std::runtime_error("All leaves should have data in decision table learner");
      }
    }
  }
}

int32_t DecisionTableLearner::ForceSplits(Tree* tree, Json& forced_split_json, int32_t* cur_depth, const score_t* gradients, const score_t* hessians) {
  int32_t result_count = 0;
  // start at root leaf
  int32_t left_leaf = 0;
  int32_t right_leaf;
  std::queue<std::pair<Json, int>> q;
  Json left = forced_split_json;
  Json right;
  std::unordered_map<int, SplitInfo> forceSplitMap;
  //TODO: Decide whether to pull this out.
  std::vector<int8_t> is_feature_used(train_data_->num_features(), true);
  q.push(std::make_pair(forced_split_json, left_leaf));
  while (!q.empty()) {
    //Needed to ensure that split computation below is accurate.
    ConstructHistograms(is_feature_used, tree->num_leaves(), gradients, hessians);
    // then, compute own splits
    SplitInfo left_split;
    SplitInfo right_split;

    if (!left.is_null()) {
      const int left_feature = left["feature"].int_value();
      const double left_threshold_double = left["threshold"].number_value();
      const int left_inner_feature_index = train_data_->InnerFeatureIndex(left_feature);
      const uint32_t left_threshold = train_data_->BinThreshold(
              left_inner_feature_index, left_threshold_double);
      FeatureHistogram* leaf_histogram_array;
      histogram_pool_.Get(left_leaf, &leaf_histogram_array);
      auto& left_leaf_splits = leaf_splits_[left_leaf];
      leaf_histogram_array[left_inner_feature_index].GatherInfoForThreshold(
              left_leaf_splits->sum_gradients(),
              left_leaf_splits->sum_hessians(),
              left_threshold,
              left_leaf_splits->num_data_in_leaf(),
              &left_split);
      left_split.feature = left_feature;
      forceSplitMap[left_leaf] = left_split;
      if (left_split.gain < 0) {
        forceSplitMap.erase(left_leaf);
      }
    }

    if (!right.is_null()) {
      const int right_feature = right["feature"].int_value();
      const double right_threshold_double = right["threshold"].number_value();
      const int right_inner_feature_index = train_data_->InnerFeatureIndex(right_feature);
      const uint32_t right_threshold = train_data_->BinThreshold(
              right_inner_feature_index, right_threshold_double);
      FeatureHistogram* leaf_histogram_array;
      histogram_pool_.Get(right_leaf, &leaf_histogram_array);
      auto& right_leaf_splits = leaf_splits_[right_leaf];
      leaf_histogram_array[right_inner_feature_index].GatherInfoForThreshold(
        right_leaf_splits->sum_gradients(),
        right_leaf_splits->sum_hessians(),
        right_threshold,
        right_leaf_splits->num_data_in_leaf(),
        &right_split);
      right_split.feature = right_feature;
      forceSplitMap[right_leaf] = right_split;
      if (right_split.gain < 0) {
        forceSplitMap.erase(right_leaf);
      }
    }

    std::pair<Json, int> pair = q.front();
    q.pop();
    int current_leaf = pair.second;
    // split info should exist because searching in bfs fashion - should have added from parent
    if (forceSplitMap.find(current_leaf) == forceSplitMap.end()) {
        break;
    }
    SplitInfo current_split_info = forceSplitMap[current_leaf];
    const int inner_feature_index = train_data_->InnerFeatureIndex(
            current_split_info.feature);
    auto threshold_double = train_data_->RealThreshold(
            inner_feature_index, current_split_info.threshold);

    // split tree, will return right leaf
    left_leaf = current_leaf;
    if (train_data_->FeatureBinMapper(inner_feature_index)->bin_type() == BinType::NumericalBin) {
      right_leaf = tree->Split(current_leaf,
			       inner_feature_index,
			       current_split_info.feature,
			       current_split_info.threshold,
			       threshold_double,
			       static_cast<double>(current_split_info.left_output),
			       static_cast<double>(current_split_info.right_output),
			       static_cast<data_size_t>(current_split_info.left_count),
			       static_cast<data_size_t>(current_split_info.right_count),
			       static_cast<float>(current_split_info.gain),
			       train_data_->FeatureBinMapper(inner_feature_index)->missing_type(),
			       current_split_info.default_left);
      data_partition_->Split(current_leaf, train_data_, inner_feature_index,
                             &current_split_info.threshold, 1,
                             current_split_info.default_left, right_leaf);
    } else {
      std::vector<uint32_t> cat_bitset_inner = Common::ConstructBitset(
              current_split_info.cat_threshold.data(), current_split_info.num_cat_threshold);
      std::vector<int> threshold_int(current_split_info.num_cat_threshold);
      for (int i = 0; i < current_split_info.num_cat_threshold; ++i) {
        threshold_int[i] = static_cast<int>(train_data_->RealThreshold(
                    inner_feature_index, current_split_info.cat_threshold[i]));
      }
      std::vector<uint32_t> cat_bitset = Common::ConstructBitset(
              threshold_int.data(), current_split_info.num_cat_threshold);
      right_leaf = tree->SplitCategorical(current_leaf,
					  inner_feature_index,
					  current_split_info.feature,
					  cat_bitset_inner.data(),
					  static_cast<int>(cat_bitset_inner.size()),
					  cat_bitset.data(),
					  static_cast<int>(cat_bitset.size()),
					  static_cast<double>(current_split_info.left_output),
					  static_cast<double>(current_split_info.right_output),
					  static_cast<data_size_t>(current_split_info.left_count),
					  static_cast<data_size_t>(current_split_info.right_count),
					  static_cast<float>(current_split_info.gain),
					  train_data_->FeatureBinMapper(inner_feature_index)->missing_type());
      data_partition_->Split(current_leaf, train_data_, inner_feature_index,
                             cat_bitset_inner.data(), static_cast<int>(cat_bitset_inner.size()),
                             current_split_info.default_left, right_leaf);
    }

    PerformSplit(left_leaf, right_leaf, gradients, hessians, is_feature_used);

    left = Json();
    right = Json();
    if ((pair.first).object_items().count("left") > 0) {
      left = (pair.first)["left"];
      if (left.object_items().count("feature") > 0 && left.object_items().count("threshold") > 0) {
        q.push(std::make_pair(left, left_leaf));
      }
    }
    if ((pair.first).object_items().count("right") > 0) {
      right = (pair.first)["right"];
      if (right.object_items().count("feature") > 0 && right.object_items().count("threshold") > 0) {
        q.push(std::make_pair(right, right_leaf));
      }
    }
    result_count++;
    *(cur_depth) = std::max(*(cur_depth), tree->leaf_depth(left_leaf));
  }
  return result_count;
}


}  // namespace LightGBM
