#include<LightGBM/feature_histogram.h>
#include<vector>

namespace LightGBM {

void FeatureHistogram::Init(HistogramBinEntry* data, const FeatureMetainfo* meta){
  meta_ = meta;
  data_ = data;
  if (meta_->bin_type == BinType::NumericalBin) {
    find_best_threshold_fun_ = std::bind(&FeatureHistogram::FindBestThresholdNumerical, this, std::placeholders::_1
					 , std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
  } else {
    find_best_threshold_fun_ = std::bind(&FeatureHistogram::FindBestThresholdCategorical, this, std::placeholders::_1
					 , std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
  }
}

void FeatureHistogram::FindBestThresholdNumerical(double sum_gradient, double sum_hessian, data_size_t num_data, double min_constraint, double max_constraint, SplitInfo* output) {
  is_splittable_ = false;
  double gain_shift = GetLeafSplitGain(sum_gradient, sum_hessian,
				       meta_->config->lambda_l1, meta_->config->lambda_l2, meta_->config->max_delta_step);
  double min_gain_shift = gain_shift + meta_->config->min_gain_to_split;
  if (meta_->num_bin > 2 && meta_->missing_type != MissingType::None) {
    if (meta_->missing_type == MissingType::Zero) {
      FindBestThresholdSequence(sum_gradient, sum_hessian, num_data, min_constraint, max_constraint, min_gain_shift, output, -1, true, false);
      FindBestThresholdSequence(sum_gradient, sum_hessian, num_data, min_constraint, max_constraint, min_gain_shift, output, 1, true, false);
    } else {
      FindBestThresholdSequence(sum_gradient, sum_hessian, num_data, min_constraint, max_constraint, min_gain_shift, output, -1, false, true);
      FindBestThresholdSequence(sum_gradient, sum_hessian, num_data, min_constraint, max_constraint, min_gain_shift, output, 1, false, true);
    }
  } else {
    FindBestThresholdSequence(sum_gradient, sum_hessian, num_data, min_constraint, max_constraint, min_gain_shift, output, -1, false, false);
    // fix the direction error when only have 2 bins
    if (meta_->missing_type == MissingType::NaN) {
      output->default_left = false;
    }
  }
  output->gain -= min_gain_shift;
  output->monotone_type = meta_->monotone_type;
  output->min_constraint = min_constraint;
  output->max_constraint = max_constraint;
}

void FeatureHistogram::FindBestThresholdCategorical(double sum_gradient, double sum_hessian, data_size_t num_data,
				  double min_constraint, double max_constraint,
				  SplitInfo* output) {
  output->default_left = false;
  double best_gain = kMinScore;
  data_size_t best_left_count = 0;
  double best_sum_left_gradient = 0;
  double best_sum_left_hessian = 0;
  double gain_shift = GetLeafSplitGain(sum_gradient, sum_hessian, meta_->config->lambda_l1, meta_->config->lambda_l2, meta_->config->max_delta_step);

  double min_gain_shift = gain_shift + meta_->config->min_gain_to_split;
  bool is_full_categorical = meta_->missing_type == MissingType::None;
  int used_bin = meta_->num_bin - 1 + is_full_categorical;

  std::vector<int> sorted_idx;
  double l2 = meta_->config->lambda_l2;
  bool use_onehot = meta_->num_bin <= meta_->config->max_cat_to_onehot;
  int best_threshold = -1;
  int best_dir = 1;

  if (use_onehot) {
    for (int t = 0; t < used_bin; ++t) {
      // if data not enough, or sum hessian too small
      if (data_[t].cnt < meta_->config->min_data_in_leaf
	  || data_[t].sum_hessians < meta_->config->min_sum_hessian_in_leaf) continue;
      data_size_t other_count = num_data - data_[t].cnt;
      // if data not enough
      if (other_count < meta_->config->min_data_in_leaf) continue;

      double sum_other_hessian = sum_hessian - data_[t].sum_hessians - kEpsilon;
      // if sum hessian too small
      if (sum_other_hessian < meta_->config->min_sum_hessian_in_leaf) continue;

      double sum_other_gradient = sum_gradient - data_[t].sum_gradients;
      // current split gain
      double current_gain = GetSplitGains(sum_other_gradient, sum_other_hessian, data_[t].sum_gradients, data_[t].sum_hessians + kEpsilon,
					  meta_->config->lambda_l1, l2, meta_->config->max_delta_step,
					  min_constraint, max_constraint, 0);
      // gain with split is worse than without split
      if (current_gain <= min_gain_shift) continue;

      // mark to is splittable
      is_splittable_ = true;
      // better split point
      if (current_gain > best_gain) {
	best_threshold = t;
	best_sum_left_gradient = data_[t].sum_gradients;
	best_sum_left_hessian = data_[t].sum_hessians + kEpsilon;
	best_left_count = data_[t].cnt;
	best_gain = current_gain;
      }
    }
  } else {
    for (int i = 0; i < used_bin; ++i) {
      if (data_[i].cnt >= meta_->config->cat_smooth) {
	sorted_idx.push_back(i);
      }
    }
    used_bin = static_cast<int>(sorted_idx.size());

    l2 += meta_->config->cat_l2;

    auto ctr_fun = [this](double sum_grad, double sum_hess) {
		     return (sum_grad) / (sum_hess + meta_->config->cat_smooth);
		   };
    std::sort(sorted_idx.begin(), sorted_idx.end(),
	      [this, &ctr_fun](int i, int j) {
		return ctr_fun(data_[i].sum_gradients, data_[i].sum_hessians) < ctr_fun(data_[j].sum_gradients, data_[j].sum_hessians);
	      });

    std::vector<int> find_direction(1, 1);
    std::vector<int> start_position(1, 0);
    find_direction.push_back(-1);
    start_position.push_back(used_bin - 1);
    const int max_num_cat = std::min(meta_->config->max_cat_threshold, (used_bin + 1) / 2);

    is_splittable_ = false;
    for (size_t out_i = 0; out_i < find_direction.size(); ++out_i) {
      auto dir = find_direction[out_i];
      auto start_pos = start_position[out_i];
      data_size_t min_data_per_group = meta_->config->min_data_per_group;
      data_size_t cnt_cur_group = 0;
      double sum_left_gradient = 0.0f;
      double sum_left_hessian = kEpsilon;
      data_size_t left_count = 0;
      for (int i = 0; i < used_bin && i < max_num_cat; ++i) {
	auto t = sorted_idx[start_pos];
	start_pos += dir;

	sum_left_gradient += data_[t].sum_gradients;
	sum_left_hessian += data_[t].sum_hessians;
	left_count += data_[t].cnt;
	cnt_cur_group += data_[t].cnt;

	if (left_count < meta_->config->min_data_in_leaf
	    || sum_left_hessian < meta_->config->min_sum_hessian_in_leaf) continue;
	data_size_t right_count = num_data - left_count;
	if (right_count < meta_->config->min_data_in_leaf || right_count < min_data_per_group) break;

	double sum_right_hessian = sum_hessian - sum_left_hessian;
	if (sum_right_hessian < meta_->config->min_sum_hessian_in_leaf) break;

	if (cnt_cur_group < min_data_per_group) continue;

	cnt_cur_group = 0;

	double sum_right_gradient = sum_gradient - sum_left_gradient;
	double current_gain = GetSplitGains(sum_left_gradient, sum_left_hessian, sum_right_gradient, sum_right_hessian,
					    meta_->config->lambda_l1, l2, meta_->config->max_delta_step,
					    min_constraint, max_constraint, 0);
	if (current_gain <= min_gain_shift) continue;
	is_splittable_ = true;
	if (current_gain > best_gain) {
	  best_left_count = left_count;
	  best_sum_left_gradient = sum_left_gradient;
	  best_sum_left_hessian = sum_left_hessian;
	  best_threshold = i;
	  best_gain = current_gain;
	  best_dir = dir;
	}
      }
    }
  }

  if (is_splittable_) {
    output->left_output = CalculateSplittedLeafOutput(best_sum_left_gradient, best_sum_left_hessian,
						      meta_->config->lambda_l1, l2, meta_->config->max_delta_step,
						      min_constraint, max_constraint);
    output->left_count = best_left_count;
    output->left_sum_gradient = best_sum_left_gradient;
    output->left_sum_hessian = best_sum_left_hessian - kEpsilon;
    output->right_output = CalculateSplittedLeafOutput(sum_gradient - best_sum_left_gradient,
						       sum_hessian - best_sum_left_hessian,
						       meta_->config->lambda_l1, l2, meta_->config->max_delta_step,
						       min_constraint, max_constraint);
    output->right_count = num_data - best_left_count;
    output->right_sum_gradient = sum_gradient - best_sum_left_gradient;
    output->right_sum_hessian = sum_hessian - best_sum_left_hessian - kEpsilon;
    output->gain = best_gain - min_gain_shift;
    if (use_onehot) {
      output->num_cat_threshold = 1;
      output->cat_threshold = std::vector<uint32_t>(1, static_cast<uint32_t>(best_threshold));
    } else {
      output->num_cat_threshold = best_threshold + 1;
      output->cat_threshold = std::vector<uint32_t>(output->num_cat_threshold);
      if (best_dir == 1) {
	for (int i = 0; i < output->num_cat_threshold; ++i) {
	  auto t = sorted_idx[i];
	  output->cat_threshold[i] = t;
	}
      } else {
	for (int i = 0; i < output->num_cat_threshold; ++i) {
	  auto t = sorted_idx[used_bin - 1 - i];
	  output->cat_threshold[i] = t;
	}
      }
    }
    output->monotone_type = 0;
    output->min_constraint = min_constraint;
    output->max_constraint = max_constraint;
  }
}

void FeatureHistogram::GatherInfoForThresholdNumerical(double sum_gradient, double sum_hessian,
				     uint32_t threshold, data_size_t num_data,
				     bool default_left, SplitInfo *output) {
  double gain_shift = GetLeafSplitGain(sum_gradient, sum_hessian,
				       meta_->config->lambda_l1, meta_->config->lambda_l2,
				       meta_->config->max_delta_step);
  double min_gain_shift = gain_shift + meta_->config->min_gain_to_split;

  // do stuff here
  const int8_t bias = meta_->bias;

  double sum_right_gradient = 0.0f;
  double sum_right_hessian = kEpsilon;
  data_size_t right_count = 0, left_count = 0;
  double sum_left_gradient = 0.0;
  double sum_left_hessian = kEpsilon;

  // set values
  bool use_na_as_missing = false;
  bool skip_default_bin = false;
  if (meta_->missing_type == MissingType::Zero) {
    skip_default_bin = true;
  } else if (meta_->missing_type == MissingType::NaN) {
    use_na_as_missing = true;
  }

  if(default_left){
    const int t_end = meta_->num_bin - 2 - bias;
    int t = 0;
    if (use_na_as_missing && bias == 1) {
      sum_left_gradient = sum_gradient;
      sum_left_hessian = sum_hessian - kEpsilon;
      left_count = num_data;
      for (int i = 0; i < meta_->num_bin - bias; ++i) {
	sum_left_gradient -= data_[i].sum_gradients;
	sum_left_hessian -= data_[i].sum_hessians;
	left_count -= data_[i].cnt;
      }
      t = -1;
    }
    for(; t <= t_end; ++t){
      if (static_cast<uint32_t>(t + bias) > threshold) { break; }
      if (skip_default_bin && (t + bias) == static_cast<int>(meta_->default_bin)) { continue; }
      if (t >= 0) {
	sum_left_gradient += data_[t].sum_gradients;
	sum_left_hessian += data_[t].sum_hessians;
	left_count += data_[t].cnt;
      }
    }
    sum_right_gradient = sum_gradient - sum_left_gradient;
    sum_right_hessian = sum_hessian - sum_left_hessian;
    right_count = num_data - left_count;
  } else {
    int t = meta_->num_bin - 1 - bias - use_na_as_missing;
    const int t_end = 1 - bias;

    // from right to left, and we don't need data in bin0
    for (; t >= t_end; --t) {
      if (static_cast<uint32_t>(t + bias) < threshold) { break; }

      // need to skip default bin
      if (skip_default_bin && (t + bias) == static_cast<int>(meta_->default_bin)) { continue; }

      sum_right_gradient += data_[t].sum_gradients;
      sum_right_hessian += data_[t].sum_hessians;
      right_count += data_[t].cnt;
      std::cout << "DEBUG: " << t << " g " << sum_right_gradient << " h " << sum_right_hessian << std::endl;
    }
    sum_left_gradient = sum_gradient - sum_right_gradient;
    sum_left_hessian = sum_hessian - sum_right_hessian;
    left_count = num_data - right_count;
  }

  double current_gain = GetLeafSplitGain(sum_left_gradient, sum_left_hessian,
					 meta_->config->lambda_l1, meta_->config->lambda_l2,
					 meta_->config->max_delta_step)
    + GetLeafSplitGain(sum_right_gradient, sum_right_hessian,
		       meta_->config->lambda_l1, meta_->config->lambda_l2,
		       meta_->config->max_delta_step);

  // gain with split is worse than without split
  if (std::isnan(current_gain) || current_gain <= min_gain_shift) {
    output->gain = kMinScore;
    Log::Warning("'Forced Split' will be ignored since the gain getting worse. ");
    return;
  }

  // update split information
  output->threshold = threshold;
  output->left_output = CalculateSplittedLeafOutput(sum_left_gradient, sum_left_hessian,
						    meta_->config->lambda_l1, meta_->config->lambda_l2,
						    meta_->config->max_delta_step);
  output->left_count = left_count;
  output->left_sum_gradient = sum_left_gradient;
  output->left_sum_hessian = sum_left_hessian - kEpsilon;
  output->right_output = CalculateSplittedLeafOutput(sum_gradient - sum_left_gradient,
						     sum_hessian - sum_left_hessian,
						     meta_->config->lambda_l1, meta_->config->lambda_l2,
						     meta_->config->max_delta_step);
  output->right_count = num_data - left_count;
  output->right_sum_gradient = sum_gradient - sum_left_gradient;
  output->right_sum_hessian = sum_hessian - sum_left_hessian - kEpsilon;
  output->gain = current_gain;
  output->gain -= min_gain_shift;
  output->default_left = default_left;
}

void FeatureHistogram::GatherInfoForThresholdCategorical(double sum_gradient, double sum_hessian,
					  std::vector<uint32_t> threshold, data_size_t num_data, SplitInfo *output) {
  // get SplitInfo for a given one-hot categorical split.
  output->default_left = false;
  double gain_shift = GetLeafSplitGain(
				       sum_gradient, sum_hessian,
				       meta_->config->lambda_l1, meta_->config->lambda_l2,
				       meta_->config->max_delta_step);
  double min_gain_shift = gain_shift + meta_->config->min_gain_to_split;
  bool is_full_categorical = meta_->missing_type == MissingType::None;
  int used_bin = meta_->num_bin - 1 + is_full_categorical;
  for(auto t : threshold){
    if (t >= static_cast<uint32_t>(used_bin)) {
      output->gain = kMinScore;
      Log::Warning("Invalid categorical threshold split");
      return;
    }
  }

  double l2 = meta_->config->lambda_l2;
  data_size_t left_count = 0;
  double sum_left_hessian = kEpsilon;
  double sum_left_gradient = 0;
  for(auto t : threshold){
    left_count += data_[t].cnt;
    sum_left_hessian += data_[t].sum_hessians;
    sum_left_gradient += data_[t].sum_gradients;
  }
  data_size_t right_count = num_data - left_count;
  double sum_right_hessian = sum_hessian - sum_left_hessian;
  double sum_right_gradient = sum_gradient - sum_left_gradient;
  // current split gain
  double current_gain = GetLeafSplitGain(sum_right_gradient, sum_right_hessian,
					 meta_->config->lambda_l1, l2,
					 meta_->config->max_delta_step)
    + GetLeafSplitGain(sum_left_gradient, sum_right_hessian,
		       meta_->config->lambda_l1, l2,
		       meta_->config->max_delta_step);
  if (std::isnan(current_gain) || current_gain <= min_gain_shift) {
    output->gain = kMinScore;
    Log::Warning("'Forced Split' will be ignored since the gain getting worse. ");
    return;
  }

  output->left_output = CalculateSplittedLeafOutput(sum_left_gradient, sum_left_hessian,
						    meta_->config->lambda_l1, l2,
						    meta_->config->max_delta_step);
  output->left_count = left_count;
  output->left_sum_gradient = sum_left_gradient;
  output->left_sum_hessian = sum_left_hessian - kEpsilon;
  output->right_output = CalculateSplittedLeafOutput(sum_right_gradient, sum_right_hessian,
						     meta_->config->lambda_l1, l2,
						     meta_->config->max_delta_step);
  output->right_count = right_count;
  output->right_sum_gradient = sum_gradient - sum_left_gradient;
  output->right_sum_hessian = sum_right_hessian - kEpsilon;
  output->gain = current_gain - min_gain_shift;
  output->num_cat_threshold = threshold.size();
  output->cat_threshold = threshold;
}

void FeatureHistogram::FindBestThresholdSequence(double sum_gradient, double sum_hessian, data_size_t num_data, double min_constraint, double max_constraint,
			       double min_gain_shift, SplitInfo* output, int dir, bool skip_default_bin, bool use_na_as_missing) {
  const int8_t bias = meta_->bias;

  double best_sum_left_gradient = NAN;
  double best_sum_left_hessian = NAN;
  double best_gain = kMinScore;
  data_size_t best_left_count = 0;
  uint32_t best_threshold = static_cast<uint32_t>(meta_->num_bin);

  if (dir == -1) {
    double sum_right_gradient = 0.0f;
    double sum_right_hessian = kEpsilon;
    data_size_t right_count = 0;

    int t = meta_->num_bin - 1 - bias - use_na_as_missing;
    const int t_end = 1 - bias;

    // from right to left, and we don't need data in bin0
    for (; t >= t_end; --t) {
      // need to skip default bin
      if (skip_default_bin && (t + bias) == static_cast<int>(meta_->default_bin)) { continue; }

      sum_right_gradient += data_[t].sum_gradients;
      sum_right_hessian += data_[t].sum_hessians;
      right_count += data_[t].cnt;
      // if data not enough, or sum hessian too small
      if (right_count < meta_->config->min_data_in_leaf
	  || sum_right_hessian < meta_->config->min_sum_hessian_in_leaf) continue;
      data_size_t left_count = num_data - right_count;
      // if data not enough
      if (left_count < meta_->config->min_data_in_leaf) break;

      double sum_left_hessian = sum_hessian - sum_right_hessian;
      // if sum hessian too small
      if (sum_left_hessian < meta_->config->min_sum_hessian_in_leaf) break;

      double sum_left_gradient = sum_gradient - sum_right_gradient;
      // current split gain
      double current_gain = GetSplitGains(sum_left_gradient, sum_left_hessian, sum_right_gradient, sum_right_hessian,
					  meta_->config->lambda_l1, meta_->config->lambda_l2, meta_->config->max_delta_step,
					  min_constraint, max_constraint, meta_->monotone_type);
      // gain with split is worse than without split
      if (current_gain <= min_gain_shift) continue;

      // mark to is splittable
      is_splittable_ = true;
      // better split point
      if (current_gain > best_gain) {
	best_left_count = left_count;
	best_sum_left_gradient = sum_left_gradient;
	best_sum_left_hessian = sum_left_hessian;
	// left is <= threshold, right is > threshold.  so this is t-1
	best_threshold = static_cast<uint32_t>(t - 1 + bias);
	best_gain = current_gain;
      }
    }
  } else {
    double sum_left_gradient = 0.0f;
    double sum_left_hessian = kEpsilon;
    data_size_t left_count = 0;

    int t = 0;
    const int t_end = meta_->num_bin - 2 - bias;

    if (use_na_as_missing && bias == 1) {
      sum_left_gradient = sum_gradient;
      sum_left_hessian = sum_hessian - kEpsilon;
      left_count = num_data;
      for (int i = 0; i < meta_->num_bin - bias; ++i) {
	sum_left_gradient -= data_[i].sum_gradients;
	sum_left_hessian -= data_[i].sum_hessians;
	left_count -= data_[i].cnt;
      }
      t = -1;
    }

    for (; t <= t_end; ++t) {
      // need to skip default bin
      if (skip_default_bin && (t + bias) == static_cast<int>(meta_->default_bin)) { continue; }
      if (t >= 0) {
	sum_left_gradient += data_[t].sum_gradients;
	sum_left_hessian += data_[t].sum_hessians;
	left_count += data_[t].cnt;
      }
      // if data not enough, or sum hessian too small
      if (left_count < meta_->config->min_data_in_leaf
	  || sum_left_hessian < meta_->config->min_sum_hessian_in_leaf) continue;
      data_size_t right_count = num_data - left_count;
      // if data not enough
      if (right_count < meta_->config->min_data_in_leaf) break;

      double sum_right_hessian = sum_hessian - sum_left_hessian;
      // if sum hessian too small
      if (sum_right_hessian < meta_->config->min_sum_hessian_in_leaf) break;

      double sum_right_gradient = sum_gradient - sum_left_gradient;
      // current split gain
      double current_gain = GetSplitGains(sum_left_gradient, sum_left_hessian, sum_right_gradient, sum_right_hessian,
					  meta_->config->lambda_l1, meta_->config->lambda_l2, meta_->config->max_delta_step,
					  min_constraint, max_constraint, meta_->monotone_type);
      // gain with split is worse than without split
      if (current_gain <= min_gain_shift) continue;

      // mark to is splittable
      is_splittable_ = true;
      // better split point
      if (current_gain > best_gain) {
	best_left_count = left_count;
	best_sum_left_gradient = sum_left_gradient;
	best_sum_left_hessian = sum_left_hessian;
	best_threshold = static_cast<uint32_t>(t + bias);
	best_gain = current_gain;
      }
    }
  }

  if (is_splittable_ && best_gain > output->gain) {
    // update split information
    output->threshold = best_threshold;
    output->left_output = CalculateSplittedLeafOutput(best_sum_left_gradient, best_sum_left_hessian,
						      meta_->config->lambda_l1, meta_->config->lambda_l2, meta_->config->max_delta_step,
						      min_constraint, max_constraint);
    output->left_count = best_left_count;
    output->left_sum_gradient = best_sum_left_gradient;
    output->left_sum_hessian = best_sum_left_hessian - kEpsilon;
    output->right_output = CalculateSplittedLeafOutput(sum_gradient - best_sum_left_gradient,
						       sum_hessian - best_sum_left_hessian,
						       meta_->config->lambda_l1, meta_->config->lambda_l2, meta_->config->max_delta_step,
						       min_constraint, max_constraint);
    output->right_count = num_data - best_left_count;
    output->right_sum_gradient = sum_gradient - best_sum_left_gradient;
    output->right_sum_hessian = sum_hessian - best_sum_left_hessian - kEpsilon;
    output->gain = best_gain;
    output->default_left = dir == -1;
  }
}



}
