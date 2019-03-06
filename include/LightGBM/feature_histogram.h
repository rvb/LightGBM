#ifndef LIGHTGBM_FEATURE_HISTOGRAM_H_
#define LIGHTGBM_FEATURE_HISTOGRAM_H_

#include <LightGBM/split_info.h>

#include <LightGBM/utils/array_args.h>
#include <LightGBM/dataset.h>

#include <cstring>
#include <cmath>
#include <vector>

namespace LightGBM {

class FeatureMetainfo {
 public:
  int num_bin;
  MissingType missing_type;
  int8_t bias = 0;
  uint32_t default_bin;
  int8_t monotone_type;
  double penalty;
  /*! \brief pointer of tree config */
  const Config* config;
  BinType bin_type;
};
/*!
* \brief FeatureHistogram is used to construct and store a histogram for a feature.
*/
class FeatureHistogram {
 public:
  FeatureHistogram() {
    data_ = nullptr;
  }

  ~FeatureHistogram() {
  }

  /*! \brief Disable copy */
  FeatureHistogram& operator=(const FeatureHistogram&) = delete;
  /*! \brief Disable copy */
  FeatureHistogram(const FeatureHistogram&) = delete;

  /*!
  * \brief Init the feature histogram
  * \param feature the feature data for this histogram
  * \param min_num_data_one_leaf minimal number of data in one leaf
  */
  void Init(HistogramBinEntry* data, const FeatureMetainfo* meta);

  HistogramBinEntry* RawData() {
    return data_;
  }
  /*!
  * \brief Subtract current histograms with other
  * \param other The histogram that want to subtract
  */
  void Subtract(const FeatureHistogram& other) {
    for (int i = 0; i < meta_->num_bin - meta_->bias; ++i) {
      data_[i].cnt -= other.data_[i].cnt;
      data_[i].sum_gradients -= other.data_[i].sum_gradients;
      data_[i].sum_hessians -= other.data_[i].sum_hessians;
    }
  }

  void FindBestThreshold(double sum_gradient, double sum_hessian, data_size_t num_data, double min_constraint, double max_constraint,
                         SplitInfo* output) {
    output->default_left = true;
    output->gain = kMinScore;
    find_best_threshold_fun_(sum_gradient, sum_hessian + 2 * kEpsilon, num_data, min_constraint, max_constraint, output);
    output->gain *= meta_->penalty;
  }

  void FindBestThresholdNumerical(double sum_gradient, double sum_hessian, data_size_t num_data, double min_constraint, double max_constraint,
                                  SplitInfo* output);

  void FindBestThresholdCategorical(double sum_gradient, double sum_hessian, data_size_t num_data,
                                    double min_constraint, double max_constraint,
                                    SplitInfo* output);

  void GatherInfoForThreshold(double sum_gradient, double sum_hessian,
                              uint32_t threshold, data_size_t num_data, SplitInfo *output) {
    if (meta_->bin_type == BinType::NumericalBin) {
      GatherInfoForThresholdNumerical(sum_gradient, sum_hessian, threshold,
                                      num_data, false, output);
    } else {
      std::vector<uint32_t> thresholds(1, threshold);
      GatherInfoForThresholdCategorical(sum_gradient, sum_hessian, thresholds,
                                        num_data, output);
    }
  }

  void GatherInfoForThresholdNumerical(double sum_gradient, double sum_hessian,
                                       uint32_t threshold, data_size_t num_data,
                                       bool default_left, SplitInfo *output);

  void GatherInfoForThresholdCategorical(double sum_gradient, double sum_hessian,
                                         std::vector<uint32_t> threshold, data_size_t num_data, SplitInfo *output);

  /*!
  * \brief Binary size of this histogram
  */
  int SizeOfHistgram() const {
    return (meta_->num_bin - meta_->bias) * sizeof(HistogramBinEntry);
  }

  /*!
  * \brief Restore histogram from memory
  */
  void FromMemory(char* memory_data) {
    std::memcpy(data_, memory_data, (meta_->num_bin - meta_->bias) * sizeof(HistogramBinEntry));
  }

  /*!
  * \brief True if this histogram can be splitted
  */
  bool is_splittable() { return is_splittable_; }

  /*!
  * \brief Set splittable to this histogram
  */
  void set_is_splittable(bool val) { is_splittable_ = val; }

  static double ThresholdL1(double s, double l1) {
    const double reg_s = std::max(0.0, std::fabs(s) - l1);
    return Common::Sign(s) * reg_s;
  }

  static double CalculateSplittedLeafOutput(double sum_gradients, double sum_hessians, double l1, double l2, double max_delta_step) {
    double ret = -ThresholdL1(sum_gradients, l1) / (sum_hessians + l2);
    if (max_delta_step <= 0.0f || std::fabs(ret) <= max_delta_step) {
      return ret;
    } else {
      return Common::Sign(ret) * max_delta_step;
    }
  }

  int num_bins(){ return meta_->num_bin;}
  int bias(){return meta_->bias;}
  data_size_t count(int bin){ return data_[bin].cnt;}
  double sum_gradients(int bin){ return data_[bin].sum_gradients;}
  double sum_hessians(int bin){ return data_[bin].sum_hessians;}

  static double GetSplitGains(double sum_left_gradients, double sum_left_hessians,
                              double sum_right_gradients, double sum_right_hessians,
                              double l1, double l2, double max_delta_step,
                              double min_constraint, double max_constraint, int8_t monotone_constraint) {
    double left_output = CalculateSplittedLeafOutput(sum_left_gradients, sum_left_hessians, l1, l2, max_delta_step, min_constraint, max_constraint);
    double right_output = CalculateSplittedLeafOutput(sum_right_gradients, sum_right_hessians, l1, l2, max_delta_step, min_constraint, max_constraint);
    if (((monotone_constraint > 0) && (left_output > right_output)) ||
      ((monotone_constraint < 0) && (left_output < right_output))) {
      return 0;
    }
    return GetLeafSplitGainGivenOutput(sum_left_gradients, sum_left_hessians, l1, l2, left_output)
      + GetLeafSplitGainGivenOutput(sum_right_gradients, sum_right_hessians, l1, l2, right_output);
  }

 private:

  /*!
  * \brief Calculate the output of a leaf based on regularized sum_gradients and sum_hessians
  * \param sum_gradients
  * \param sum_hessians
  * \return leaf output
  */
  static double CalculateSplittedLeafOutput(double sum_gradients, double sum_hessians, double l1, double l2, double max_delta_step,
                                            double min_constraint, double max_constraint) {
    double ret = CalculateSplittedLeafOutput(sum_gradients, sum_hessians, l1, l2, max_delta_step);
    if (ret < min_constraint) {
      ret = min_constraint;
    } else if (ret > max_constraint) {
      ret = max_constraint;
    }
    return ret;
  }

  /*!
  * \brief Calculate the split gain based on regularized sum_gradients and sum_hessians
  * \param sum_gradients
  * \param sum_hessians
  * \return split gain
  */
  static double GetLeafSplitGain(double sum_gradients, double sum_hessians, double l1, double l2, double max_delta_step) {
    double output = CalculateSplittedLeafOutput(sum_gradients, sum_hessians, l1, l2, max_delta_step);
    return GetLeafSplitGainGivenOutput(sum_gradients, sum_hessians, l1, l2, output);
  }

  static double GetLeafSplitGainGivenOutput(double sum_gradients, double sum_hessians, double l1, double l2, double output) {
    const double sg_l1 = ThresholdL1(sum_gradients, l1);
    return -(2.0 * sg_l1 * output + (sum_hessians + l2) * output * output);
  }

  void FindBestThresholdSequence(double sum_gradient, double sum_hessian, data_size_t num_data, double min_constraint, double max_constraint,
                                 double min_gain_shift, SplitInfo* output, int dir, bool skip_default_bin, bool use_na_as_missing);

  const FeatureMetainfo* meta_;
  /*! \brief sum of gradient of each bin */
  HistogramBinEntry* data_;
  // std::vector<HistogramBinEntry> data_;
  bool is_splittable_ = true;

  std::function<void(double, double, data_size_t, double, double, SplitInfo*)> find_best_threshold_fun_;
};
class HistogramPool {
 public:
  /*!
  * \brief Constructor
  */
  HistogramPool() {
    cache_size_ = 0;
    total_size_ = 0;
  }
  /*!
  * \brief Destructor
  */
  ~HistogramPool() {
  }
  /*!
  * \brief Reset pool size
  * \param cache_size Max cache size
  * \param total_size Total size will be used
  */
  void Reset(int cache_size, int total_size) {
    cache_size_ = cache_size;
    // at least need 2 bucket to store smaller leaf and larger leaf
    CHECK(cache_size_ >= 2);
    total_size_ = total_size;
    if (cache_size_ > total_size_) {
      cache_size_ = total_size_;
    }
    is_enough_ = (cache_size_ == total_size_);
    if (!is_enough_) {
      mapper_.resize(total_size_);
      inverse_mapper_.resize(cache_size_);
      last_used_time_.resize(cache_size_);
      ResetMap();
    }
  }
  /*!
  * \brief Reset mapper
  */
  void ResetMap() {
    if (!is_enough_) {
      cur_time_ = 0;
      std::fill(mapper_.begin(), mapper_.end(), -1);
      std::fill(inverse_mapper_.begin(), inverse_mapper_.end(), -1);
      std::fill(last_used_time_.begin(), last_used_time_.end(), 0);
    }
  }

  void DynamicChangeSize(const Dataset* train_data, const Config* config, int cache_size, int total_size) {
    if (feature_metas_.empty()) {
      int num_feature = train_data->num_features();
      feature_metas_.resize(num_feature);
      #pragma omp parallel for schedule(static, 512) if (num_feature >= 1024)
      for (int i = 0; i < num_feature; ++i) {
        feature_metas_[i].num_bin = train_data->FeatureNumBin(i);
        feature_metas_[i].default_bin = train_data->FeatureBinMapper(i)->GetDefaultBin();
        feature_metas_[i].missing_type = train_data->FeatureBinMapper(i)->missing_type();
        feature_metas_[i].monotone_type = train_data->FeatureMonotone(i);
        feature_metas_[i].penalty = train_data->FeaturePenalte(i);
        if (train_data->FeatureBinMapper(i)->GetDefaultBin() == 0) {
          feature_metas_[i].bias = 1;
        } else {
          feature_metas_[i].bias = 0;
        }
        feature_metas_[i].config = config;
        feature_metas_[i].bin_type = train_data->FeatureBinMapper(i)->bin_type();
      }
    }
    uint64_t num_total_bin = train_data->NumTotalBin();
    Log::Info("Total Bins %d", num_total_bin);
    int old_cache_size = static_cast<int>(pool_.size());
    Reset(cache_size, total_size);

    if (cache_size > old_cache_size) {
      pool_.resize(cache_size);
      data_.resize(cache_size);
    }

    OMP_INIT_EX();
    #pragma omp parallel for schedule(static)
    for (int i = old_cache_size; i < cache_size; ++i) {
      OMP_LOOP_EX_BEGIN();
      pool_[i].reset(new FeatureHistogram[train_data->num_features()]);
      data_[i].resize(num_total_bin);
      uint64_t offset = 0;
      for (int j = 0; j < train_data->num_features(); ++j) {
        offset += static_cast<uint64_t>(train_data->SubFeatureBinOffset(j));
        pool_[i][j].Init(data_[i].data() + offset, &feature_metas_[j]);
        auto num_bin = train_data->FeatureNumBin(j);
        if (train_data->FeatureBinMapper(j)->GetDefaultBin() == 0) {
          num_bin -= 1;
        }
        offset += static_cast<uint64_t>(num_bin);
      }
      CHECK(offset == num_total_bin);
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  }

  void ResetConfig(const Config* config) {
    int size = static_cast<int>(feature_metas_.size());
    #pragma omp parallel for schedule(static, 512) if (size >= 1024)
    for (int i = 0; i < size; ++i) {
      feature_metas_[i].config = config;
    }
  }
  /*!
  * \brief Get data for the specific index
  * \param idx which index want to get
  * \param out output data will store into this
  * \return True if this index is in the pool, False if this index is not in the pool
  */
  bool Get(int idx, FeatureHistogram** out) {
    if (is_enough_) {
      *out = pool_[idx].get();
      return true;
    } else if (mapper_[idx] >= 0) {
      int slot = mapper_[idx];
      *out = pool_[slot].get();
      last_used_time_[slot] = ++cur_time_;
      return true;
    } else {
      // choose the least used slot
      int slot = static_cast<int>(ArrayArgs<int>::ArgMin(last_used_time_));
      *out = pool_[slot].get();
      last_used_time_[slot] = ++cur_time_;

      // reset previous mapper
      if (inverse_mapper_[slot] >= 0) mapper_[inverse_mapper_[slot]] = -1;

      // update current mapper
      mapper_[idx] = slot;
      inverse_mapper_[slot] = idx;
      return false;
    }
  }

  /*!
  * \brief Move data from one index to another index
  * \param src_idx
  * \param dst_idx
  */
  void Move(int src_idx, int dst_idx) {
    if (is_enough_) {
      std::swap(pool_[src_idx], pool_[dst_idx]);
      return;
    }
    if (mapper_[src_idx] < 0) {
      return;
    }
    // get slot of src idx
    int slot = mapper_[src_idx];
    // reset src_idx
    mapper_[src_idx] = -1;

    // move to dst idx
    mapper_[dst_idx] = slot;
    last_used_time_[slot] = ++cur_time_;
    inverse_mapper_[slot] = dst_idx;
  }

 private:
  std::vector<std::unique_ptr<FeatureHistogram[]>> pool_;
  std::vector<std::vector<HistogramBinEntry>> data_;
  std::vector<FeatureMetainfo> feature_metas_;
  int cache_size_;
  int total_size_;
  bool is_enough_ = false;
  std::vector<int> mapper_;
  std::vector<int> inverse_mapper_;
  std::vector<int> last_used_time_;
  int cur_time_ = 0;
};

}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_FEATURE_HISTOGRAM_H_
