#ifndef LIGHTGBM_PROFILING_H_
#define LIGHTGBM_PROFILING_H_

#include<chrono>

typedef std::chrono::duration<double, std::milli> duration_millis;

extern duration_millis dataset_load_time;
extern duration_millis dataset_save_time;
extern duration_millis dataset_add_feature_time;
extern duration_millis dataset_add_data_time;

extern duration_millis learner_construct_histogram_time;
extern duration_millis learner_find_splits_from_histograms_time;
#endif
