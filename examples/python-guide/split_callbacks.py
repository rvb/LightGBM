import math
import numpy as np
import lightgbm as lgbm
import ctypes

N = 1000
np.random.seed(42)
X = np.random.random((N,2))
X[:,1] = np.random.randint(0, 4, (N,))
Y = np.random.random((N,)) + X[:,1]*10

ds = lgbm.Dataset(X, categorical_feature=[1]).construct()
ds.set_label(Y)

def split_callback(config, hist, leaf):
    n_bins = hist.num_bins()
    bias = hist.bias()
    best_gain = -math.inf
    best_threshold = -1
    total_g = leaf.sum_gradients()
    total_h = leaf.sum_hessians()
    lambda_l1 = config.lambda_l1()
    lambda_l2 = config.lambda_l2()
    max_delta = config.max_delta_step()
    min_c = leaf.min_constraint()
    max_c = leaf.max_constraint()
    monotone = hist.monotone_constraint()
    left_g = 0
    left_h = 0
    for i in range(n_bins - bias):
        left_g += hist.sum_gradient(i)
        left_h += hist.sum_hessian(i)
        right_g = total_g - left_g
        right_h = total_h - left_h
        gain = lgbm.split_gain(left_g, left_h, right_g, right_h, lambda_l1, lambda_l2, max_delta, min_c, max_c, monotone)
        if gain > best_gain:
            best_gain = gain
            best_threshold = i
    return (True, best_threshold + bias)

def categorical_callback(config, hist, leaf, cats):
    lambda_l1 = config.lambda_l1()
    lambda_l2 = config.lambda_l2()
    max_delta = config.max_delta_step()
    min_c = leaf.min_constraint()
    max_c = leaf.max_constraint()
    monotone = hist.monotone_constraint()
    n_bins = hist.num_bins()
    t_g = leaf.sum_gradients()
    t_h = leaf.sum_hessians()
    best_gain = -math.inf
    best_threshold = None
    for i in range(n_bins):
        g = hist.sum_gradient(i)
        h = hist.sum_hessian(i)
        gain = lgbm.split_gain(g, h, t_g - g, t_h - h, lambda_l1, lambda_l2, max_delta, min_c, max_c, monotone)
        if gain > best_gain:
            best_gain = gain
            best_threshold = i
    if best_threshold is not None:
        cats.add(best_threshold)


booster = lgbm.Booster(train_set=ds)
booster.set_split_callback(split_callback)
booster.set_categorical_split_callback(categorical_callback)

booster.update()
