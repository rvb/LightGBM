import lightgbm as lgbm
import numpy as np
import unittest

class TestBasic(unittest.TestCase):
    def test_small_categorical_split(self):
        N = 1000
        X = np.zeros((N, 1))
        Y = np.zeros((N,))

        for i in range(4):
            X[range(i, N, 4),:] = i
            Y[range(i, N, 4)] = 100 if i == 2 else 0

        ds = lgbm.Dataset(X, categorical_feature=[0]).construct()
        ds.set_label(Y)

        learner = 'table'
        booster = lgbm.Booster(train_set=ds, params={'tree_learner': learner, 'learning_rate': 1.0})
        booster.update()

        tree = booster.get_tree(0,0)
        self.assertEqual(1, tree.num_splits())
        self.assertEqual(0, tree.split_feature(0))
        self.assertTrue(tree.is_categorical(0))
        self.assertEqual(tree.categorical_threshold(0), [2])

    def test_large_categorical_split(self):
        N = 1000
        X = np.zeros((N, 1))
        Y = np.zeros((N,))

        for i in range(100):
            X[range(i, N, 100),:] = i
            Y[range(i, N, 100)] = 100 if i % 2 == 1 else 0

        ds = lgbm.Dataset(X, categorical_feature=[0]).construct()
        ds.set_label(Y)

        learner = 'table'
        booster = lgbm.Booster(train_set=ds, params={'tree_learner': learner, 'learning_rate': 1.0, 'max_cat_threshold': 100, 'min_data_per_group': 1})
        booster.update()

        tree = booster.get_tree(0,0)
        self.assertEqual(1, tree.num_splits())
        self.assertEqual(0, tree.split_feature(0))
        self.assertTrue(tree.is_categorical(0))
        expected_split_features = [i for i in range(100) if i % 2 == 1]
        self.assertEqual(expected_split_features, tree.categorical_threshold(0))
