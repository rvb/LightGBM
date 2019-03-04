import lightgbm as lgbm
import numpy as np
import unittest

class TestBasic(unittest.TestCase):
    def test_basic_categorical_split(self):
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

        booster.save_model('test_categorical_%s.txt' % (learner,), num_iteration=1)

        tree = booster.get_tree(0,0)
        self.assertEqual(1, tree.num_splits())
        self.assertEqual(0, tree.split_feature(0))
        self.assertTrue(tree.is_categorical(0))
        self.assertEqual(tree.categorical_threshold(0), [2])
