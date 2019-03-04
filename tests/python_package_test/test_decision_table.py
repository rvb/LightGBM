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

        booster = lgbm.Booster(train_set=ds, params={'tree_learner': 'table', 'max_cat_threshold': 100, 'min_data_per_group': 1})
        booster.update()

        tree = booster.get_tree(0,0)
        self.assertEqual(1, tree.num_splits())
        self.assertEqual(0, tree.split_feature(0))
        self.assertTrue(tree.is_categorical(0))
        expected_split_features = [i for i in range(100) if i % 2 == 1]
        self.assertEqual(expected_split_features, tree.categorical_threshold(0))

    def test_small_categorical_twolevel(self):
        N = 1000
        X = np.zeros((N, 2))
        Y = np.zeros((N,))

        for i in range(4):
            X[range(i, N, 4),0] = i

        for i in range(2):
            for j in range(4):
                X[range(4*i+j, N, 8),1] = i

        Y = (X[:,0] == 2)*100+(X[:,1] == 1)*20

        #For this situation, the best approach is to first split on the first feature, moving category 2 to the left,
        #then split on the second, moving category 1 to the left.
        ds = lgbm.Dataset(X, categorical_feature=[0,1]).construct()
        ds.set_label(Y)

        booster = lgbm.Booster(train_set=ds, params={'tree_learner': 'table'})
        booster.update()

        tree = booster.get_tree(0,0)
        self.assertEqual(3, tree.num_splits())
        self.assertTrue(all(tree.is_categorical(i) for i in range(tree.num_splits())))
        self.assertEqual(0, tree.split_feature(0))
        self.assertEqual(1, tree.split_feature(1))
        self.assertEqual(1, tree.split_feature(2))
        self.assertEqual([2], tree.categorical_threshold(0))
        self.assertEqual([1], tree.categorical_threshold(1))
        self.assertEqual([1], tree.categorical_threshold(2))

    def test_large_categorical_twolevel(self):
        N = 1000
        X = np.zeros((N, 2))
        Y = np.zeros((N,))

        for i in range(4):
            X[range(i, N, 4),0] = i

        for i in range(2):
            for j in range(4):
                X[range(4*i+j, N, 8),1] = i

        Y = (X[:,0] == 2)*100+(X[:,1] == 1)*20

        #For this situation, the best approach is to first split on the first feature, moving category 2 to the left,
        #then split on the second, moving category 1 to the left.
        ds = lgbm.Dataset(X, categorical_feature=[0,1]).construct()
        ds.set_label(Y)

        booster = lgbm.Booster(train_set=ds, params={'tree_learner': 'table', 'max_cat_to_onehot': 1})
        booster.update()

        tree = booster.get_tree(0,0)
        self.assertEqual(1, tree.num_splits())
        self.assertTrue(all(tree.is_categorical(i) for i in range(tree.num_splits())))
        self.assertEqual(0, tree.split_feature(0))
        self.assertEqual([2], tree.categorical_threshold(0))
        
    def test_numerical_split(self):
        N = 1000
        X = np.zeros((N, 1))

        X[range(0,N,2),:] = 1
        Y = X[:,0]

        ds = lgbm.Dataset(X).construct()
        ds.set_label(Y)

        booster = lgbm.Booster(train_set=ds, params={'tree_learner': 'table', 'max_cat_to_onehot': 1})
        booster.update()

        tree = booster.get_tree(0,0)
        self.assertEqual(1, tree.num_splits())
        self.assertFalse(tree.is_categorical(0))
        self.assertTrue(abs(tree.threshold(0)) < 1e-6)

    def test_numerical_split_2level(self):
        N = 1000
        X = np.zeros((N, 2))

        X[range(0,N,2),0] = 1
        for i in range(2):
            X[range(2*i,N,4),1] = i
            X[range(2*i+1,N,4),1] = i
        Y = X[:,0]+0.1*X[:,1]

        ds = lgbm.Dataset(X).construct()
        ds.set_label(Y)

        booster = lgbm.Booster(train_set=ds, params={'tree_learner': 'table', 'max_cat_to_onehot': 1})
        booster.update()

        tree = booster.get_tree(0,0)
        self.assertEqual(3, tree.num_splits())
        self.assertFalse(any(tree.is_categorical(i) for i in range(tree.num_splits())))
        self.assertTrue(all(abs(tree.threshold(i) < 1e-6) for i in range(tree.num_splits())))

    def _tree_oblivious(self, tree):
        num_leaves = tree.num_leaves()
        num_levels = 0
        while num_leaves > 1:
            num_levels+= 1
            num_leaves >>= 1
        for i in range(num_levels):
            split_features = []
            split_thresholds = []
            for node in range((1 << i)-1, (1 << (i + 1))- 1):
                split_features.append(tree.split_feature(node))
                split_thresholds.append(tree.threshold(node))
            print('DEBUG: %d: %s, %s' % (i, split_features, split_thresholds))
            if len(set(split_features)) > 1 or len(set(split_thresholds)) > 1:
                return False
        return True
        
    def test_numerical_splits_generate_oblivious_trees(self):
        N = 1000
        M = 100
        K = 10
        for i in range(K):
            X = np.random.random((N, M))
            Y = np.random.random((N,))
            ds = lgbm.Dataset(X).construct()
            ds.set_label(Y)

            booster = lgbm.Booster(train_set=ds, params={'tree_learner': 'table'})
            booster.update()

            tree = booster.get_tree(0,0)
            self.assertTrue(self._tree_oblivious(tree))
