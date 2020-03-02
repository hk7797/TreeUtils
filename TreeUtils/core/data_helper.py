from TreeUtils.core.graph_df import GRAPH_COL, TREE_LEAF, TYPE_LEFT, TYPE_ROOT, TYPE_RIGHT, ROOT_PARENT, FEATURE_DATA_KEY
from TreeUtils.core.exceptions import NotFittedError
from TreeUtils.utils import dict_utils, utils
from TreeUtils.core.score_handler import _default_fit_score_handler, _default_compare_score_handler
import numpy as np
import pandas as pd


class DataHelper:
    @classmethod
    def get_child_index(cls, X, index, feature_data):
        if feature_data.split_criteria == '<=':
            feature, threshold = feature_data.split_feature, feature_data.split_threshold
            left = X[:, feature] <= threshold
            left_index = index[left]
            right_index = index[~left]
            return left_index, right_index
        raise NotImplementedError

    @classmethod
    def get_node_index(cls, graph, X, **kwargs):
        index = dict_utils.get(kwargs, 'index', list(range(X.shape[0])))

        X = utils.make_ndarray(X)
        index = utils.make_ndarray(index, shape=-1)
        if X.shape[0] != index.shape[0]:
            raise ValueError('Mismatch data and index shape')

        result = {}
        root = 0
        verify = dict_utils.get(kwargs, 'verify', True)
        feature_data = graph.get_data(FEATURE_DATA_KEY)
        n_childes = graph.n_child()

        def recurse(node, index):
            result[node] = index
            if n_childes[node] == 2:
                left_index, right_index = cls.get_child_index(X[index, ], index, feature_data[node])
                recurse(graph.df.at[node, 'child_left'], left_index)
                recurse(graph.df.at[node, 'child_right'], right_index)
        recurse(root, index)

        if verify:
            leaf_nodes = np.where(n_childes == 0)[0]
            node_index = dict_utils.subset_dict(result, leaf_nodes)
            result_array = np.concatenate(list(node_index.values()))
            unique, counts = np.unique(result_array, return_counts=True)
            if np.any(counts != 1):
                raise ValueError('Index in multiple leaf')
            if unique.shape[0] != index.shape[0]:
                raise ValueError('Missing index in result')
        return result

    @classmethod
    def _fit(cls, graph, X, y, score_handler, **kwargs):
        if isinstance(y, pd.Series):
            y = y.values
        if not isinstance(X, np.ndarray):
            y = np.array(y).reshape(-1)
        node_index = cls.get_node_index(graph, X, **kwargs)
        population_cal_field = score_handler.population_cal_field_fn(graph, y, **kwargs)
        scores = {}
        leaf_nodes = None

        def recurse(node):
            scores[node] = score_handler.score_fn(graph, node, y[node_index[node]], population_cal_field, scores, **kwargs)
            if graph.n_child(node) == 2:
                recurse(graph.df.at[node, 'child_left'])
                recurse(graph.df.at[node, 'child_right'])
        recurse(0)
        return population_cal_field, scores

    @classmethod
    def fit(cls, graph, X, y, **kwargs):
        score_handler = dict_utils.get(kwargs, 'score_handler', _default_fit_score_handler)
        kwargs['score_handler'] = score_handler
        population_cal_field, scores = cls._fit(graph, X, y, **kwargs)
        score_data = {'pred_score_key': score_handler.pred_score_key,
                      'color_score_key': score_handler.color_score_key,
                      'population_cal_field_key': score_handler.population_cal_field_key,
                      score_handler.population_cal_field_key: population_cal_field}
        return graph.update_graph(score=scores, new_graph_info={'score_data': score_data,
                                                                '_fitted': True, '_internally_fitted': True},
                                  inplace=False)

    @classmethod
    def compare_fit(cls, graph, X_test, y_test, **kwargs):
        score_handler = dict_utils.get(kwargs, 'score_handler', _default_compare_score_handler)
        if not graph.is_fitted():
            raise NotFittedError
        try:
            population_cal_field = graph.score_data[graph.score_data['population_cal_field_key']]
        except:
            population_cal_field = {}
        kwargs['fit_population_cal_field'] = population_cal_field
        kwargs['fit_score_dict'] = graph.get_score().to_dict()
        kwargs['score_handler'] = score_handler
        actual_population_cal_field, scores = cls._fit(graph, X_test, y_test, **kwargs)
        score_data = {'pred_score_key': score_handler.pred_score_key,
                      'color_score_key': score_handler.color_score_key,
                      'population_cal_field_key': score_handler.population_cal_field_key,
                      score_handler.population_cal_field_key: population_cal_field,
                      'actual_population_cal_field_key': actual_population_cal_field}
        return graph.update_graph(score=scores, new_graph_info={'score_data': score_data},
                                  inplace=False)

    @classmethod
    def predict(cls, graph, X, **kwargs):
        node_index = cls.get_node_index(graph, X, **kwargs)
        if not graph.is_fitted():
            raise NotFittedError
        score = graph.get_score(graph.score_data['pred_score_key'])
        pred = np.empty(X.shape[0])
        pred[:] = np.nan
        for node, index in node_index.items():
            pred[index] = score[node]
        return pred
