from ordered_set import OrderedSet
from TreeUtils.utils import dict_utils, utils
from TreeUtils.core.graph_df import FEATURE_DATA_KEY
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, LinearSegmentedColormap


class NetworkxHelper:
    def __init__(self, graph, **kwargs):
        self.show_id = dict_utils.get(kwargs, 'show_id', None)
        self.decimals = dict_utils.get(kwargs, 'decimals', 4)

        graph_info = graph.get_graph_info(exclude_df=True)
        score_data = dict_utils.get(graph_info, 'score_data')
        self.graph_info = graph_info
        self.features_name = dict_utils.get_first([kwargs, graph_info], 'features_name')
        self.pred_score_key = dict_utils.get_first([kwargs, score_data], 'pred_score_key')
        self.color_score_key = dict_utils.get_first([kwargs, score_data], 'color_score_key')
        self.objective = dict_utils.get_first([kwargs, graph_info], 'objective')
        self.n_class = dict_utils.get_first([kwargs, graph_info], 'n_class')
        self.fitted = dict_utils.get_first([kwargs, graph_info], '_fitted')

        keys = dict_utils.get(kwargs, 'keys', None)
        self.data_keys = OrderedSet(dict_utils.get(kwargs, 'data_keys', keys))
        self.score_keys = OrderedSet(dict_utils.get(kwargs, 'score_keys', keys))
        if FEATURE_DATA_KEY not in self.data_keys:
            self.data_keys = OrderedSet((FEATURE_DATA_KEY,)) | self.data_keys

        if self.pred_score_key is not None and self.pred_score_key not in self.score_keys:
            self.score_keys = self.score_keys | OrderedSet((self.pred_score_key,))

        self.filled = False
        self.color = None
        self.labels = None
        self.execute_function(graph, **kwargs)

    def execute_function(self, graph, **kwargs):
        self.set_color(graph, **kwargs)
        self.set_label(graph, **kwargs)

    def _set_color_scalar(self, score, **kwargs):
        if not np.all(np.isnan(score)):
            normalize_bound = dict_utils.get(kwargs, 'normalize_bound', lambda x: (np.nanmin(x), np.nanmax(x)))
            reverse_cmap = dict_utils.get(kwargs, 'reverse_cmap', False)
            if callable(normalize_bound):
                normalize_bound = normalize_bound(score)
            self.color = self.default_color_map_scalar(score, normalize_bound, reverse_cmap)
            self.filled = True

    def set_color(self, graph, **kwargs):
        if self.fitted and self.color_score_key is not None:
            score = graph.get_score(self.color_score_key, None)
            score = score.apply(pd.Series)
            valid_score = score.isna().sum(axis=1)
            valid_score = np.all(valid_score.isin([0, score.shape[1]])) and np.any(valid_score == 0)
            if valid_score and 1 <= score.shape[1] <= 2:
                score = score.iloc[:, -1]
                self._set_color_scalar(score, **kwargs)

    def set_label(self, graph, **kwargs):
        feature_data = graph.get_data(FEATURE_DATA_KEY)
        feature_data = feature_data.apply(lambda x: x.to_html(self.features_name))

        data = graph.get_data()
        score = graph.get_score()

        def get_label(key, value, index):
            return f'{key} = {utils.try_round(value, self.decimals)}' if key != FEATURE_DATA_KEY else feature_data[index]

        def get_extra_data(index):
            _data = data[index]
            order_dict = dict_utils.subset_dict(_data, self.data_keys, order_dict=True)
            if order_dict is None:
                return ''
            return [get_label(key, value, index) for key, value in order_dict.items() if value is not None]

        def get_extra_score(index):
            _score = score[index]
            order_dict = dict_utils.subset_dict(_score, self.score_keys, order_dict=True)
            if order_dict is None:
                return ''
            return [f'{key} = {utils.try_round(value, self.decimals)}' for key, value in order_dict.items() if
                    value is not None]

        index_series = data.index.to_series()
        extra_data = index_series.apply(get_extra_data)
        extra_score = index_series.apply(get_extra_score)

        labels = []
        for i, (ex_data, ex_score) in enumerate(zip(extra_data, extra_score)):
            labels_list = []
            if self.show_id:
                labels_list.append(f'id = {i}')
            labels_list += ex_data
            labels_list += ex_score
            labels_list = [label for label in labels_list if label != '']
            # labels_list = [escape(label) for label in labels_list]
            labels.append('<' + '<br/>'.join(labels_list) + '>')
        self.labels = labels

    @staticmethod
    def default_color_map_scalar(score, normalize_bound, reverse_cmap=False):
        min_score, max_score = normalize_bound
        if reverse_cmap:
            cmap = LinearSegmentedColormap.from_list("", ["#4fa8e8", "#fefeff", "#e99254"])
        else:
            cmap = LinearSegmentedColormap.from_list("", ["#e99254", "#fefeff", "#4fa8e8"])
        norm = plt.Normalize(min_score, max_score)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        mid_score= 0.5*(min_score + max_score)
        return np.array([rgb2hex(sm.to_rgba(x if not pd.isnull(x) else mid_score)) for x in score])

