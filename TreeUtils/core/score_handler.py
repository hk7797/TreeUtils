import numpy as np
from TreeUtils.utils import dict_utils


class ScoreHandler:
    def __init__(self, pred_score_key, color_score_key, population_cal_field_key, population_cal_field_fn, score_fn):
        self.pred_score_key = pred_score_key
        self.color_score_key = color_score_key
        self.population_cal_field_key = population_cal_field_key
        self.population_cal_field_fn = population_cal_field_fn
        self.score_fn = score_fn


def default_population_cal_field_fn(graph, y, **kwargs):
    return {}


def default_fit_score_fn(graph, node, y, population_cal_field, score_dict, **kwargs):
    return {'pred_y': np.mean(y)}


def default_compare_score_fn(graph, node, y, population_cal_field, score_dict,
                             fit_population_cal_field, fit_score_dict, **kwargs):
    pred_score_key = graph.score_data['pred_score_key']
    return {'train_y': dict_utils.get(fit_score_dict[node], pred_score_key, np.nan),
            'actual_y': np.mean(y)}


_default_fit_score_handler = ScoreHandler('pred_y', 'pred_y', 'population_cal_field',
                                          default_population_cal_field_fn, default_fit_score_fn)
_default_compare_score_handler = ScoreHandler('train_y', 'train_y', 'train_population_cal_field',
                                              default_population_cal_field_fn, default_compare_score_fn)
