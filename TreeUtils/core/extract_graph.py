from TreeUtils.core.graph_df import GRAPH_COL, TREE_LEAF, TYPE_LEFT, TYPE_ROOT, TYPE_RIGHT, ROOT_PARENT, FEATURE_DATA_KEY
from TreeUtils.core.split_data import SplitData
from TreeUtils.utils import dict_utils
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
# noinspection PyProtectedMember
from sklearn.tree import _tree, DecisionTreeClassifier, DecisionTreeRegressor


class ExtractGraph(ABC):
    @staticmethod
    @abstractmethod
    def verify(tree, kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def set_defaults(tree, kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_dump(tree, kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_iterator(tree_dump, kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def has_child(tree_iter, node, kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_child(tree_iter, node, kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_data(tree_iter, node, is_leaf, kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_score(tree_iter, node, is_leaf, kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_graph_info(tree, tree_dump, tree_iter, kwargs):
        raise NotImplementedError

    # noinspection PyProtectedMember
    @classmethod
    def extract_graph(cls, tree, kwargs=None):
        kwargs = kwargs if kwargs is not None else {}
        cls.verify(tree, kwargs)
        cls.set_defaults(tree, kwargs)
        copy_df = dict_utils.get(kwargs, 'copy_df', False)

        tree_dump = cls.get_dump(tree, kwargs)
        tree_iter, root_node = cls.get_iterator(tree_dump, kwargs)
        root_is_leaf = not cls.has_child(tree_iter, root_node, kwargs)
        root_data = cls.get_data(tree_iter, root_node, is_leaf=root_is_leaf, kwargs=kwargs)
        root_score = cls.get_score(tree_iter, root_node, is_leaf=root_is_leaf, kwargs=kwargs)
        root = dict(zip(GRAPH_COL, [0, ROOT_PARENT, TREE_LEAF, TREE_LEAF, TYPE_ROOT, root_data, root_score]))
        row_list = [root]

        # noinspection PyShadowingNames
        def recurse(tree_iter, parent, parent_node, depth):  # parent: internal node id; parent_node: tree node id
            if not cls.has_child(tree_iter, parent_node, kwargs):
                return

            left_tree_iter, left_node, right_tree_iter, right_node = cls.get_child(tree_iter, parent_node, kwargs)
            left_is_leaf = not cls.has_child(left_tree_iter, left_node, kwargs)
            right_is_leaf = not cls.has_child(right_tree_iter, right_node, kwargs)

            left_data = cls.get_data(left_tree_iter, left_node, is_leaf=left_is_leaf, kwargs=kwargs)
            left_score = cls.get_score(left_tree_iter, left_node, is_leaf=left_is_leaf, kwargs=kwargs)
            right_data = cls.get_data(right_tree_iter, right_node, is_leaf=right_is_leaf, kwargs=kwargs)
            right_score = cls.get_score(right_tree_iter, right_node, is_leaf=right_is_leaf, kwargs=kwargs)

            left = dict(zip(GRAPH_COL, [depth, parent, TREE_LEAF, TREE_LEAF, TYPE_LEFT, left_data, left_score]))
            right = dict(zip(GRAPH_COL, [depth, parent, TREE_LEAF, TREE_LEAF, TYPE_RIGHT, right_data, right_score]))

            left_id = len(row_list)
            right_id = left_id + 1

            row_list.append(left)
            row_list.append(right)
            parent = row_list[parent]
            parent.update({'child_left': left_id,
                           'child_right': right_id})

            recurse(left_tree_iter, left_id, left_node, depth + 1)
            recurse(right_tree_iter, right_id, right_node, depth + 1)

        recurse(tree_iter, parent=0, parent_node=root_node, depth=1)

        graph_info = cls.get_graph_info(tree, tree_dump, tree_iter, kwargs)
        graph_info = graph_info if graph_info is not None else {}
        df = pd.DataFrame(row_list, columns=GRAPH_COL)
        return df, graph_info


class Generated(ExtractGraph):
    """
    tree: max_depth
    kwargs: not used
    tree_dump: max_depth
    tree_iter: max_depth
    node: depth
    """
    @staticmethod
    def verify(tree, kwargs):
        if tree < 0:
            raise ValueError("max_depth can't be negative")

    @staticmethod
    def set_defaults(tree, kwargs):
        pass

    @staticmethod
    def get_dump(tree, kwargs):
        return tree

    @staticmethod
    def get_iterator(tree_dump, kwargs):
        return tree_dump, 0

    @staticmethod
    def has_child(tree_iter, node, kwargs):
        return node < tree_iter

    @staticmethod
    def get_child(tree_iter, node, kwargs):
        return tree_iter, node + 1, tree_iter, node + 1

    @staticmethod
    def get_data(tree_iter, node, is_leaf, kwargs):
        return None

    @staticmethod
    def get_score(tree_iter, node, is_leaf, kwargs):
        return None

    @staticmethod
    def get_graph_info(tree, tree_dump, tree_iter, kwargs):
        pass


class FromNetworkx(ExtractGraph):
    """
    tree: TODO
    kwargs: TODO
    tree_dump: TODO
    tree_iter: TODO
    node: TODO
    """

    @staticmethod
    def verify(tree, kwargs):
        if len(tree.nodes) == 0:
            raise ValueError('Empty Graph')

    @staticmethod
    def set_defaults(tree, kwargs):
        kwargs['root_id'] = dict_utils.get(kwargs, 'root_id', None)

    @staticmethod
    def get_dump(tree, kwargs):
        return tree

    @staticmethod
    def get_iterator(tree_dump, kwargs):
        root_id = kwargs['root_id']
        root_id = root_id if root_id is not None else tree_dump.nodes.__iter__().__next__()
        return tree_dump, root_id

    @staticmethod
    def has_child(tree_iter, node, kwargs):
        edges = tree_iter.edges(node)
        return len(edges) > 0

    @staticmethod
    def get_child(tree_iter, node, kwargs):
        edges = tree_iter.edges(node)
        if len(edges) == 0:
            return
        if len(edges) != 2:
            raise ValueError('Invalid BiGraph')

        (start1, end1), (start2, end2) = edges
        if start1 != node or start2 != node:
            raise ValueError('Invalid BiGraph')

        type1 = dict_utils.get(tree_iter.nodes[end1], 'type')
        type2 = dict_utils.get(tree_iter.nodes[end2], 'type')
        if type1 not in [-1, 1]:
            type1 = None

        if type1 == 1 or (type1 != -1 and type2 == -1):  # left != end1
            end1, end2 = end2, end1  # Now end1 become left
        return tree_iter, end1, tree_iter, end2

    @staticmethod
    def get_data(tree_iter, node, is_leaf, kwargs):
        return dict_utils.get(tree_iter.nodes[node], 'data')

    @staticmethod
    def get_score(tree_iter, node, is_leaf, kwargs):
        return dict_utils.get(tree_iter.nodes[node], 'score')

    @staticmethod
    def get_graph_info(tree, tree_dump, tree_iter, kwargs):
        pass


class FromSklearn(ExtractGraph):
    """
    tree: TODO
    kwargs: TODO
    tree_dump: TODO
    tree_iter: TODO
    node: TODO
    """

    @staticmethod
    def verify(tree, kwargs):
        if type(tree) not in [DecisionTreeClassifier, DecisionTreeRegressor]:
            raise NotImplementedError('Currently we only support DecisionTreeClassifier & DecisionTreeRegressor from sklearn')

    @staticmethod
    def set_defaults(tree, kwargs):
        kwargs['impurity_key'] = tree.criterion
        kwargs['pred_score_key'] = 'value'
        kwargs['color_score_key'] = 'value'
        if type(tree) == DecisionTreeClassifier:
            objective = 'classification'
        else:
            objective = 'regression'
        kwargs['objective'] = objective

    @staticmethod
    def get_dump(tree, kwargs):
        return tree

    @staticmethod
    def get_iterator(tree_dump, kwargs):
        return tree_dump.tree_, 0

    @staticmethod
    def has_child(tree_iter, node, kwargs):
        return tree_iter.children_left[node] != _tree.TREE_LEAF

    @staticmethod
    def get_child(tree_iter, node, kwargs):
        return tree_iter, tree_iter.children_left[node], tree_iter, tree_iter.children_right[node]

    @staticmethod
    def get_data(tree_iter, node, is_leaf, kwargs):
        if is_leaf:
            return {FEATURE_DATA_KEY: SplitData()}
        # return {FEATURE_DATA_KEY: (tree_iter.feature[node], tree_iter.threshold[node])}
        return {FEATURE_DATA_KEY: SplitData(split_feature=tree_iter.feature[node],
                                            split_criteria='<=',
                                            split_threshold=tree_iter.threshold[node])}

    @staticmethod
    def get_score(tree_iter, node, is_leaf, kwargs):
        impurity_key = kwargs['impurity_key']
        if kwargs['objective'] == 'classification':
            value = tree_iter.value[node][0, :]
            value = value/np.sum(value)
        else:
            value = tree_iter.value[node][0, :][0]
        score = {impurity_key: tree_iter.impurity[node],
                 'sample': tree_iter.n_node_samples[node],
                 'value': value}
        return score

    @staticmethod
    def get_graph_info(tree, tree_dump, tree_iter, kwargs):
        if kwargs['objective'] == 'classification':
            n_class = tree.n_classes_
            class_name = tree.classes_
        else:
            n_class = None
            class_name = None
        n_features = tree.n_features_
        features_name = dict_utils.get(kwargs, 'features_name')
        graph_info = {'_fitted': True,
                      'objective': kwargs['objective'], 'n_class': n_class, 'class_name': class_name,
                      'n_features': n_features, 'features_name': features_name,
                      'score_data': {'pred_score_key': kwargs['pred_score_key'],
                                     'color_score_key': kwargs['color_score_key']}}
        return graph_info


class FromLightGradientBoost(ExtractGraph):
    """
    tree: TODO
    kwargs: TODO
    tree_dump: TODO
    tree_iter: TODO
    node: TODO
    """

    score_keys = ['split_gain', 'internal_weight', 'internal_count', 'internal_value']

    @staticmethod
    def verify(tree, kwargs):
        pass

    @staticmethod
    def set_defaults(tree, kwargs):
        kwargs['tree_index'] = int(dict_utils.get(kwargs, 'tree_index', 0))

    @staticmethod
    def get_dump(tree, kwargs):
        tree_index = kwargs['tree_index']
        model_dump = tree.dump_model()
        
        objective = model_dump['objective']
        if 'regression' in objective:
            kwargs['objective'] = 'regression'
            kwargs['n_class'] = None
            # kwargs['class_name'] = None
        elif 'binary' in objective:
            kwargs['objective'] = 'classification'
            kwargs['n_class'] = 2
            # kwargs['class_name'] = None
        elif 'multiclass' in objective:
            kwargs['objective'] = 'classification'
            kwargs['n_class'] = model_dump['num_class']
            # kwargs['class_name'] = None
        else:
            raise NotImplementedError(
                'Currently we only support regression & classification booster from lightgbm')

        # kwargs['class_name'] = None
        kwargs['n_features'] = model_dump['max_feature_idx']
        tree_dump = None
        for tree_ in model_dump['tree_info']:
            if tree_['tree_index'] == tree_index:
                tree_dump = tree_
        if tree_dump is None:
            raise ValueError('Invalid tree index: not able to find tree with index')
        return tree_dump

    @staticmethod
    def get_iterator(tree_dump, kwargs):
        return tree_dump['tree_structure'], None

    @staticmethod
    def has_child(tree_iter, node, kwargs):
        return 'left_child' in tree_iter.keys()

    @staticmethod
    def get_child(tree_iter, node, kwargs):
        return tree_iter['left_child'], None, tree_iter['right_child'], None

    @staticmethod
    def get_data(tree_iter, node, is_leaf, kwargs):
        if is_leaf:
            return {FEATURE_DATA_KEY: SplitData()}
        return {FEATURE_DATA_KEY: SplitData(split_feature=tree_iter['split_feature'],
                                            split_criteria=tree_iter['decision_type'],
                                            split_threshold=tree_iter['threshold'],
                                            default_left=tree_iter['default_left'],
                                            missing_type=tree_iter['missing_type'])}

    @staticmethod
    def get_score(tree_iter, node, is_leaf, kwargs):
        return dict_utils.subset_dict(tree_iter, FromLightGradientBoost.score_keys)

    @staticmethod
    def get_graph_info(tree, tree_dump, tree_iter, kwargs):
        features_name = dict_utils.get(kwargs, 'features_name')
        class_name = dict_utils.get(kwargs, 'class_name')
        graph_info = {'_fitted': True,
                      'objective': kwargs['objective'], 'n_class': kwargs['n_class'], 'class_name': class_name,
                      'n_features': kwargs['n_features'], 'features_name': features_name,
                      'score_data': {'pred_score_key': None,
                                     'color_score_key': None}}
        return graph_info
