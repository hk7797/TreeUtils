from TreeUtils.core.split_data import SplitData
from TreeUtils.utils import dict_utils, utils
import numpy as np
import pandas as pd
import warnings

GRAPH_COL = ['depth', 'parent', 'child_left', 'child_right', 'type', 'data', 'score']
TREE_LEAF = -1
TYPE_LEFT, TYPE_ROOT, TYPE_RIGHT = -1, 0, 1
ROOT_PARENT = 0
FEATURE_DATA_KEY = 'feature_data'


class BiGraphDF:
    @staticmethod
    def copy(df, copy=True):
        if copy:
            df = df.copy()
        return df

    @staticmethod
    def apply(df, verify=True, reindex=True, copy=False):
        df = BiGraphDF.copy(df, copy)
        if verify:
            df = BiGraphDF.verify_dataframe(df)
        if reindex:
            df = BiGraphDF.reindex_dataframe(df)
        return df

    @staticmethod
    def n_nodes(df):
        return df.shape[0]

    @staticmethod
    def n_child(df, node=None):
        if node is None:
            return (df[['child_left', 'child_right']] != TREE_LEAF).sum(axis=1)
        return int(df.at[node, 'child_left'] != TREE_LEAF) + int(df.at[node, 'child_right'] != TREE_LEAF)

    @staticmethod
    def leaf_node(df, node=None):
        return BiGraphDF.n_child(df, node) == 0

    @staticmethod
    def has_split(df, node=None):
        if node is not None:
            split = dict_utils.get(df.at[node, 'data'], key=FEATURE_DATA_KEY)
            if split is None or split.is_none():
                return False
        splits = BiGraphDF.get_data(df, keys=FEATURE_DATA_KEY)
        return splits.apply(lambda x: x is not None and x.is_split())

    @staticmethod
    def get_data(df, keys=None, default=None, order_dict=False, missing='ignore'):
        if keys is None:
            return df['data']
        keys = utils.flatten_list(keys)
        if len(keys) == 1:
            series = df['data'].apply(lambda x: dict_utils.get(x, keys[0], default))
        else:
            series = df['data'].apply(lambda x: dict_utils.subset_dict(x, keys, default, order_dict, missing))
            series.rename(keys, inplace=True)
        return series

    @staticmethod
    def get_score(df, keys=None, default=None, order_dict=False, missing='ignore'):
        if keys is None:
            return df['score']
        keys = utils.flatten_list(keys)
        if len(keys) == 1:
            series = df['score'].apply(lambda x: dict_utils.get(x, keys[0], default))
        else:
            series = df['score'].apply(lambda x: dict_utils.subset_dict(x, keys, default, order_dict, missing))
            series.rename(keys, inplace=True)
        return series

    @staticmethod
    def update_data(df, data_dict, reindex=False):
        leaf_nodes = BiGraphDF.leaf_node(df, node=None)
        for node, data in data_dict.items():
            if node < BiGraphDF.n_nodes(df):
                df.at[node, 'data'] = dict_utils.update(df.at[node, 'data'], data)
                if dict_utils.has_key(df.at[node, 'data'], FEATURE_DATA_KEY) and leaf_nodes[node]:
                    df = BiGraphDF.add_child(df, node)
        if reindex:
            df = BiGraphDF.reindex_dataframe(df)
        return df

    @staticmethod
    def replace_data(df, new_data, replace_omitted_with_none=False, reindex=False):
        if replace_omitted_with_none:
            df['data'] = None
        else:
            for node, data in new_data.items():
                if node < BiGraphDF.n_nodes(df):
                    df.at[node, 'data'] = None
        df = BiGraphDF.update_data(df, new_data, reindex)
        return df

    @staticmethod
    def update_score(df, score_dict):
        for node, data in score_dict.items():
            if node < BiGraphDF.n_nodes(df):
                df.at[node, 'score'] = dict_utils.update(df.at[node, 'score'], data)
        return df

    @staticmethod
    def replace_score(df, new_score, replace_omitted_with_none=False):
        if replace_omitted_with_none:
            df['score'] = None
        else:
            for node, data in new_score.items():
                if node < BiGraphDF.n_nodes(df):
                    df.at[node, 'score'] = None
        df = BiGraphDF.update_score(df, new_score)
        return df

    @staticmethod
    def get_row_list(df):
        return df.to_dict('records')

    @staticmethod
    def sort_dataframe(df):
        return df.sort_values(by=['depth', 'parent', 'type'])

    @staticmethod
    def reindex_dataframe(df):
        df = BiGraphDF.sort_dataframe(df)
        index_map = dict(zip(df.index, range(df.shape[0])))
        df = df.reset_index(drop=True)
        col_to_remap = ['parent', 'child_left', 'child_right']
        df.loc[:, col_to_remap] = df.loc[:, col_to_remap].replace(index_map)
        return df

    @staticmethod
    def verify_dataframe(df, warn=True):
        if not np.all(df.columns == GRAPH_COL):
            raise ValueError('invalid graph columns')
        if df.shape[0] == 0:
            raise ValueError('Empty Dataframe')
        df = BiGraphDF.sort_dataframe(df)
        if df.at[0, 'parent'] != ROOT_PARENT:
            raise ValueError(f'could not able to find root, set parent of root node = {ROOT_PARENT}')
        if df.at[0, 'type'] != 0:
            raise ValueError(f'invalid type of root node = {ROOT_PARENT}')
        node_covered = []

        # noinspection PyPep8
        def verify(node, parent, depth):
            if node in node_covered:
                raise ValueError(f'Cycle in node {node}')
            node_covered.append(node)
            if df.at[node, 'parent'] != parent:
                raise ValueError(f'Invalid Parent Child Assignment between node {node} and {parent}')
            if df.at[node, 'depth'] != depth:
                raise ValueError(f'Invalid depth node {node}')
            if BiGraphDF.n_child(df, node) == 1:
                raise ValueError(f'node {node} has only one child')
            if df.at[node, 'child_left'] != TREE_LEAF:
                left_id, right_id = df.at[node, 'child_left'], df.at[node, 'child_right']
                if not (utils.is_integer(left_id) and utils.is_integer(right_id)):
                    raise ValueError(f'Invalid child of node {node}')
                if df.at[left_id, 'type'] != TYPE_LEFT:
                    raise ValueError(f'invalid type of node {left_id}')
                if df.at[right_id, 'type'] != TYPE_RIGHT:
                    raise ValueError(f'invalid type of node {right_id}')
                left_id, right_id = int(left_id), int(right_id)
                verify(left_id, node, depth + 1)
                verify(right_id, node, depth + 1)

        verify(0, ROOT_PARENT, 0)
        if len(node_covered) != df.shape[0]:
            extra_node = set(df.index) - set(node_covered)
            if warn:
                warnings.warn(f'FOUND REDUNDANT NODES: Deleting nodes {extra_node}', stacklevel=2)
            df = df.drop(extra_node, axis=0)
        return df

    @staticmethod
    def verify_and_reindex_dataframe(df, warn=True):
        df = BiGraphDF.verify_dataframe(df, warn)
        df = BiGraphDF.reindex_dataframe(df)
        return df

    @staticmethod
    def add_child(df, parent_node, left_data=None, right_data=None, left_score=None, right_score=None, reindex=False):
        n_nodes = BiGraphDF.n_nodes(df)
        if parent_node >= n_nodes:
            raise ValueError("parent node doesn't exists")
        row_list = []
        if df.at[parent_node, 'child_left'] != TREE_LEAF:
            raise ValueError('parent node already has children')
        left = dict(
            zip(GRAPH_COL, [df.at[parent_node, 'depth'] + 1, parent_node, TREE_LEAF, TREE_LEAF, TYPE_LEFT,
                            left_data, left_score]))
        right = dict(
            zip(GRAPH_COL, [df.at[parent_node, 'depth'] + 1, parent_node, TREE_LEAF, TREE_LEAF, TYPE_RIGHT,
                            right_data, right_score]))
        row_list.append(left)
        row_list.append(right)
        df.loc[parent_node, ['child_left', 'child_right']] = n_nodes, n_nodes + 1
        df = df.append(pd.DataFrame(row_list), sort=False, ignore_index=True)
        if reindex:
            df = BiGraphDF.reindex_dataframe(df)
        return df

    @staticmethod
    def remove_redundant(df):
        has_feature = df['data'].apply(lambda x: dict_utils.has_key(x, FEATURE_DATA_KEY)).values
        if not has_feature[0]:
            raise ValueError("Root doesn't have feature data.")
        df.loc[~has_feature, ['child_left', 'child_right']] = TREE_LEAF
        df = df.loc[df['parent'].isin(np.append(np.where(has_feature)[0], ROOT_PARENT)), :]
        df = BiGraphDF.verify_and_reindex_dataframe(df, True)
        return df

    @staticmethod
    def subgraph(df, root):
        df.loc[0] = df.loc[root]
        df.at[0, 'parent'] = ROOT_PARENT
        df.at[0, 'type'] = TYPE_ROOT
        df['depth'] = (df['depth'] - df['depth'].iloc[0]).clip(lower=0)
        df.at[df.at[0, 'child_left'], 'parent'] = 0
        df.at[df.at[0, 'child_right'], 'parent'] = 0
        df = BiGraphDF.verify_and_reindex_dataframe(df, False)
        return df

    @staticmethod
    def convert_to_leaf(df, nodes):
        df.loc[nodes, ['child_left', 'child_right']] = TREE_LEAF
        nodes = utils.flatten_list(nodes)
        for node in nodes:
            dict_utils.pop(df.at[node, 'data'], FEATURE_DATA_KEY)
        df = BiGraphDF.verify_and_reindex_dataframe(df, False)
        return df
