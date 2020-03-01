from TreeUtils.core.graph_df import GRAPH_COL, TREE_LEAF, TYPE_LEFT, TYPE_ROOT, TYPE_RIGHT, ROOT_PARENT, FEATURE_DATA_KEY
from TreeUtils.core.graph_df import BiGraphDF
from TreeUtils.core.extract_graph import Generated, FromNetworkx, FromSklearn, FromLightGradientBoost
from TreeUtils.core.networkx_helper import NetworkxHelper
from TreeUtils.core.data_helper import DataHelper
from TreeUtils.utils import dict_utils
from networkx.drawing.nx_pydot import to_pydot
import networkx as nx
import numpy as np
import pandas as pd
from copy import deepcopy


class BiGraph:
    def __init__(self, df=None, graph_info=None, verify=True, reindex=True, copy_df=False, copy_graph=False, copy=None):
        copy_df = copy_df if copy is None else copy
        copy_graph = copy_graph if copy is None else copy
        self._fitted = False
        self._internally_fitted = False
        self.objective = None
        self.n_class = None
        self.class_name = None
        self.features_name = None
        self.score_data = None
        # self.graph_data = None

        if df is not None:
            df = BiGraphDF.apply(df, verify, reindex, copy_df)
        else:
            root = dict(zip(GRAPH_COL, [0, ROOT_PARENT, TREE_LEAF, TREE_LEAF, TYPE_ROOT, None, None]))
            df = pd.DataFrame([root], columns=GRAPH_COL)

        if graph_info is not None:
            if copy_graph:
                graph_info = deepcopy(graph_info)
            self.update_graph_info(**graph_info)
        self.df = df

    def is_fitted(self):
        if not self._fitted:
            return False
        try:
            key = self.score_data['pred_score_key']
            assert not pd.isnull(key)
            score = self.get_score(key)
            return np.all(~pd.isna(score))
        except:
            return False

    def update_graph(self, score=None, new_score=None, graph_info=None, new_graph_info=None, inplace=False):
        if inplace:
            graph = self
        else:
            graph = self.copy()
        if score is not None:
            graph.replace_score(score)
        if new_score is not None:
            graph.update_score(new_score)

        if graph_info is not None:
            graph.replace_graph_info(**graph_info)
        if new_graph_info is not None:
            graph.update_graph_info(**new_graph_info)
        return graph

    # properties: graph_info
    def update_graph_info(self, **kwargs):
        if 'df' in kwargs.keys():
            raise ValueError('key: df is reserved')
        self.__dict__.update(**kwargs)

    def replace_graph_info(self, **kwargs):
        self.__dict__ = {'df': self.df}
        self.update_graph_info(**kwargs)

    def get_graph_info(self, exclude_df=True):
        graph_info = self.__dict__
        exclude_key = ['df'] if exclude_df else []
        return {key: graph_info[key] for key in graph_info if key not in exclude_key}

    # object method
    def copy(self, copy_df=True, copy_graph=True):
        graph_info = self.get_graph_info(exclude_df=True)
        return BiGraph(self.df, graph_info, verify=False, reindex=False, copy_df=copy_df, copy_graph=copy_graph)

    def __repr__(self):
        return repr(self.df)

    # properties: BiGraphDF
    @property
    def n_nodes(self):
        return BiGraphDF.n_nodes(self.df)

    def n_child(self, node=None):
        return BiGraphDF.n_child(self.df, node)

    def leaf_node(self, node=None):
        return BiGraphDF.leaf_node(self.df, node)

    def has_split(self, node=None):
        return BiGraphDF.has_split(self.df, node)

    def get_data(self, keys=None, default=None, order_dict=False, missing='ignore'):
        return BiGraphDF.get_data(self.df, keys, default, order_dict, missing)

    def get_score(self, keys=None, default=None, order_dict=False, missing='ignore'):
        return BiGraphDF.get_score(self.df, keys, default, order_dict, missing)

    def update_data(self, data_dict, reindex=False):
        return BiGraphDF.update_data(self.df, data_dict, reindex)

    def replace_data(self, new_data, replace_omitted_with_none=False, reindex=False):
        return BiGraphDF.replace_data(self.df, new_data, replace_omitted_with_none, reindex)

    def update_score(self, data_dict):
        return BiGraphDF.update_score(self.df, data_dict)

    def replace_score(self, new_data, replace_omitted_with_none=False):
        return BiGraphDF.replace_score(self.df, new_data, replace_omitted_with_none)

    def get_row_list(self):
        return BiGraphDF.get_row_list(self.df)

    # methods: BiGraphDF
    def reindex_graph(self):
        self.df = BiGraphDF.reindex_dataframe(self.df)
        return self

    def add_child(self, parent_node, left_data=None, right_data=None, left_score=None, right_score=None, reindex=False):
        self.df = BiGraphDF.add_child(self.df, parent_node, left_data, right_data, left_score, right_score, reindex)
        return self

    def remove_redundant(self):
        self.df = BiGraphDF.remove_redundant(self.df)
        return self

    def subgraph(self, root, inplace=False):
        if inplace:
            graph = self
        else:
            graph = self.copy()
        graph.df = BiGraphDF.subgraph(graph.df, root)
        return graph

    def convert_to_leaf(self, nodes, inplace=False):
        graph = self
        if inplace:
            graph = self
        else:
            graph = self.copy()
            graph = self.copy()
        graph.df = BiGraphDF.convert_to_leaf(graph.df, nodes)
        return graph

    # method: Visualization
    def to_networkx(self, **kwargs):
        max_depth = dict_utils.get(kwargs, 'max_depth', np.Inf)
        NetworkxHelperClass = dict_utils.get(kwargs, 'nx_helper', NetworkxHelper)

        if max_depth < 0:
            raise ValueError("max depth should be non negative")

        nx_helper = NetworkxHelperClass(self, **kwargs)

        graph = nx.DiGraph(node={'color': 'black', 'fontname': 'helvetica', 'shape': 'box',
                                 'style': 'filled, ' * nx_helper.filled + 'rounded'},
                           edge={'fontname': 'helvetica'},
                           graph_info=nx_helper.graph_info)
        
        def add_node(node, parent, depth):
            if depth > max_depth:
                return
            if nx_helper.filled and (not pd.isnull(nx_helper.color[node])):
                graph.add_node(node, type=self.df.at[node, 'type'], data=self.df.at[node, 'data'],
                               label=nx_helper.labels[node], fillcolor=nx_helper.color[node])
            else:
                graph.add_node(node, type=self.df.at[node, 'type'], data=self.df.at[node, 'data'],
                               label=nx_helper.labels[node])
            if depth > 1:
                graph.add_edge(parent, node)
            elif depth == 1:
                graph.add_edge(parent, node, label=self.df.at[node, 'type'] == -1)
            if self.n_child(node) == 2:
                add_node(self.df.at[node, 'child_left'], node, depth + 1)
                add_node(self.df.at[node, 'child_right'], node, depth + 1)

        add_node(0, ROOT_PARENT, 0)
        return graph



    def to_pydot(self, **kwargs):
        return to_pydot(self.to_networkx(**kwargs))

    def create_png(self, **kwargs):
        return self.to_pydot(**kwargs).create_png()

    # static method: ExtractGraph
    @staticmethod
    def from_custom_extractor(extractor, tree, **kwargs):
        """
        :param extractor: Custom graph extractor that return df, graph_info
        :param tree: first input of extractor
        :param kwargs: additional kwargs of extractor
            keys used as input of BiGraph init: verify (default False), reindex (default False),
                 copy_df (default False), copy_graph (default False), copy (default None)
        :return: BiGraph
        """
        verify = dict_utils.get(kwargs, 'verify', False)
        reindex = dict_utils.get(kwargs, 'reindex', False)
        copy_df = dict_utils.get(kwargs, 'copy_df', False)
        copy_graph = dict_utils.get(kwargs, 'copy_graph', False)
        copy = dict_utils.get(kwargs, 'copy', None)
        df, graph_info = extractor.extract_graph(tree, kwargs=kwargs)
        return BiGraph(df=df, graph_info=graph_info, verify=verify, reindex=reindex,
                       copy_df=copy_df, copy_graph=copy_graph, copy=copy)

    @staticmethod
    def generate_graph(max_depth, **kwargs):
        """
        :param max_depth: int
        :param kwargs:
            keys used as input of BiGraph init: verify (default False), reindex (default False),
                 copy_df (default False), copy_graph (default False), copy (default None)
        :return: BiGraph
        """
        return BiGraph.from_custom_extractor(Generated, tree=max_depth, **kwargs)

    @staticmethod
    def from_networkx(graph, **kwargs):
        """
        :param graph: networkx graph
        :param kwargs:
            keys used as input of BiGraph init: verify (default False), reindex (default False),
                 copy_df (default False), copy_graph (default False), copy (default None)
        :return: BiGraph
        """
        return BiGraph.from_custom_extractor(FromNetworkx, tree=graph, **kwargs)

    @staticmethod
    def from_sklearn_tree(tree, **kwargs):
        """
        :param tree: sklearn DecisionTree
        :param kwargs:
            keys used as input of BiGraph init: verify (default False), reindex (default False),
                 copy_df (default False), copy_graph (default False), copy (default None)
        :return: BiGraph
        """
        return BiGraph.from_custom_extractor(FromSklearn, tree=tree, **kwargs)

    @staticmethod
    def from_lgboost(tree, **kwargs):
        """
        :param tree: lgb tree
        :param kwargs:
            keys used as input of BiGraph init: verify (default False), reindex (default False),
                 copy_df (default False), copy_graph (default False), copy (default None)
        :return: BiGraph
        """
        return BiGraph.from_custom_extractor(FromLightGradientBoost, tree=tree, **kwargs)

    # method: Analysis
    def fit(self, X, y, **kwargs):
        data_helper = dict_utils.get(kwargs, 'data_helper', DataHelper)
        return data_helper.fit(self, X, y, **kwargs)

    def compare_fit(self, X, y, **kwargs):
        data_helper = dict_utils.get(kwargs, 'data_helper', DataHelper)
        return data_helper.compare_fit(self, X, y, **kwargs)

    def predict(self, X, **kwargs):
        data_helper = dict_utils.get(kwargs, 'data_helper', DataHelper)
        return data_helper.predict(self, X, **kwargs)
