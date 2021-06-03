# Tree Utils

* This package makes result of simple decision tree user friendly. 
* It also helps in visualizing the tree output.
* It currently support sklearn, lightgbm.
* It can estimate node level custom score using any data without refitting.
* Compare train and test score at node level. Which can helpful in evaluating the performance and identifying the overfitted node.
* We can modify threshold of node to make result more interpretable and re-calculating the score

## Install
Currently this package is at early stage and not yet available on pypi. To use this either clone this in you project directory or clone in anywhere and add path by using following command.

```python
import sys
sys.path.append('<path>')
```

## Getting Started

Import BiGraph from TreeUtils.graph

```python
from TreeUtils.graph import BiGraph
```

Now with BiGraph you can copy tree structure from sklearn, lightgbm using method `from_sklearn_tree`, `from_lgboost` respectively.
Also, you can generate empty graph using `generate_graph`.

### BiGraph from sklearn
```python
graph = BiGraph.from_sklearn_tree(clf) # Here clf is sklearn decision tree
Image(graph.create_png(features_name=feature_name)) # To visulaize
```
![alt text](Screenshots/sklearn_decision_tree.png 'Sklearn Decision Tree')

Reterive BiGraph internal DataFrame for full data
```python
graph.df
```
|  | depth | parent | child_left | child_right | type | data | score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 1 | 2 | 0 | {'feature_data': {'split_feature': 12, 'split_criteria': '<=', 'split_threshold': -2.0825871229171753}} | {'gini': 0.4999999943111111, 'sample': 37500, 'value': [0.5000533333333333, 0.49994666666666665]} |
| 1 | 1 | 0 | 3 | 4 | -1 | {'feature_data': {'split_feature': 51, 'split_criteria': '<=', 'split_threshold': -1.3576117753982544}} | {'gini': 0.4768391903696647, 'sample': 12401, 'value': [0.6076122893315056, 0.3923877106684945]} |
| 2 | 1 | 0 | 9 | 10 | 1 | {'feature_data': {'split_feature': 52, 'split_criteria': '<=', 'split_threshold': 0.9673941433429718}} | {'gini': 0.4943629538037777, 'sample': 25099, 'value': [0.4469102354675485, 0.5530897645324515]} |
| 3 | 2 | 1 | 5 | 6 | -1 | {'feature_data': {'split_feature': 52, 'split_criteria': '<=', 'split_threshold': 3.68274986743927}} | {'gini': 0.3953342751715758, 'sample': 4085, 'value': [0.7287637698898409, 0.2712362301101591]} |
| 4 | 2 | 1 | 7 | 8 | 1 | {'feature_data': {'split_feature': 9, 'split_criteria': '<=', 'split_threshold': -0.6838884949684143}} | {'gini': 0.4953727707455461, 'sample': 8316, 'value': [0.5481000481000481, 0.4518999518999519]} |
| 5 | 3 | 3 | -1 | -1 | -1 | {'feature_data': None} | {'gini': 0.3395860090944033, 'sample': 3335, 'value': [0.783208395802099, 0.21679160419790106]} |
| 6 | 3 | 3 | -1 | -1 | 1 | {'feature_data': None} | {'gini': 0.49964444444444445, 'sample': 750, 'value': [0.4866666666666667, 0.5133333333333333]} |
| 7 | 3 | 4 | -1 | -1 | -1 | {'feature_data': None} | {'gini': 0.4578563995837669, 'sample': 3565, 'value': [0.6451612903225806, 0.3548387096774194]} |
| 8 | 3 | 4 | -1 | -1 | 1 | {'feature_data': None} | {'gini': 0.4987766924164144, 'sample': 4751, 'value': [0.47526836455483057, 0.5247316354451694]} |
| 9 | 2 | 2 | 11 | 12 | -1 | {'feature_data': {'split_feature': 51, 'split_criteria': '<=', 'split_threshold': -0.6225283443927765}} | {'gini': 0.49893471319709315, 'sample': 14342, 'value': [0.5230790684702273, 0.4769209315297727]} |
| 10 | 2 | 2 | 13 | 14 | 1 | {'feature_data': {'split_feature': 33, 'split_criteria': '<=', 'split_threshold': 0.9380942583084106}} | {'gini': 0.45217078326230953, 'sample': 10757, 'value': [0.3453565120386725, 0.6546434879613275]} |
| 11 | 3 | 9 | -1 | -1 | -1 | {'feature_data': None} | {'gini': 0.46155014673009653, 'sample': 5676, 'value': [0.6386539816772375, 0.36134601832276253]} |
| 12 | 3 | 9 | -1 | -1 | 1 | {'feature_data': None} | {'gini': 0.4944623906964244, 'sample': 8666, 'value': [0.4473805677359797, 0.5526194322640203]} |
| 13 | 3 | 10 | -1 | -1 | -1 | {'feature_data': None} | {'gini': 0.4044053843016141, 'sample': 6550, 'value': [0.2813740458015267, 0.7186259541984733]} |
| 14 | 3 | 10 | -1 | -1 | 1 | {'feature_data': None} | {'gini': 0.49394398471900636, 'sample': 4207, 'value': [0.44497266460660806, 0.555027335393392]} |

Reterive BiGraph info
```python
graph.get_graph_info()
```
{'_fitted': True,
 '_internally_fitted': False,
 'objective': 'classification',
 'n_class': 2,
 'class_name': array([0, 1]),
 'features_name': None,
 'score_data': {'pred_score_key': 'value', 'color_score_key': 'value'},
 'n_features': 100}


### Color Baised on a key
```python
# score keys args in below function to select score in graph. default: identified prediction key
# color_score_key to change color key
Image(graph.create_png(features_name=feature_name, score_keys=['gini','sample', 'value'], color_score_key='gini'))
```

![alt text](Screenshots/tree_with_color_key.png 'Sklearn Decision With Color Key')


### Custom Score
```python
# Defining Handlers

from TreeUtils.core.score_handler import ScoreHandler

def custom_population_cal_field_fn(graph, y, **kwargs):
    return {'Total': y.shape[0], 'Good': np.sum(y), 'Bad': np.sum(y==0)}


def custom_fit_score_fn(graph, node, y, population_cal_field, score_dict, **kwargs):
    pop = np.round(100*y.shape[0]/population_cal_field['Total'], 2)
    return {'PopulationFraction': f'{pop} %', 'Pr_Good': np.mean(y)}

def custom_compare_score_fn(graph, node, y, population_cal_field, score_dict,
                             fit_population_cal_field, fit_score_dict, **kwargs):
    pred_score_key = graph.score_data['pred_score_key']
    Pr_Good_train = dict_utils.get(fit_score_dict[node], pred_score_key, np.nan)
    Pr_Good_test = np.mean(y)
    return {'Pr_Good_train': Pr_Good_train,
            'Pr_Good_test': Pr_Good_test, 
            'error': Pr_Good_train - Pr_Good_test}

custom_fit_score_handler = ScoreHandler('Pr_Good', 'Pr_Good', 'population_cal_field',
                                          custom_population_cal_field_fn, custom_fit_score_fn)

custom_compare_score_handler = ScoreHandler('Pr_Good_train', 'error', 'train_population_cal_field',
                                              custom_population_cal_field_fn, custom_compare_score_fn)
```

Estimating custom score
```python
# fitting graph to evaluate custom score
graph = graph.fit(X_train, y_train, score_handler=custom_fit_score_handler)

# score keys args in below function to select score in graph.
Image(graph.create_png(features_name=feature_name, score_keys=['PopulationFraction', 'Pr_Good']))
```

![alt text](Screenshots/custom_tree.png 'Custom Tree')

## Pydot/NetworkX Object
BiGraph support conversion to Pydot or NetworkX by simplify calling `graph.to_pydot()` or `graph.to_networkx()`.

You can also convert pydot or NetworkX object to bigraph. Using `BiGraph.from_pydot` or `BiGraph.from_networkx`.


## Example

You can check notebook1 for basic implementation.
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hk7797/TreeUtils/master?filepath=notebook1.ipynb)
