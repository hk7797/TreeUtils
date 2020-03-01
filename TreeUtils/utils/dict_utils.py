from collections import OrderedDict
from ordered_set import OrderedSet
from TreeUtils.utils import utils


def get(dict_data, key, default=None):
    return dict_data.get(key, default) if dict_data is not None else default


def get_first(dicts, key, default=None):
    if dicts is None or len(dicts) == 0:
        return default
    if has_key(dicts[0], key):
        return get(dicts[0], key, default)
    return get_first(dicts[1:], key, default)


def pop(dict_data, key):
    return dict_data.pop(key, None) if dict_data is not None else None


def subset_dict(dict_data, keys, default=None, order_dict=False, missing='ignore'):
    if dict_data is None or keys is None:
        return None
    keys = utils.flatten_list(keys)
    if missing == 'ignore':
        keys = OrderedSet(keys) & dict_data.keys()
    if missing != 'keep' and len(keys - dict_data.keys()) > 0:
        raise ValueError('Invalid key!')
    if order_dict:
        return OrderedDict([(key, get(dict_data, key, default)) for key in keys])
    return {key: get(dict_data, key, default) for key in keys}


def has_key(dict_data, key):
    return key in dict_data.keys() if dict_data is not None else False


def update(dict_data, update_dict):
    if update_dict is None:
        return dict_data
    dict_data = dict_data if dict_data is not None else {}
    dict_data.update(update_dict)
    return dict_data
