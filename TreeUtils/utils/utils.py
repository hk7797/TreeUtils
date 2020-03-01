import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame
from ordered_set import OrderedSet
from collections.abc import Iterable


def color_brew(n):
    """Generate n colors with equally spaced hues.
    Parameters
    ----------
    n : int
        The number of colors required.
    Returns
    -------
    color_list : list, length n
        numpy array of n color of form (R, G, B).
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360. / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [(c, x, 0),
               (x, c, 0),
               (0, c, x),
               (0, x, c),
               (x, 0, c),
               (c, 0, x),
               (c, x, 0)]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))),
               (int(255 * (g + m))),
               (int(255 * (b + m)))]
        color_list.append(rgb)

    return np.array(color_list)


def flatten(items, split_main_str=False, split_inner_str=False):
    """Yield items from any nested iterable"""
    def recurse(items):
        for x in items:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                for sub_x in recurse(x):
                    yield sub_x
            elif isinstance(x, (str, bytes)) and split_inner_str:
                for sub_x in x:
                    yield sub_x
            else:
                yield x
    if isinstance(items, Iterable):
        if isinstance(items, (str, bytes)) and not split_main_str:
            items = [items]
        for item in recurse(items):
            yield item
    else:
        yield items


def flatten_list(items, split_main_str=False, split_inner_str=False):
    """return flatten list from any nested iterable"""
    return list(flatten(items, split_main_str, split_inner_str))


def order_set_diff(a, b, order_set=False):
    """
    :param a: set a
    :param b: set b
    :param order_set: if true then return order set else list
    :return: set a - set b
    """
    if order_set:
        return OrderedSet(a) - b
    return [x for x in a if a in b]


def try_round(x, decimals):
    """ return rounded number if possible """
    try:
        x = np.round(x, decimals)
    except:
        pass
    return x


def is_integer(x):
    try:
        if not float(x).is_integer():
            return False
    except:
        return False
    return True


def make_ndarray(data, shape=None):
    if isinstance(data, NDFrame):
        data = data.values
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if shape is not None:
        data = data.reshape(shape)
    return data
