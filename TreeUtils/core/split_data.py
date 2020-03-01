from TreeUtils.utils import utils


class SplitData(object):
    def __init__(self, split_feature=None, split_criteria=None, split_threshold=None, **kwargs):
        attr = {'split_feature': split_feature,
                'split_criteria': split_criteria,
                'split_threshold': split_threshold}
        attr.update(kwargs)
        if ((attr['split_feature'] is None) + (attr['split_threshold'] is None)) == 1:
            raise ValueError('Invalid split')
        if attr['split_feature'] is not None and attr['split_criteria'] is None:
            attr['split_criteria'] = '<='
        for k, v in attr.items():
            object.__setattr__(self, k, v)

    def is_none(self):
        return self.split_feature is None

    def is_split(self):
        return self.split_feature is not None

    def update(self, **kwargs):
        dict_ = self.__dict__
        dict_.update(kwargs)
        return self.__class__(**dict_)

    def to_string(self, features_name=None):
        if self.is_none():
            return ''
        elif features_name is None:
            return str(self)
        return f'{features_name[self.split_feature]} {self.split_criteria} {self.split_threshold}'

    def to_html(self, features_name=None):
        split_criteria = '&le;' if self.split_criteria == '<=' else self.split_criteria
        if self.is_none():
            return ''
        elif features_name is None:
            return f'{self.split_feature} {split_criteria} {utils.try_round(self.split_threshold, 2)}'
        return f'{features_name[self.split_feature]} {split_criteria} {utils.try_round(self.split_threshold, 2)}'

    def __getattribute__(self, item):
        result = super(SplitData, self).__getattribute__(item)
        if item == '__dict__':
            return dict(**result)
        return result

    def __setattr__(self, *args):
        """Disables setting attributes via
        item.prop = val or item['prop'] = val
        """
        raise TypeError('Immutable objects cannot have properties set after init')

    def __delattr__(self, *args):
        """Disables deleting properties"""
        raise TypeError('Immutable objects cannot have properties deleted')

    def dict(self):
        return {'split_feature': self.split_feature,
                'split_criteria': self.split_criteria,
                'split_threshold': self.split_threshold}

    def __str__(self):
        if self.is_none():
            return 'None'
        # return f'{self.split_feature} {self.split_criteria} {utils.try_round(self.split_threshold, 2)}'
        return str(self.dict())

    def __repr__(self):
        return repr(self.dict())