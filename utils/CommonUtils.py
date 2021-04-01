import datetime
import itertools
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


class CommonUtils:
    @staticmethod
    def orderDict(data, index, **kwargs):
        """
        将字典按key/value排序
        :param data: 待排序的字典
        :param index: 0/1, 0: 按key，1：按value
        :param kwargs: reverse: bool, 是否降序, 默认False
        :return: dict
        """
        _ = sorted(data.items(), key=lambda x: x[index], reverse=kwargs.get('reverse', True))
        return {item[0]: item[1] for item in _}
