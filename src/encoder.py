import enum
from json import JSONEncoder

import numpy as np
import pandas as pd


class AppEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_json(orient='records')
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, enum.Enum):
            return obj.value
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return JSONEncoder.default(self, obj)
