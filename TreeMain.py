from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from TreeConfig import TreeConfig
import pandas as pd
import numpy as np


tc = TreeConfig().read_config()
model = tc.model
params = tc.params

if model == 'GradientBoosting':
    base_model = GradientBoostingClassifier()
elif model == 'RandomForestClassifier':
    base_model = RandomForestClassifier()
elif model == 'ExtraTreesClassifier':
    base_model = ExtraTreesClassifier()

grid_search = GridSearchCV(estimator=base_model,
                           param_grid=params,
                           scoring='roc_auc',
                           n_jobs=-1 if 'n_jobs' not in params else 1,
                           iid=False,
                           cv=5)