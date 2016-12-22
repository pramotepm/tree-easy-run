from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import os
from TreeConfig import TreeConfig
from ParamLoad import ParamLoad


tc = TreeConfig().read_config()
model = tc.model
tree_params = tc.params

params = ParamLoad().read_config()
df_samples = pd.read_csv(params.input_path, header=None, sep=params.delim)
df_classes = pd.read_csv(params.class_path, header=None)
X = np.asfortranarray(df_samples.values).astype(np.float32)
y = np.asfortranarray(df_classes.values.reshape(1, df_classes.shape[0])).astype(np.int32)[0]

if model == 'GradientBoosting':
    base_model = GradientBoostingClassifier()
elif model == 'RandomForestClassifier':
    base_model = RandomForestClassifier()
elif model == 'ExtraTreesClassifier':
    base_model = ExtraTreesClassifier()
grid_search = GridSearchCV(estimator=base_model,
                           param_grid=tree_params,
                           scoring='f1_micro',
                           n_jobs=-1 if 'n_jobs' not in tree_params else 1,
                           iid=False,
                           cv=params.n_folds)
grid_search.fit(X, y)

print "Best Parameters:"
for k in grid_search.best_params_:
    print '\t%s: %r' % (k, grid_search.best_params_[k])
print "Best Score: %.5f" % grid_search.best_score_

'''
Write a combination of parameters
'''
# excel_count = 0
# while True:
#     excel_count += 1
#     excel_out = os.path.join(params.out_dir, 'params_search_result_%03d.xlsx' % excel_count)
#     if not os.path.exists(excel_out):
#         break
# pd.DataFrame(grid_search.cv_results_).to_excel(excel_out)
csv_param_count = 0
while True:
    csv_param_count += 1
    csv_out = os.path.join(params.out_dir, 'params_search_result_%03d.csv' % csv_param_count)
    if not os.path.exists(csv_out):
        break
pd.DataFrame(grid_search.cv_results_).to_csv(csv_out, index=False, sep='\t')


'''
Write a feature importances
'''
feature_names = map(lambda x: x.strip(), open(os.path.join(params.featu_path)).readlines())
df_feature = pd.DataFrame(data=grid_search.best_estimator_.feature_importances_,
                          index=feature_names)
df_feature.sort_values(by=0, ascending=False, inplace=True)
df_feature.to_csv(os.path.join(params.out_dir, 'feature_importance.csv'), header=False, sep=params.delim)

'''
Write predicted classes for all samples
'''
pred_class = grid_search.best_estimator_.predict(X)
pd.Series(data=pred_class).to_csv(os.path.join(params.out_dir, 'predicted_sample_class.txt'), index=False)
