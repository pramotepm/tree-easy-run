from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib
from scipy import interp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from TreeConfig import TreeConfig
from ParamLoad import ParamLoad

tc = TreeConfig().read_config()
model = tc.model
tree_params = tc.params

params = ParamLoad().read_config()

if os.path.isfile(params.model_export_path):
    print 'Loading model from [%s]' % params.model_export_path
    best_clf = joblib.load(params.model_export_path)
else:
    print 'Loading input data...'
    df_samples = pd.read_csv(params.input_path, header=None, sep=params.delim)
    df_classes = pd.read_csv(params.true_class_path, header=None)
    X = np.asfortranarray(df_samples.values).astype(np.float32)
    y = np.asfortranarray(df_classes.values.reshape(1, df_classes.shape[0])).astype(np.int32)[0]

    print 'Fitting data to model...'
    k_fold_cv = StratifiedKFold(n_splits=params.n_folds, random_state=0)

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
    Write best prediction model
    '''
    print 'Saving model to [%s]' % params.model_export_path
    best_clf = grid_search.best_estimator_
    joblib.dump(best_clf, params.model_export_path)

    '''
    Write a combination of parameters
    '''
    print 'Writing fitted result...'
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
    print 'Writing feature importances...'
    feature_names = map(lambda x: x.strip(), open(os.path.join(params.feature_name_path)).readlines())
    df_feature = pd.DataFrame(data=best_clf.feature_importances_,
                              index=feature_names)
    df_feature.sort_values(by=0, ascending=False, inplace=True)
    df_feature.to_csv(os.path.join(params.out_dir, 'feature_importance.csv'), header=False, sep=params.delim)

    '''
    Calculate AUC-ROC score & Plot ROC curve
    '''
    print 'Calculating ROC score...'
    # for plotting graph
    fig = plt.figure()
    axe = fig.add_subplot(111)
    axe.set_xlim(xmin=-0.025)
    axe.set_ylim(ymax=1.025)
    axe.plot([0, 1], [0, 1], 'r--')
    # for calculating mean of roc score
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    # train and test with N-fold cross validation (number of lines equal to N-fold)
    for train_idx, test_idx in k_fold_cv.split(X, y):
        best_clf.fit(X[train_idx], y[train_idx])
        best_clf.predict_proba(X[test_idx])
        fp_rate, tp_rate, _ = roc_curve(y[test_idx], best_clf.predict_proba(X[test_idx])[:, 1])
        mean_tpr += interp(mean_fpr, fp_rate, tp_rate)
        mean_tpr[0] = 0.0
        axe.plot(fp_rate, tp_rate, 'b', alpha=0.4, color='gray')
        axe.set_ylabel('True Positive Rate')
        axe.set_xlabel('False Positive Rate')
    mean_tpr /= params.n_folds
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    axe.plot(mean_fpr, mean_tpr, color='black', label='Mean ROC = %0.2f' % mean_auc, lw=2)
    axe.legend(loc='lower right')
    axe.set_title('ROC Curve')
    fig.savefig(os.path.join(params.out_dir, 'roc_curve.pdf'), format='pdf', dpi=300)

'''
Write predicted classes for unseen samples
'''
if params.unseen_sample_path is not None:
    print 'Predicting Unseen data...'
    df_unseen = pd.read_csv(params.unseen_sample_path, header=None, sep=params.delim)
    X_unseen = np.asfortranarray(df_unseen.values).astype(np.float32)
    pred_class = best_clf.predict(X_unseen)
    pd.Series(data=pred_class).to_csv(os.path.join(params.out_dir, 'predicted_unseen_sample.txt'), index=False)
