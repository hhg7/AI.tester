import argparse
import sys
import re
import os
# the ML libraries take forever to load, so checking for the json file first saves time
parser = argparse.ArgumentParser(description='Pass a file name.')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--file',        required = True)
required.add_argument('--output_stem', required = True)
required.add_argument('--target',      required = True)
optional.add_argument('--categorical', nargs='+', required = False)
optional.add_argument('--drop', nargs='+', required = False)
optional.add_argument('--suptitle', required = False, default = None)
args = parser.parse_args()

if not os.path.isfile(args.file):
	sys.exit(args.file + ' is not a file')

import mlflow
import json
import mlflow.sklearn
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
import xgboost
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, RocCurveDisplay, roc_curve, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score,f1_score,recall_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def ref_to_json_file(data, filename):
	json1=json.dumps(data)
	f = open(filename,"w+")
	print(json1,file=f)

xgb_dir = args.output_stem + '.reg.output/'
if not os.path.isdir(xgb_dir):
	os.mkdir(xgb_dir)
xgb_x_test_dir = xgb_dir + 'X_test/'
xgb_json_dir   = xgb_dir + 'json/'
xgb_result_obj_dir = xgb_dir + 'clf/'
image_dir      = xgb_dir + 'Images/'
for directory in xgb_x_test_dir, xgb_json_dir, xgb_result_obj_dir, image_dir:
	if not os.path.isdir(directory):
		os.mkdir(directory)
fig, axs = plt.subplots(3, 3, layout = 'constrained', sharex = False)
axrow = 0
axcol = 0
# read in the input file
if re.search(r'\.json$', args.file):
 pandasDF = pd.read_json(args.file)
elif re.search(r'\.csv$', args.file):
 pandasDF = pd.read_csv(args.file)
elif re.search(r'\.xlsx$', args.file):
 pandasDF = pd.read_excel(args.file)
if args.drop:
	for col in args.drop:
		pandasDF = pandasDF.drop(columns = [col])
print(pandasDF.columns)
if args.categorical != None:
	for category in args.categorical:
		pandasDF = pd.get_dummies(pandasDF, prefix = [category], columns = [category], drop_first = True)
print(pandasDF.dtypes)
results_data = {}
roc_data = {}
for metric in ('precision_score', 'accuracy', 'roc_auc', 'test_accuracy','test_precision', 'test_recall', 'test_f1', 'test_roc_auc'):
	results_data[metric] = {}
def classifier_wrapper(category_cols, dependent_var, output_stem, method):
  #https://xgboost.readthedocs.io/en/latest/parameter.html
  global axrow
  global axcol
  if method == 'AdaBoostClassifier':
    clf = AdaBoostClassifier()
  elif method == 'DecisionTreeClassifier':
    clf = DecisionTreeClassifier()
  elif method == 'GaussianNB':
    clf = GaussianNB()
  elif method == "KNeighborsClassifier":
    clf = KNeighborsClassifier()
  elif method == "LogisticRegression":
    clf = LogisticRegression()
  elif method == 'MLPClassifier':
    clf = MLPClassifier(alpha=1, max_iter=1000)
  elif method == 'RandomForestClassifier':
    clf = RandomForestClassifier()
  elif method == "SVC":
    clf = svm.SVC()
  elif method == 'xgboost':
    clf = xgboost.XGBClassifier(use_label_encoder=False,eval_metric="logloss")
  else:
    sys.exit(method + ' is not recognized')
  # https://stackoverflow.com/questions/58101126/using-scikit-learn-onehotencoder-with-a-pandas-dataframe

  Y = pandasDF[dependent_var]
  X = pandasDF.drop([dependent_var], axis=1)
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
  mlflow.sklearn.autolog()
  # With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
  with mlflow.start_run():
    # Set the model parameters. 
    n_estimators = 200
    #colsample_bytree = 0.3 # colsample_bytree is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
    #learning_rate = 0.05
    #max_depth = 6# default 6; max. depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguided growing policy when tree_method is set as hist or gpu_hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.
    #min_child_rate = 0
    #gamma = 0 # default = 0; Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.

    # Create and train model.
    clf.fit(X_train, y_train)#, n_estimators=n_estimators, colsample_bytree=colsample_bytree, learning_rate=learning_rate, max_depth=max_depth, gamma = gamma, use_label_encoder=False, eval_metric = 'logloss')
    #xg_clf.fit(X_train, y_train)
    # Use the model to make predictions on the test dataset.
    predictions = clf.predict(X_test)
  accuracy = accuracy_score(y_test, predictions)
  pre_score  = precision_score(y_test, predictions)
  ##---precision-recall
  precision_curve, recall_curve, _ = precision_recall_curve(y_test, predictions)
  prdata = [precision_curve.tolist(), recall_curve.tolist()]
  ref_to_json_file(prdata, xgb_json_dir + output_stem + '.' + method + '.precision.recall.json')
  ##
  return_dict = {}
  return_dict['Feature Importance'] = {}
  #for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
  #  return_dict['Feature Importance'][importance_type] = clf.get_booster().get_score( importance_type = importance_type)
  kfold = KFold(n_splits=30)
  results = cross_val_score(clf, X, Y, cv=kfold)
  accuracy = results.mean() * 100
  y_score = clf.predict_proba(X_test)[:, 1]
  fpr, tpr, _ = roc_curve(y_test, y_score)
  roc_data = [fpr.tolist(), tpr.tolist()]
  ref_to_json_file(roc_data, xgb_json_dir + output_stem + '.' + method + '.roc.data.json')
  roc = RocCurveDisplay.from_predictions(y_test, y_score, name = method, ax = axs[axrow,axcol])
  axs[axrow,axcol].set_title(method)
  axcol += 1
  if axcol % 3 == 0:
  	axrow += 1
  	axcol = 0
  roc_svg = image_dir + output_stem + '_ROC.svg'
  return_dict['ROC_SVG'] = roc_svg
  return_dict['precision_score'] = pre_score
  results_data['precision_score'][method] = pre_score
  results_data['accuracy'][method] = accuracy
  results_data['roc_auc'][method] = roc.roc_auc
  return_dict['accuracy']        = accuracy
  return_dict['roc_auc']         = roc.roc_auc
  scores = cross_validate(estimator=clf, X=X_train, y=y_train, cv=kfold, n_jobs=8, scoring=['accuracy', 'roc_auc', 'precision', 'recall', 'f1'])
  return_dict['cross_validate'] = {}
  return_dict['cross_validate']['AUC mean'] = scores['test_roc_auc'].mean()
  return_dict['cross_validate']['Accuracy mean'] = scores['test_accuracy'].mean()
  return_dict['cross_validate']['Precision mean'] = scores['test_precision'].mean()
  return_dict['cross_validate']['Recall mean'] = scores['test_recall'].mean()
  return_dict['cross_validate']['F1 mean'] = scores['test_f1'].mean()
  for metric in ('test_roc_auc', 'test_accuracy','test_precision','test_recall','test_f1'):
  	results_data[metric][method] = scores[metric].mean()
  ref_to_json_file(return_dict, xgb_json_dir + output_stem + '.' + method + '.json')
  xgb_obj = open(xgb_result_obj_dir + output_stem + '.' + method + '.obj', 'wb')
  pickle.dump(clf, xgb_obj)
  X_test_obj = open(xgb_x_test_dir + output_stem + '.' + method + '.X_test.obj', 'wb')
  pickle.dump(X_test, X_test_obj)
  return 0
methods = ['AdaBoostClassifier', 'DecisionTreeClassifier', 'GaussianNB', "KNeighborsClassifier", "LogisticRegression", 'MLPClassifier', 'RandomForestClassifier', 'xgboost']#'SVC' AttributeError: predict_proba is not available when probability=False
for method in methods:
	tmp = classifier_wrapper(args.categorical, args.target, args.output_stem, method)
for i in range(axs.shape[0]): # remove empty plots
 for j in range(axs.shape[1]):
     if axs[i, j].has_data() == False:
     	axs[i,j].remove()
if args.suptitle:
	plt.suptitle(args.suptitle, horizontalalignment = 'center')
fig.set_figwidth(12)
fig.set_figheight(12)
plt.savefig( image_dir + args.output_stem + '_ROC.svg', bbox_inches='tight', metadata={'Creator': 'made/written by' + __file__})
plt.close()
df = pd.DataFrame(results_data)
df.to_excel(args.output_stem + '.xlsx')
