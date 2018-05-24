import numpy as np
import pandas as pd
import os
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc,recall_score,precision_score,roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import gc as g
import time
import lightgbm as lg


out = 0
debug = 0
local = 1
sam = 1


data_types = {
        'ip'            : 'uint64',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint32',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32',
        }

def preprocess_clicktime(data) :

    data['hour'] = pd.to_datetime(data.click_time).dt.hour.astype('uint8')
    data['day'] = pd.to_datetime(data.click_time).dt.day.astype('uint8')
    data['minute'] = pd.to_datetime(data.click_time).dt.minute.astype('uint8')

    return data

start_time = time.time()


if local==0:
    path =  'https://s3.amazonaws.com/kagglead/test.csv'

    train_path = path+'test.csv'
    test_path = path+'train.csv'
    trainsam_path = path+'train_sample.csv'

    train_df = pd.read_csv(train_path , parse_dates=['click_time'],dtype=data_types, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
    train_df.to_csv('/scratch/train_f.csv',index=False)

    test_df = pd.read_csv(test_path, parse_dates=['click_time'], dtype=data_types, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    test_df.to_csv('/scratch/test_f.csv',index=False)
    trainsam_df = pd.read_csv(trainsam_path , parse_dates=['click_time'],dtype=data_types, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
    trainsam_df.to_csv('/scratch/trainsam_f.csv',index=False)



print('[{}] Finished loading data'.format(time.time() - start_time))


local_train =  '/scratch/train_f.csv'

local_trainsam ='/scratch/trainsam_f.csv'

local_test = '/scratch/test_f.csv'


if sam==0:
    train = pd.read_csv(local_train)
else :
    train =pd.read_csv(local_trainsam)

test = pd.read_csv(local_test)


if out :
    train.head()
    test.head()



train = preprocess_clicktime(train)

test = preprocess_clicktime(test)


features = []

features.extend(['ip', 'app', 'device', 'os', 'channel','hour','day','minute'])



y_train = train['is_attributed']

x_train = train[features].values

x_test  = test[features].values



models = {
    'ExtraTrees': ExtraTreesClassifier(),
    'RandomForest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'xgboost': xgb.XGBClassifier()

}
parameters = {
    'ExtraTrees': { 'n_estimators': [16, 32] },
    'RandomForest': { 'n_estimators': [16, 32] },
    'AdaBoost':  { 'n_estimators': [16, 32] },
    'GradientBoosting': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },

    'xgboost' :{
    'max_depth': [4], #[4,8,10,15], #4 seems decent 
    'subsample': [0.9], #[0.4,0.7,0.9]
    'colsample_bytree': [0.7], 
    'n_estimators': [1200], 
    'reg_alpha': [0.04] #[0.01,0.04,0.08 ]
},

}


mkeys = models.keys()
gridinfo = {}

table =[]


for k in mkeys:
	print("Running GridSearchCV for %s." % k)
	model = models[k]
	params = parameters[k]
	res = GridSearchCV(model, params, cv=5, n_jobs=-1,verbose=1)
	res.fit(x_train,y_train)
	gridinfo[k] = res



for k in mkeys:
	for ginfo in gridinfo[k].grid_scores_ :
			stat = {
                 'model': k,
                 'minimum_score': min(ginfo.cv_validation_scores),
                 'maximum_score': max(ginfo.cv_validation_scores),
                 'mean_score': np.mean(ginfo.cv_validation_scores),
                 'std_score': np.std(ginfo.cv_validation_scores),
            }
			table.append(pd.Series({**ginfo.parameters,**stat}))


alltab = pd.concat(table, axis=1).T.sort_values(['minimum_score'], ascending=False)



columns = ['model', 'minimum_score', 'mean_score', 'maximum_score', 'std_score']
print(alltab[columns])



print('[{}] Finish all models Training'.format(time.time() - start_time))

g.collect()

# printing AUC curve 


xgb = xgb.XGBClassifier()

xgb_fitter = GridSearchCV(xgb,params,cv=10,scoring="roc_auc",n_jobs=-1,verbose=2)

xgb_fitter.fit(x_train, y_train)

estimator = xgb_fitter.best_estimator_

print(estimator)



# Roc AUC with all train data
predicted_y = estimator.predict_proba(x_train)
print(" Area Under the Receiver Operating Characteristic curve.: ", roc_auc_score(y_train, predicted[:,1]))
fpr, tpr, _ = roc_curve(y_train, predicted_y[:,1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='red',lw=2, label=' (area = %0.5f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='C0', lw=lw, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.08])

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.title('AUC - ROC ')
plt.legend(loc="lower right")
plt.show()




'''
Feature engineering : creating aggregates to experiment with .
'''



temp = train[['ip']+['channel']].groupby(['ip'])['channel'].nunique().reset_index().rename(columns={'channel':'X_channel'})
train = train.merge(temp, on=['ip'], how='left')
del temp
train['X_channel'] = train['X_channel'].astype('uint8')
g.collect()


temp = train[['ip', 'device', 'os']+['app']].groupby(['ip', 'device', 'os'])['app'].cumcount()
train['X_appc']=temp.values
del temp
train['X_appc'] = train['X_appc'].astype('uint32')
g.collect()


temp = train[['ip', 'day']+['hour']].groupby(['ip', 'day'])['hour'].nunique().reset_index().rename(columns={'hour':'X_hour'})
train = train.merge(temp, on=['ip', 'day'], how='left')
del temp
train['X_hour'] = train['X_hour'].astype('uint8')
g.collect()


temp = train[['ip']+['app']].groupby(['ip'])['app'].nunique().reset_index().rename(columns={'app':'X_app'})
train = train.merge(temp, on=['ip'], how='left')
del temp
train['X_app'] = train['X_app'].astype('uint8')
g.collect()



temp = train[['ip', 'app']+['os']].groupby(['ip', 'app'])['os'].nunique().reset_index().rename(columns={'os':'X_os'})
train = train.merge(temp, on=['ip', 'app'], how='left')
del temp
train['X_os'] = train['X_os'].astype('uint8')
g.collect()


temp = train[['ip']+['device']].groupby(['ip'])['device'].nunique().reset_index().rename(columns={'device':'X_device'})
train = train.merge(temp, on=['ip'], how='left')
del temp
train['X_device'] = train['X_device'].astype('uint16')
g.collect()




temp = train[['app']+['channel']].groupby(['app'])['channel'].nunique().reset_index().rename(columns={'channel':'X_channelu'})
train = train.merge(temp, on=['app'], how='left')
del temp
train['X_channelu'] = train['X_channelu'].astype('uint32')
g.collect()



temp = train[['ip']+['os']].groupby(['ip'])['os'].cumcount()
train['X_osc']=temp.values
del temp
train['X_osc'] = train['X_osc'].astype('uint32')
g.collect()


temp = train[['ip', 'device', 'os']+['app']].groupby(['ip', 'device', 'os'])['app'].nunique().reset_index().rename(columns={'app':'X_appu'})
train = train.merge(temp, on=['ip', 'device', 'os'], how='left')
del temp
train['X_appu'] = train['X_appu'].astype('uint32')
g.collect()



temp = train[['ip', 'day', 'hour']][['ip', 'day', 'hour']].groupby(['ip', 'day', 'hour']).size().rename('ip_totcount').to_frame().reset_index()
train = train.merge(temp, on=['ip', 'day', 'hour'], how='left')
del temp
train['ip_totcount'] = train['ip_totcount'].astype('uint32')
g.collect()




temp = train[['ip', 'app']][['ip', 'app']].groupby(['ip', 'app']).size().rename('ipapp_count').to_frame().reset_index()
train = train.merge(temp, on=['ip', 'app'], how='left')
del temp
train['ipapp_count'] = train['ipapp_count'].astype('uint32')
g.collect()



temp = train[['ip', 'app', 'os']][['ip', 'app', 'os']].groupby(['ip', 'app', 'os']).size().rename('ipappos_count').to_frame().reset_index()
train = train.merge(temp, on=['ip', 'app', 'os'], how='left')
del temp
train['ipappos_count'] = train['ipappos_count'].astype('uint32')
g.collect()


temp = train[['ip', 'day', 'channel']+['hour']].groupby(['ip', 'day', 'channel'])['hour'].var().reset_index().rename(columns={'hour':'ip_totchan_count'})
train = train.merge(temp, on=['ip', 'day', 'channel'], how='left')
del temp
train['ip_totchan_count'] = train['ip_totchan_count'].astype('float32')
g.collect()


temp = train[['ip', 'app', 'os']+['hour']].groupby(['ip', 'app', 'os'])['hour'].var().reset_index().rename(columns={'hour':'ipappos_var'})
train = train.merge(temp, on=['ip', 'app', 'os'], how='left')
del temp
train['ipappos_var'] = train['ipappos_var'].astype('float32')
g.collect()



temp = train[['ip', 'app', 'channel']+['day']].groupby(['ip', 'app', 'channel'])['day'].var().reset_index().rename(columns={'day':'ipappchannelvar_day'})
train = train.merge(temp, on=['ip', 'app', 'channel'], how='left')
del temp
train['ipappchannelvar_day'] = train['ipappchannelvar_day'].astype('float32')
g.collect()


temp = train[['ip', 'app', 'channel']+['hour']].groupby(['ip', 'app', 'channel'])['hour'].mean().reset_index().rename(columns={'hour':'ipappchannelmean_hour'})
train = train.merge(temp, on=['ip', 'app', 'channel'], how='left')
del temp

train['ipappchannelmean_hour'] = train['ipappchannelmean_hour'].astype('float32')
g.collect()


train['ip_totcount'] = train['ip_totcount'].astype('uint16')
train['ipapp_count'] = train['ipapp_count'].astype('uint16')
train['ipappos_count'] = train['ipappos_count'].astype('uint16')

target = 'is_attributed'
features.extend([ 'ip_totcount', 'ip_totchan_count', 'ipapp_count',
              'ipappos_count', 'ipappos_var',
              'ipappchannelvar_day','ipappchannelmean_hour',
              'X_channel', 'X_appc', 'X_hour', 'X_app', 'X_os', 'X_device', 'X_channelu', 'X_osc', 'X_appu'])



cat = ['app', 'device', 'os', 'channel', 'hour', 'day']

lg_parameters = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'learning_rate': 0.15, # it should be below 0.2
        'gamma': 6e-08,
        'max_delta_step': 20,
        'num_leaves': 15,  # smaller than 2^(6)
        'max_depth': 6,  # -1 was of no use 
        'min_child_samples': 150,  # nealy default
        'max_bin': 250, 
        'subsample': 0.9,  
        'subsample_freq': 1 , 
        'colsample_bytree': 1.0,  
        'min_child_weight': 5,
        'subsample_for_bin': 200500, 
        'min_split_gain': 0,  
        'reg_alpha': 1e-07,  
        'reg_lambda': 1000.0,
        'scale_pos_weight': 354,
        'colsample_bylevel': 0.1,
        'n_estimators': 100,
        'nthread': 4,
        'verbose': 0,
        'metric':'auc'
    }



x_train = lg.Dataset(train[features].values, label=train[target].values,feature_name=features,categorical_feature=cat)
x_valid = lg.Dataset(val_df[features].values, label=val_df[target].values,feature_name=features,categorical_feature=cat)

results = {}

l_trained = lg.train(lg_parameters, x_train, valid_sets=[x_train, x_valid], valid_names=['train','valid'],evals_result=results, num_boost_round=1000,early_stopping_rounds=50,verbose_eval=10)

print('Plotting feature importances')
p1 = lg.plot_importance(l_trained)
plt.show()

g.collect()
