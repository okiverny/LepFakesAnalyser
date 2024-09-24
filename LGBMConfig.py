import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

try:
    import lightgbm as lgbm
except ImportError:
    raise ImportError('Cannot import lightgbm')

from LFAHelpers import style

def prepare_data_lgbm(data):

    df_qjet = data[data['tau_1_tmp']=='q-jet'].reset_index()
    df_gjet = data[data['tau_1_tmp']=='g-jet'].reset_index()

    df_qjet['class']=1
    df_gjet['class']=0 

    # Checking for balance
    print("Balancing samples...")
    nevents_qjet = df_qjet['weight_nominal'].sum()
    nevents_gjet = df_gjet['weight_nominal'].sum()
    print('Number of sig/bkg events before balance = ',nevents_qjet,'/', nevents_gjet,'( factor of ',nevents_gjet/nevents_qjet,')')

    #  Weight including balance
    df_qjet['weight_for_train'] = df_qjet.apply(lambda row: row['weight_nominal']*nevents_gjet/nevents_qjet, axis=1)
    df_gjet['weight_for_train'] = df_gjet.apply(lambda row: row['weight_nominal'], axis=1)


    # Splitting into train/test
    df_qjet_train, df_qjet_test = train_test_split(df_qjet, train_size=0.7, shuffle=True)
    df_gjet_train, df_gjet_test = train_test_split(df_gjet, train_size=0.7, shuffle=True)
    df_train = pd.concat([df_qjet_train, df_gjet_train], axis=0)
    df_test = pd.concat([df_qjet_test, df_gjet_test], axis=0)

    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    # Defining inputs
    # features = ['pt_lep1','eta_lep1','charge_lep1','type_lep1','isTight_lep1',
    #                       'pt_lep2','eta_lep2','charge_lep2','type_lep2','isTight_lep2',
    #                       'fs_had_tau_1_pt','fs_had_tau_1_eta','fs_had_tau_1_nTrack','fs_had_tau_1_tight',
    #                       'fs_had_tau_1_RNNScore','fs_had_tau_1_JetTrackWidth','fs_had_tau_1_JetCaloWidth',
    #                       'pt_jet1','pt_jet2']
    features = ['pt_lep1','eta_lep1',
                    'pt_lep2','eta_lep2',
                    'fs_had_tau_1_pt','fs_had_tau_1_eta','fs_had_tau_1_nTrack',
                    'fs_had_tau_1_RNNScore',
                    'fs_had_tau_1_JetTrackWidth','fs_had_tau_1_JetCaloWidth',
                    'pt_jet1','pt_jet2']
    X_train, X_test = df_train[features], df_test[features]
    Y_train, Y_test = df_train['class'], df_test['class']
    weight_train, weight_test = df_train['weight_for_train'],df_test['weight_for_train']

    return X_train, X_test, Y_train, Y_test, weight_train, weight_test, features

# Averaging the predictions
def boosters_predict(boosters, data):
    num_models = len(boosters)
    result = 0
    for booster in boosters:
        result += booster.predict(data)

    result = result/num_models

    return result

def lgbm_class_model(X_train, X_test, Y_train, Y_test, w_train, w_test):

    params = {
    'application': 'binary',
    'objective': 'binary', # binary cross_entropy
    'metric': 'auc', # auc or binary_logloss
    'is_unbalance': 'false',
    'boosting': 'gbdt', # methods: gbdt, rf, dart, goss
    ## optimal used: 20, 5, lr=0.05
    'num_leaves': 20,
    'max_depth': 5,
    #'max_bin' : 10,
    #'feature_fraction': 0.5,
    #'bagging_fraction': 0.5,
    #'bagging_freq': 20,
    'learning_rate': 0.05,
    #'n_estimators': 4000,
    'min_data_in_leaf':10,
    #'reg_lambda': 1.75,
    #'subsample': 0.7,
    'verbose': 0
    }

    # my_features = ['pt_lep1','eta_lep1','charge_lep1','type_lep1','isTight_lep1',
    #                 'pt_lep2','eta_lep2','charge_lep2','type_lep2','isTight_lep2',
    #                 'fs_had_tau_1_pt','fs_had_tau_1_eta','fs_had_tau_1_nTrack','fs_had_tau_1_tight',
    #                 'fs_had_tau_1_RNNScore','fs_had_tau_1_JetTrackWidth','fs_had_tau_1_JetCaloWidth',
    #                 'pt_jet1','pt_jet2']

    my_features = ['pt_lep1','eta_lep1',
                    'pt_lep2','eta_lep2',
                    'fs_had_tau_1_pt','fs_had_tau_1_eta','fs_had_tau_1_nTrack',
                    'fs_had_tau_1_RNNScore',
                    'fs_had_tau_1_JetTrackWidth','fs_had_tau_1_JetCaloWidth',
                    'pt_jet1','pt_jet2']

    train_data = lgbm.Dataset(X_train, label=Y_train, weight=w_train.values, feature_name=my_features)
    valid_data = lgbm.Dataset(X_test, label=Y_test, weight=w_test.values, reference=train_data, feature_name=my_features)

    lgb_cv = lgbm.cv(
        params = params,
        train_set = train_data,
        shuffle = True,
        nfold = 5,
        return_cvbooster=True,
        verbose_eval=20,
        seed = 22,
        num_boost_round=200,
        early_stopping_rounds=100
    )

    optimal_rounds = np.argmax(lgb_cv['auc-mean'])
    best_cv_score = max(lgb_cv['auc-mean'])
    best_models = lgb_cv['cvbooster']

    print('optimal_rounds',optimal_rounds)
    print('best_cv_score',best_cv_score)
    print('best_model_iter', best_models.best_iteration-1)
    print(len(best_models.boosters))
    print('best_model (current iter)', best_models.boosters[0].current_iteration())

    plt.rcParams.update({'font.size': 22})
    f = plt.figure(figsize=(20,8))
    x = np.arange(len(lgb_cv['auc-mean']))
    y = lgb_cv['auc-mean']
    yerr = lgb_cv['auc-stdv']
    plt.errorbar(x, y, yerr=yerr, label='cv evolution')
    plt.xlabel('Iterations')
    plt.ylabel('auc')
    plt.savefig('roc.png')

    plt.figure(figsize=(10,20))
    ax = plt.gca()
    lgbm.plot_importance(best_models.boosters[0], ax=ax, max_num_features = 10,height=.5) #, height=.9
    plt.savefig('features.png')
    
    return optimal_rounds,best_cv_score,best_models.boosters