import logging
import pandas as pd
import numpy as np
from itertools import cycle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, auc, roc_curve, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer

### tensorflow libraries
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
print(keras.__version__)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Dropout,AlphaDropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from bayes_opt import BayesianOptimization

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import shap
import ternary

def plot_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix', is_train=False):
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

    f = plt.figure(figsize=(8,8))
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', y=0.95)
    plt.xlabel('Predicted label')
    if not is_train:
        plt.savefig('cnf_matrix.png')
    else:
        plt.savefig('cnf_matrix_train.png')
    plt.clf() # clear the current figure

def plot_evolution(cv_results, metric_name):
    plt.rcParams.update({'font.size': 22})
    f = plt.figure(figsize=(20,8))
    x = np.arange(len(cv_results[metric_name+'-mean']))
    y = cv_results[metric_name+'-mean']
    yerr = cv_results[metric_name+'-stdv']
    plt.errorbar(x, y, yerr=yerr, label='cv evolution')
    plt.xlabel('Iterations')
    plt.ylabel('LogLoss mean')
    plt.tight_layout()
    plt.legend()
    plt.savefig('cv_evolution.png')
    plt.clf() # clear the current figure

def plot_scatter(data, xname='tH_LHscore', yname='others_LHscore', classname='process'):
    # Create a scatter plot with sample weights
    fig, ax = plt.subplots(figsize=(10, 10))

    data[classname] = np.where(data[classname].isin(['tH', 'tZ']), data[classname], 'others')
    sns.scatterplot(x=xname, y=yname, hue=classname, data=data, alpha=0.5)
    plt.savefig('LGBM_scatter.png')
    plt.clf()


def plot_weight(df):
    # Separate positive and negative values
    positive_weights = df[df['weight_nominal'] > 0]['weight_nominal']
    negative_weights = abs(df[df['weight_nominal'] < 0]['weight_nominal'])

    # Plot the histograms
    f = plt.figure(figsize=(10,10))
    plt.hist(positive_weights, bins=100, color='blue', alpha=0.5, label='Positive', range=(0, 10./1000000), density=True)
    plt.hist(negative_weights, bins=100, color='red', alpha=0.5, label='Negative', range=(0, 10./1000000), density=True)

    # Add labels and title
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('Distribution of Positive and Negative Weights')
    plt.legend()

    plt.savefig('weights.png')
    plt.clf()

def plot_probs(df):
    # remove data events
    df_mc = df[df['process']!='data']

    classes = ['tH','tZ','others']
    colors = {'tH':'blue', 'tZ':'green', 'others':'orange'}
    classes_basic = [cl for cl in classes if cl!='others']

    f = plt.figure(figsize=(10,10))
    for cl in classes:
        if cl=='others':
            mask = ~df_mc['process'].isin(classes_basic)
        else:
            mask = df_mc['process']==cl

        # If want to check a particular mask
        #mask = df['process']=='tH'

        plt.hist(df[cl+'_prob'][mask],
                density=True,
                 weights=df[mask]['weight_nominal'].abs(),
                 bins=40, color=colors[cl], alpha=0.5, label='P('+cl+') for '+cl+' MC')
        # plt.hist(df[mask]['tH_prob'],
        #          density=True,
        #          weights=df[mask]['weight_nominal'].abs(),
        #          bins=40, color=colors[cl], alpha=0.5, label='P(tH) for '+cl+' MC')

    # Add labels and title
    plt.xlabel('LGBM class probabilities')
    plt.ylabel('Normalized events')
    plt.title('Distribution of LGBM class probs')
    plt.legend()
    plt.savefig('LGBM_probs.png')
    plt.clf()

def plot_scores(df, varname='tH_score'):
    # remove data events
    df_mc = df[df['process']!='data']

    classes = ['tH','tZ','others']
    colors = {'tH':'blue', 'tZ':'green', 'others':'orange'}
    classes_basic = [cl for cl in classes if cl!='others']

    f = plt.figure(figsize=(10,10))
    for cl in classes:
        if cl=='others':
            mask = ~df_mc['process'].isin(classes_basic)
        else:
            mask = df_mc['process']==cl

        # If want to check particular mask
        # mask = df['process']=='tH'

        plt.hist(df[varname][mask],
                 density=True,
                 weights=df[mask]['weight_nominal'].abs(),
                 bins=40, color=colors[cl], alpha=0.5, label='LH score for '+cl)

    # Add labels and title
    plt.xlabel('LH score')
    plt.ylabel('Normalized events')
    plt.title('Distribution of LH scores')
    plt.legend()
    plt.savefig('LGBM_'+varname+'.png')
    plt.clf()

def plot_QGTagger(df, varname):
    classes = ['q-jet', 'g-jet']
    classes = [0, 1]
    target_names = {0: 'q-jet', 1: 'g-jet'}
    colors = {'q-jet':'blue', 'g-jet':'green'}

    if varname=='fs_had_tau_1_pt':
        bins = [i for i in range(20, 40)]
        more_bins = [i for i in range(40, 50, 2)]
        for i in more_bins:
            bins.append(i)
        more_bins = [i for i in range(50, 105, 5)]
        for i in more_bins:
            bins.append(i)
        more_bins = [105, 110, 115, 120, 125, 130, 140, 150,200, 2000]
        for i in more_bins:
            bins.append(i)
    elif varname=='fs_had_tau_1_eta':
        bins = [i*0.1 for i in range(25)]
    else:
        bins=40

    f = plt.figure(figsize=(10,10))
    for cl in classes:
        mask = df['tau_1_tmp']==cl

        plt.hist(df[varname][mask],
                 #density=True,
                 weights=df[mask]['weight_nominal'].abs(),
                 bins=bins, color=colors[target_names[cl]], alpha=0.5, label=target_names[cl])
        
    # Add labels and title
    plt.xlabel(varname)
    plt.ylabel('Normalized events')
    plt.title('Distribution of '+varname)
    plt.legend()
    plt.savefig('LGBQGTagger_'+varname+'.png')
    plt.clf()

def get_LH_discriminant(data, sig_name='tH_prob', bkg_names=['tZ_prob', 'others_prob'], k_val=0.5, out_name='tH_score'):
    # data : dataframe
    # sig_name : name of signal probability column in data
    # bkg_names : list of bkg probabilities in data
    # k_val : k is effective tunable parameter between 0 and 1
    n_bkgs = len(bkg_names)
    if n_bkgs==2:
        data[out_name] = np.log( data[sig_name] / (k_val*data[bkg_names[0]] + (1-k_val)*data[bkg_names[1]]) )
    else:
        logging.error("Can't compute the tH LH discrminant. Check your code!")

    return data

def get_discriminant(data, sig_name='tH_prob', bkg_names=['tZ_prob', 'others_prob'], out_name='tH_score'):
    # data : dataframe
    # sig_name : name of signal probability column in data
    # bkg_names : list of bkg probabilities in data
    n_bkgs = len(bkg_names)
    if n_bkgs==2:
        data[out_name] = data[sig_name] - 0.5*(data[bkg_names[0]] + data[bkg_names[1]])
    else:
        logging.error("Can't compute the tH LH discrminant. Check your code!")

    return data


class ClassifyProcess:
    def __init__(self, df, target_col, weight_col, classes = ['tH', 'tZ', 'others']):
        #classes = ['tH', 'tZ', 'others']
        # classes = {'tH': df[target_col]=='tH',
        #            'tZ': df[target_col]=='tZ',
        #            'others': ~df[target_col].isin(['tH','tZ'])}
        self.df = df
        self.target_col = target_col
        self.weight_col = weight_col
        self.classes = classes
        self.classes_map = None

        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.test_labels = None
        self.train_weights = None
        self.test_weights = None

        self.train_weights_abs = None
        self.test_weights_abs = None

        self.scaler = None
        self.model = None

        # Names of inputs
        cat_feature_list = ['m_njets','m_nbjets','fs_had_tau_1_nTrack']
        self.discrete_dims = ['m_nbjets', 'fs_had_tau_1_nTrack']
        self.cat_features = [feature for feature in cat_feature_list if feature in df.columns.tolist()]
        self.num_features = [feature for feature in df.columns.tolist() if (feature not in self.cat_features and feature!=target_col and feature!=weight_col)]


    def label_classes(self):
        logging.info('   Labeling the classes: {}'.format(self.classes))

        if type(self.classes)==list:
            if 'others' in self.classes:
                basic_classes = [cl for cl in self.classes if cl!='others']
                classes_map = {cls: i for i, cls in enumerate(self.classes)}

                self.df[self.target_col] = np.where(self.df[self.target_col].isin(basic_classes), self.df[self.target_col], 'others')
                self.df[self.target_col] = self.df[self.target_col].replace(classes_map)
                self.df[self.target_col] = self.df[self.target_col].astype(int)
                logging.info('   Renamed the targets according to {}'.format(classes_map))
            else:
                classes_map = {cls: i for i, cls in enumerate(self.classes)}
                self.df = self.df[ self.df[self.target_col].isin(self.classes) ]
                self.df[self.target_col] = self.df[self.target_col].replace(classes_map)
                self.df[self.target_col] = self.df[self.target_col].astype(int)

                logging.info('   Renamed the targets according to {}'.format(classes_map))

        if type(self.classes)==dict:
            print(self.classes)
            if 'others' in list(self.classes.keys()):
                classes_map = {cls: i for i, cls in enumerate(self.classes.keys())}
                masks = [self.classes[class_name] for class_name in self.classes]
                replace_values = [classes_map[class_name] for class_name in classes_map]

                self.df[self.target_col] = np.select(masks, replace_values, default=np.nan)
                self.df[self.target_col] = self.df[self.target_col].astype(int)

        # Store the map 'class'->'label'
        self.classes_map = classes_map

    def balance_classes(self, reweight_dims=None):
        logging.info('   Balance classes using absolute values of weights')

        def compute_class_weight_sum(data):
            data[self.weight_col+'_abs'] = data[self.weight_col].abs()

            class_sums = data.groupby(self.target_col)[self.weight_col+'_abs'].sum()
            class_sums = dict(zip(class_sums.index, class_sums.values))

            return class_sums

        if not reweight_dims:
            class_sums = compute_class_weight_sum(self.df[[self.target_col, self.weight_col]].copy())
            print('class_sums',class_sums)
            for cls, weight_sum in class_sums.items():
                self.df.loc[self.df[self.target_col] == cls, self.weight_col] /= weight_sum

            class_sums_c = compute_class_weight_sum(self.df[[self.target_col, self.weight_col]].copy())
            print('class_sums_c',class_sums_c)

        # Make uniform dimensions
        else:
            logging.info('   Reweighting to uniform distributions for the following variables: {}'.format(reweight_dims))

            # Define a fine binning for each variable
            bins = {}
            for dim in reweight_dims:
                if dim in self.discrete_dims:
                    # Find unique values in self.df[dim]
                    unique_vals_sorted = np.sort(np.unique(self.df[dim]))
                    # Add an extra value to the end of unique_vals_sorted to represent the upper bound of the last bin
                    unique_vals_sorted = np.append(unique_vals_sorted, unique_vals_sorted[-1] + 1)
                    bins[dim] = unique_vals_sorted
                else:
                    vals = self.df[dim]
                    min_val, max_val = np.min(vals), np.max(vals)
                    bins[dim] = np.linspace(min_val, max_val, num=50)

                    if dim=='fs_had_tau_1_pt':
                        bins[dim] = [i for i in range(20, 40)]
                        more_bins = [i for i in range(40, 50, 2)]
                        for i in more_bins:
                            bins[dim].append(i)
                        more_bins = [i for i in range(50, 105, 5)]
                        for i in more_bins:
                            bins[dim].append(i)
                        more_bins = [105, 110, 115, 120, 125, 130, 140, 150,200, 2000]
                        for i in more_bins:
                            bins[dim].append(i)

            # Assume 4 variables ('m_nbjets','fs_had_tau_1_nTrack','fs_had_tau_1_pt','fs_had_tau_1_eta')
            for index0 in range(len(bins[reweight_dims[0]])-1):
                low0 = bins[reweight_dims[0]][index0]
                high0 = bins[reweight_dims[0]][index0+1]

                for index1 in range(len(bins[reweight_dims[1]])-1):
                    low1 = bins[reweight_dims[1]][index1]
                    high1 = bins[reweight_dims[1]][index1+1]

                    for index2 in range(len(bins[reweight_dims[2]])-1):
                        low2 = bins[reweight_dims[2]][index2]
                        high2 = bins[reweight_dims[2]][index2+1]

                        for index3 in range(len(bins[reweight_dims[3]])-1):
                            low3 = bins[reweight_dims[3]][index3]
                            high3 = bins[reweight_dims[3]][index3+1]

                            bin_mask = ((self.df[reweight_dims[0]]>=low0) & (self.df[reweight_dims[0]]<high0) &
                                        (self.df[reweight_dims[1]]>=low1) & (self.df[reweight_dims[1]]<high1) &
                                        (self.df[reweight_dims[2]]>=low2) & (self.df[reweight_dims[2]]<high2) &
                                        (self.df[reweight_dims[3]]>=low3) & (self.df[reweight_dims[3]]<high3))
                            bin_df = self.df[bin_mask]
                            bin_class_sums = compute_class_weight_sum(bin_df[[self.target_col, self.weight_col]].copy())

                            # Skip empty bins
                            if not len(bin_class_sums):
                                continue

                            # reweight each bin
                            for cls, weight_sum in bin_class_sums.items():
                                mask = bin_mask & (self.df[self.target_col] == cls)
                                self.df.loc[mask, self.weight_col] /= weight_sum

                            # mask = bin_mask & (self.df[self.target_col] == 1)
                            # self.df.loc[mask, self.weight_col] /= weight_sum


            # Avoid NaN if exists
            self.df[self.weight_col] = self.df[self.weight_col].fillna(0)

            # additional global reweighting
            class_sums = compute_class_weight_sum(self.df[[self.target_col, self.weight_col]].copy())
            print('class_sums',class_sums)
            for cls, weight_sum in class_sums.items():
                self.df.loc[self.df[self.target_col] == cls, self.weight_col] /= weight_sum

            class_sums_c = compute_class_weight_sum(self.df[[self.target_col, self.weight_col]].copy())
            print('class_sums_c',class_sums_c)

    
    def split_data(self, test_size=0.2, random_state=42):
        logging.info('   Splitting data into train/test')
        y = self.df[self.target_col]
        X = self.df.drop(self.target_col, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=random_state) # stratify=y

        # Apply abs weights for train set
        self.train_weights = X_train[self.weight_col]
        self.test_weights = X_test[self.weight_col]

        # get the number of positive weights
        num_positive_weights = np.sum(self.train_weights > 0)
        # get the number of negative weights
        num_negative_weights = np.sum(self.train_weights < 0)
        print('Train samples: # of pos/neg = {p}/{n}'.format(p=num_positive_weights, n=num_negative_weights))

        self.train_weights_abs = self.train_weights.abs()
        self.test_weights_abs = self.test_weights.abs()

        self.train_data = X_train.drop(columns=[self.weight_col])
        self.test_data = X_test.drop(columns=[self.weight_col])
        self.train_labels = y_train
        self.test_labels = y_test

    def scale_features(self, with_mean=True, with_std=True):
        logging.info('   Scale features')

        self.scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        if len(self.cat_features):
            # Scale numerical features
            num_train_data = pd.DataFrame(self.scaler.fit_transform(self.train_data[self.num_features], sample_weight=self.train_weights), columns=self.num_features).set_index(self.train_data.index)
            num_test_data = pd.DataFrame(self.scaler.transform(self.test_data[self.num_features]), columns=self.num_features).set_index(self.test_data.index)

            # Categorical feature
            cat_train_data = self.train_data[self.cat_features]
            cat_test_data = self.test_data[self.cat_features]

            # Concatenate scaled numerical and categorical features
            self.train_data = pd.concat([num_train_data, cat_train_data], axis=1)
            self.test_data = pd.concat([num_test_data, cat_test_data], axis=1)

        else:
            self.train_data = pd.DataFrame(self.scaler.fit_transform(self.train_data, sample_weight=self.train_weights), columns=self.train_data.columns)
            self.test_data = pd.DataFrame(self.scaler.transform(self.test_data), columns=self.test_data.columns)

        ## Pileline?
        # print(self.df.dtypes)
        # self.scaler=ColumnTransformer([
        #     ('num', StandardScaler(with_mean=with_mean, with_std=with_std), self.num_features),
        #     ('cat', LabelBinarizer(), self.cat_features),
        # ])
        # self.train_data = self.scaler.fit_transform(self.train_data)
        # self.test_data = self.scaler.transform(self.test_data)
        



    def train_model(self, params, num_boost_round=100, nfold=5, early_stopping_rounds=20, verbose_eval=1, random_state=42, do_early_stopping = False, metric_name = 'multi_logloss'):

        X_train = lgb.Dataset(self.train_data, label=self.train_labels, weight=self.train_weights_abs,
                              feature_name=self.train_data.columns.tolist(),
                              categorical_feature=self.cat_features if len(self.cat_features) else 'auto',
                              free_raw_data=False)

        if do_early_stopping:
            cv_results = lgb.cv(params,
                            X_train,
                            num_boost_round=num_boost_round,
                            nfold=nfold,
                            stratified=True,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=verbose_eval,
                            seed=random_state,
                            shuffle = True )

            best_iteration = len(cv_results[metric_name+'-mean'])
            best_loss = cv_results[metric_name+'-mean'][best_iteration-1]
            loss_spread = cv_results[metric_name+'-stdv'][best_iteration-1]

            print(f"Best iteration: {best_iteration}")
            print(f"Best loss: {best_loss:.4f} +/- {loss_spread:.4f}")

            plot_evolution(cv_results, metric_name)
            num_boost_round = best_iteration

        self.model = lgb.train(params, X_train, num_boost_round=num_boost_round, verbose_eval=verbose_eval)
        #self.model = lgb.train(params, X_train, verbose_eval=verbose_eval, num_boost_round=200)

        # mask = (self.df.m_nbjets==1) & (self.df.fs_had_tau_1_nTrack==1)
        # plot_QGTagger(self.df[mask], 'fs_had_tau_1_pt')
        # plot_QGTagger(self.df[mask], 'fs_had_tau_1_eta')

    def train_model_nn(self, params):

        num_features = self.train_data.shape[1]
        input_shape = (num_features,)

        # Input
        X_input = Input(input_shape)

        # Layer(s)
        X = Dropout(rate=params['dropout_rate_1'])(X_input)
        X = Dense(params['dense_size_1'], activation="relu",
                #kernel_initializer='lecun_normal', # he_normal, he_uniform, lecun_uniform
                name = 'Dense1')(X)
        
        X = Dropout(rate=params['dropout_rate_2'])(X)
        X = Dense(params['dense_size_2'], activation="relu",
                #kernel_initializer='lecun_normal',
                name = 'Dense2')(X)

        X = Dropout(rate=params['dropout_rate_3'])(X)
        X = Dense(params['dense_size_3'], activation="relu",
                #kernel_initializer='lecun_normal',
                name = 'Dense3')(X)

        # Output
        X_output = Dense(3, activation='softmax', name='output_layer')(X)

        # Model
        self.model = Model(inputs=X_input, outputs=X_output, name='classifier_model')

        # Preparing the training
        opt = keras.optimizers.Adam(learning_rate=params['learning_rate']) # Adam or Nadam

        self.model.compile(optimizer=opt,
                           loss = 'categorical_crossentropy',
                           weighted_metrics = ['categorical_crossentropy'])

        # Learning Rate Performance Scheduler
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7)

        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Convert integer-encoded labels to one-hot encoded labels
        train_labels = to_categorical(self.train_labels, num_classes=3)
        test_labels = to_categorical(self.test_labels, num_classes=3)

        history = self.model.fit(x=self.train_data,
                                 y=train_labels,
                                 sample_weight=self.train_weights_abs,
                                 validation_data = (self.test_data, test_labels, self.test_weights_abs),
                                 epochs=200,
                                 batch_size=params['batch_size'],
                                 callbacks=[lr_scheduler, early_stop] )

        #print(history.history['val_categorical_crossentropy'][-1], len(history.history['val_categorical_crossentropy']))
        return history.history['val_categorical_crossentropy'][-1]

    def predict(self, df):
        print('predict')
        # Scale the features using the scaler from the training data
        if self.scaler:
            logging.info('Scaling features for model evaluation.')
            if len(self.cat_features):
                logging.info('  Scaling only numerical features. Not scaling the following: {}'.format(self.cat_features))
                num_df = pd.DataFrame(self.scaler.transform(df[self.num_features]), columns=self.num_features).set_index(df.index)
                cat_df = df[self.cat_features]
                df_scaled = pd.concat([num_df, cat_df], axis=1)
            else:
                logging.info('  Scaling all features. No categorical features are found.')
                df_scaled = pd.DataFrame(self.scaler.transform(df), columns=df.columns).set_index(df.index)

        # Get the predictions from the model
        predictions = self.model.predict(df_scaled)

        # Convert the predictions to class labels
        labels = np.argmax(predictions, axis=1)

        my_labels = self.classes if type(self.classes)==list else list(self.classes.keys())
        output = {cl+'_prob': predictions[:, i] for i,cl in enumerate(my_labels)}
        result = pd.DataFrame(output).set_index(df.index)

        # result = pd.DataFrame({#'class_pred': labels,
        #                   'tH_prob': predictions[:, 0],
        #                   'tZ_prob': predictions[:, 1],
        #                   'others_prob': predictions[:, 2]}).set_index(df.index)

        return result

    def predicit_binary(self, df):
        print('predict binary')
        # Scale the features using the scaler from the training data
        if self.scaler:
            logging.info('Scaling features for model evaluation.')
            if len(self.cat_features):
                logging.info('  Scaling only numerical features. Not scaling the following: {}'.format(self.cat_features))
                num_df = pd.DataFrame(self.scaler.transform(df[self.num_features]), columns=self.num_features).set_index(df.index)
                cat_df = df[self.cat_features]
                df_scaled = pd.concat([num_df, cat_df], axis=1)
            else:
                logging.info('  Scaling all features. No categorical features are found.')
                df_scaled = pd.DataFrame(self.scaler.transform(df), columns=df.columns).set_index(df.index)

        # Get the predictions from the model
        predictions = self.model.predict(df_scaled)

        return predictions
        
    def tune_hyperparameters(self, params_grid, cv=5, verbose=2):
        X_train = self.train_data
        y_train = self.train_labels
        weights = self.train_weights_abs
        
        gbm = lgb.LGBMClassifier()
        grid = GridSearchCV(gbm, params_grid, cv=cv, verbose=verbose)
        grid.fit(X_train, y_train, sample_weight=weights)
        
        self.model = grid.best_estimator_
        print("Best parameters found: ", grid.best_params_)
        print("Best CV score: ", grid.best_score_)

    def bayes_parameter_tune(self, init_points=5, n_iter=100, n_folds=5, random_seed=6, n_estimators=500, early_stopping_rounds=20):
        # prepare data
        X_train = lgb.Dataset(self.train_data, label=self.train_labels, weight=self.train_weights_abs,
                              feature_name=self.train_data.columns.tolist(),
                              categorical_feature=self.cat_features if len(self.cat_features) else 'auto',
                              free_raw_data=False)

        # parameters
        def lgb_eval(learning_rate, num_leaves, feature_fraction, subsample, subsample_freq, max_depth, min_data_in_leaf):
            params = {'boosting_type': 'gbdt',
                      'objective': 'multiclass',
                      'num_class': 3,
                      'metric': 'multi_logloss',
                      'is_unbalance': 'false',
                      #'device_type': 'gpu',
                      }
            params['learning_rate'] = max(min(learning_rate, 1), 0)
            params["num_leaves"] = int(round(num_leaves))
            params['feature_fraction'] = max(min(feature_fraction, 1), 0)
            params['subsample'] = max(min(subsample, 1), 0)
            params['subsample_freq'] = max(min(subsample_freq, 1), 0)
            params['max_depth'] = int(round(max_depth))
            params['min_data_in_leaf'] = int(round(min_data_in_leaf))
            #params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf

            cv_results = lgb.cv(params,
                            X_train,
                            num_boost_round=n_estimators,
                            nfold=n_folds,
                            stratified=True,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=1,
                            seed=random_seed)

            return min(cv_results['multi_logloss-mean'])

        lgbBO = BayesianOptimization(lgb_eval, {'learning_rate': (0.001, 0.15),
                                                'num_leaves': (8, 48),
                                                'feature_fraction': (0.1, 1.0),
                                                'subsample': (0.1, 1.0),
                                                'subsample_freq': (5, 50),
                                                'max_depth': (-1, 30),
                                                'min_data_in_leaf': (20, 1024),
                                                #'min_sum_hessian_in_leaf':(0,1024),
                                                },
                                                random_state=200)

        #n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
        #init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.
        lgbBO.maximize(init_points=init_points, n_iter=n_iter)

        model_metric=[]
        for model in range(len( lgbBO.res)):
            model_metric.append(lgbBO.res[model]['target'])

        # return best parameters
        return lgbBO.res[pd.Series(model_metric).idxmin()]['target'], lgbBO.res[pd.Series(model_metric).idxmin()]['params']

    def bayes_parameter_tune_nn(self, init_points=5, n_iter=100):

        def nn_eval(dropout_rate_1, dropout_rate_2, dropout_rate_3, dense_size_1, dense_size_2, dense_size_3, learning_rate, batch_size):
            params = {
                'dropout_rate_1': max(min(dropout_rate_1, 1), 0),
                'dropout_rate_2': max(min(dropout_rate_2, 1), 0),
                'dropout_rate_3': max(min(dropout_rate_3, 1), 0),
                'dense_size_1': int(round(dense_size_1)),
                'dense_size_2': int(round(dense_size_2)),
                'dense_size_3': int(round(dense_size_3)),
                'learning_rate': max(min(learning_rate, 1), 0),
                'batch_size': int(round(batch_size)),
            }

            nn_loss = self.train_model_nn(params)
            return nn_loss

        nnBO = BayesianOptimization(f=nn_eval,
                                    pbounds={'dropout_rate_1': (0.0, 0.35),
                                              'dropout_rate_2': (0.001, 0.35),
                                              'dropout_rate_3': (0.0, 0.35),
                                              'dense_size_1': (20, 100),
                                              'dense_size_2': (10, 100),
                                              'dense_size_3': (5, 70),
                                              'learning_rate': (0.0001, 0.65),
                                              'batch_size': (16, 2048),},
                                    random_state=200)

        #n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
        #init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.
        nnBO.maximize(init_points=init_points, n_iter=n_iter)

        model_metric=[]
        for model in range(len( nnBO.res)):
            model_metric.append(nnBO.res[model]['target'])

        # return best parameters
        return nnBO.res[pd.Series(model_metric).idxmin()]['target'], nnBO.res[pd.Series(model_metric).idxmin()]['params']


    def evaluate_model(self):
        train_pred = self.model.predict(self.train_data)
        train_pred = [np.argmax(x) for x in train_pred]
        test_pred = self.model.predict(self.test_data)
        test_pred = [np.argmax(x) for x in test_pred]

        my_labels = self.classes if type(self.classes)==list else list(self.classes.keys())

        print("Train Accuracy with |weight|:", accuracy_score(self.train_labels, train_pred, sample_weight=self.train_weights_abs))
        print("Train Accuracy with all weights:", accuracy_score(self.train_labels, train_pred, sample_weight=self.train_weights))
        print("Test Accuracy with |weight|:", accuracy_score(self.test_labels, test_pred, sample_weight=self.test_weights_abs))
        print("Test Accuracy with all weights:", accuracy_score(self.test_labels, test_pred, sample_weight=self.test_weights))
        
        confus_matrix = confusion_matrix(self.test_labels, test_pred, sample_weight=self.test_weights_abs)
        plot_confusion_matrix(confus_matrix, my_labels, normalize=True, title='Confusion matrix')

        confus_matrix_train = confusion_matrix(self.train_labels, train_pred, sample_weight=self.train_weights_abs)
        plot_confusion_matrix(confus_matrix_train, my_labels, normalize=True, title='Confusion matrix', is_train=True)

    def explain_model(self):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.train_data)
        expected_value = explainer.expected_value
        shap_interaction_values = explainer.shap_interaction_values
        my_labels = self.classes if type(self.classes)==list else list(self.classes.keys())

        from LFAHelpers import ensure_dir
        ensure_dir('Shap/')

        plt.figure(figsize=(20,20))
        shap.summary_plot(shap_values, self.train_data, class_names=my_labels, show=False)
        plt.savefig('Shap/shap_summary_plot.png')
        plt.clf()

        # Force Plot: visualizes the contribution of each feature to the prediction of a single data point.
        # The contributions are represented by arrows, with their length proportional to the magnitude of their contribution.
        plt.figure(figsize=(20, 6))
        shap.force_plot(expected_value[0],
                        shap_values[0][0],  # class 0, feature 0
                        self.train_data.loc[0,:],
                        matplotlib=True, show=False)
        plt.savefig('Shap/shap_force_plot.png')
        plt.clf()

        # contribution of features to multiple output predictions
        plt.figure(figsize=(20, 20))
        shap.multioutput_decision_plot(expected_value,
                                       shap_values,
                                       row_index=0,
                                       feature_names=self.train_data.columns.to_list(),
                                       legend_labels=my_labels,
                                       legend_location='lower right',
                                       #link='logit',
                                       show=False)
        plt.savefig('Shap/shap_multidecision_plot.png')
        plt.clf()

        # scatter plot that shows the effect a single feature has on the predictions class shap_values[0] made by the model
        shap.dependence_plot("rank(0)", shap_values[0], self.train_data, show=False)
        plt.savefig('Shap/shap_dependence_plot.png')
        plt.clf()

        # we can use shap.approximate_interactions to guess which features
        # may interact with "rank(0)" feature
        inds = shap.approximate_interactions("rank(0)", shap_values[0], self.train_data)
        for i in range(3):
            shap.dependence_plot("rank(0)", shap_values[0], self.train_data, interaction_index=inds[i], show=False)
            plt.title("rank(0) dependence plot for tHq")
            plt.savefig('Shap/shap_dependence_plot'+str(i)+'.png')
            plt.clf()


    def plot_ternary(self):

        y_test_pred = self.model.predict(self.test_data)
        y_test_pred = pd.DataFrame(y_test_pred, columns=['pred_tH', 'pred_tZ', 'pred_others']).set_index(self.test_labels.index)
        y_test_pred['true_class'] = self.test_labels.values

        target_names = {0: 'tH', 1: 'tZ', 2: 'others'}
        colors = {0: 'blue', 1: 'green', 2: 'orange'}
        n_classes = len(np.unique(self.test_labels))
        fig, axs = plt.subplots(1, 3, figsize=(6*n_classes+2, 6))

        for class_id in range(n_classes):
            # Extract probabilities for the class
            class_probs = y_test_pred[y_test_pred['true_class'] == class_id].iloc[:, :-1]
            # Convert probabilities to lists
            class_probs = [list(row) for _, row in class_probs.iterrows()]

            # Create ternary plot
            ax = axs[class_id]
            ax.axis('off')
            tax = ternary.TernaryAxesSubplot(ax=ax, scale=1.0)
            tax.boundary(linewidth=2.0)
            tax.gridlines(multiple=0.2, color="black")
            tax.set_title(target_names[class_id], fontsize=16)
            
            # Plot data
            offset, fontsize = 0.15, 12
            tax.scatter(class_probs, marker='o', color=colors[class_id], label='pred for {}'.format(target_names[class_id]))
            #tax.heatmap(class_probs, style="triangular")
            tax.ticks(linewidth=1, multiple=0.1, tick_formats="%.1f", offset=0.02) # axis='lbr',
            tax.left_axis_label('{} score'.format(target_names[2]), fontsize=fontsize, offset=offset) # others
            tax.right_axis_label('{} score'.format(target_names[1]), fontsize=fontsize, offset=offset) # tZ
            tax.bottom_axis_label('{} score'.format(target_names[0]), fontsize=fontsize, offset=offset) # tH
            tax._redraw_labels()

            # Remove default Matplotlib Axes
            tax.clear_matplotlib_ticks()

        plt.savefig('ternary.png')
        plt.clf()

    def plot_roc_curves(self):
        y_test = self.test_labels
        y_test_pred = self.model.predict(self.test_data).ravel()

        y_train = self.train_labels
        y_train_pred = self.model.predict(self.train_data).ravel()

        fpr_train, tpr_train, thresholds_train = roc_curve(y_train.values, y_train_pred, sample_weight=self.train_weights)
        fpr_train, tpr_train = zip(*sorted(zip(fpr_train, tpr_train)))
        auc_train = auc(fpr_train, tpr_train)

        fpr_test, tpr_test, _ = roc_curve(y_test.values, y_test_pred, sample_weight=self.test_weights)
        fpr_test, tpr_test = zip(*sorted(zip(fpr_test, tpr_test)))
        auc_test = auc(fpr_test, tpr_test)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr_train, tpr_train, label='Train (AUC={:.3f})'.format(auc_train))
        ax.plot(fpr_test, tpr_test, label='Test (AUC={:.3f})'.format(auc_test))

        # for class_ in np.unique(y_test):
        #     fpr, tpr, thresholds = roc_curve(y_test==class_, y_test_pred[:, class_], sample_weight=self.test_weights)
        #     auc = roc_auc_score(y_test==class_, y_test_pred[:, class_]) # , average='micro', multi_class='ovr'
        #     ax.plot(fpr, tpr, label='Class {} (AUC={:.3f})'.format(class_, auc))

        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Quark/gluon jets')
        ax.legend(loc='lower right')
        plt.savefig('roc.png')
        plt.clf()

    def plot_roc_curves_OvR(self, subsample='test'):
        # transform labels from (n_samples,) to (n_samples, n_classes)
        
        n_classes = len(np.unique(self.test_labels))
        target_names = {0: 'tH', 1: 'tZ', 2: 'others'}
        if type(self.classes)==list:
            target_names = {i:cls for i, cls in enumerate(self.classes)}
        elif type(self.classes)==dict:
            target_names = {i:cls for i, cls in enumerate(self.classes.keys())}

        # Binarize the labels
        label_binarizer = LabelBinarizer().fit(self.train_labels)
        if subsample=='test':
            y_true = label_binarizer.transform(self.test_labels)
            y_pred = self.model.predict(self.test_data)
            weights = self.test_weights
            weights_abs = self.test_weights_abs
        elif subsample=='train':
            y_true = label_binarizer.transform(self.train_labels)
            y_pred = self.model.predict(self.train_data)
            weights = self.train_weights
            weights_abs = self.train_weights_abs

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for class_id in range(n_classes):
            fpr[class_id], tpr[class_id], _ = roc_curve(y_true[:, class_id], y_pred[:, class_id], sample_weight=weights_abs)
            roc_auc[class_id] = auc(fpr[class_id], tpr[class_id])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel(), sample_weight=np.tile(weights_abs, n_classes))
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # compute micro-average with negative weights weights
        fpr['micro_real'], tpr['micro_real'], _ = roc_curve(y_true.ravel(), y_pred.ravel(), sample_weight=np.tile(weights, n_classes))
        # re-sort and compute AUC
        fpr['micro_real'], tpr['micro_real'] = zip(*sorted(zip(fpr['micro_real'], tpr['micro_real'])))
        roc_auc['micro_real'] = auc(np.array(fpr['micro_real']), np.array(tpr['micro_real']))

        # Compute macro-average ROC curve and ROC area
        # Interpolate all ROC curves at these points
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(fpr_grid)
        for class_id in range(n_classes):
            mean_tpr += np.interp(fpr_grid, fpr[class_id], tpr[class_id])  # linear interpolation
        # Average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        _, ax = plt.subplots(figsize=(8, 8))
        #colors = cycle(['blue', 'green', 'red'])
        mycolors = ["aqua", "darkorange", "cornflowerblue", "blue","green","red"]
        colors = cycle(mycolors[:n_classes])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(target_names[i], roc_auc[i]))

        plt.plot(fpr["macro"], tpr["macro"],
                label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
                color="forestgreen", linestyle=":", linewidth=4)
            
        plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)
        
        plt.plot(fpr["micro_real"], tpr["micro_real"],
                label='micro-average ROC, all weights (area = {0:0.2f})'
                ''.format(roc_auc["micro_real"]),
                color='navy', linestyle=':', linewidth=4)

        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves (One-vs-Rest multiclass)')
        ax.legend(loc='lower right')
        plt.savefig('roc_'+subsample+'.png')
        plt.clf()

    def plot_roc_curves_QGTagger(self, subsample='test'):
        n_classes = len(np.unique(self.test_labels))
        target_names = {0: 'q-jet', 1: 'g-jet'}
        if type(self.classes)==list:
            target_names = {i:cls for i, cls in enumerate(self.classes)}
        elif type(self.classes)==dict:
            target_names = {i:cls for i, cls in enumerate(self.classes.keys())}

        # Binarize the labels
        label_binarizer = LabelBinarizer().fit(self.train_labels)
        if subsample=='test':
            y_true = label_binarizer.transform(self.test_labels)
            y_pred = self.model.predict(self.test_data)
            weights = self.test_weights
            weights_abs = self.test_weights_abs
        elif subsample=='train':
            y_true = label_binarizer.transform(self.train_labels)
            y_pred = self.model.predict(self.train_data)
            weights = self.train_weights
            weights_abs = self.train_weights_abs

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for class_id in range(n_classes):
            fpr[class_id], tpr[class_id], _ = roc_curve(y_true[:, class_id], y_pred[:, class_id], sample_weight=weights_abs)
            roc_auc[class_id] = auc(fpr[class_id], tpr[class_id])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel(), sample_weight=np.tile(weights_abs, n_classes))
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # compute micro-average with negative weights weights
        fpr['micro_real'], tpr['micro_real'], _ = roc_curve(y_true.ravel(), y_pred.ravel(), sample_weight=np.tile(weights, n_classes))
        # re-sort and compute AUC
        fpr['micro_real'], tpr['micro_real'] = zip(*sorted(zip(fpr['micro_real'], tpr['micro_real'])))
        roc_auc['micro_real'] = auc(np.array(fpr['micro_real']), np.array(tpr['micro_real']))

        # Compute macro-average ROC curve and ROC area
        # Interpolate all ROC curves at these points
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(fpr_grid)
        for class_id in range(n_classes):
            mean_tpr += np.interp(fpr_grid, fpr[class_id], tpr[class_id])  # linear interpolation
        # Average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        _, ax = plt.subplots(figsize=(10, 10))
        mycolors = ["aqua", "darkorange", "cornflowerblue", "blue","green","red"]
        colors = cycle(mycolors[:n_classes])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(target_names[i], roc_auc[i]))

        plt.plot(fpr["macro"], tpr["macro"],
                label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
                color="forestgreen", linestyle=":", linewidth=4)

        plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["micro_real"], tpr["micro_real"],
                label='micro-average ROC, all weights (area = {0:0.2f})'
                ''.format(roc_auc["micro_real"]),
                color='navy', linestyle=':', linewidth=4)

        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves (One-vs-Rest multiclass)')
        ax.legend(loc='lower right')
        plt.savefig('roc_'+subsample+'.png')
        plt.clf()

    def get_importance(self):
        plt.figure(figsize=(16,20))
        ax = plt.gca()
        lgb.plot_importance(self.model, ax=ax, max_num_features = 10,height=.7) #, height=.9
        plt.savefig('features.png')
        plt.clf()

    def plot_correlation_with_process(self):
        logging.info('   Computing feature correlations')

        def m(x, w):
            """Weighted Mean"""
            return np.sum(x * w) / np.sum(w)

        def cov(x, y, w):
            """Weighted Covariance"""
            return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)

        def corr(x, y, w):
            """Weighted Correlation"""
            return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

        corrs = []
        for col in self.train_data.columns.to_list():
            corrs.append(corr(self.train_data[col], self.train_labels, self.train_weights))

        corr_df = pd.DataFrame({'variable': self.train_data.columns.to_list(), 'correlation': corrs})
        corr_df = corr_df.sort_values(by='correlation', ascending=False)

        sns.barplot(x=corr_df['correlation'].values, y=corr_df['variable'].values)
        plt.xlabel("Correlation with Process")
        plt.yticks(rotation=30)
        plt.savefig('correlation_with_process.png')
        plt.clf()

        # Create a correlation matrix
        corr_matrix = np.zeros((len(self.train_data.columns), len(self.train_data.columns)))
        for i in range(len(self.train_data.columns)):
            for j in range(len(self.train_data.columns)):
                corr_matrix[i][j] = corr(self.train_data[self.train_data.columns[i]], self.train_data[self.train_data.columns[j]], self.train_weights)

        # Plot the correlation matrix
        plt.matshow(corr_matrix, cmap='RdBu')
        plt.xticks(range(len(self.train_data.columns)), self.train_data.columns, rotation='vertical')
        plt.yticks(range(len(self.train_data.columns)), self.train_data.columns)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('correlation_features.png')
        plt.clf()

        class_df = self.train_data.set_index(self.train_labels.index)
        class1_df = class_df[self.train_labels == 0]
        class2_df = class_df[self.train_labels == 1]
        class3_df = class_df[self.train_labels == 2]

        # Create a correlation matrix for each class
        for class_df, class_num in zip([class1_df, class2_df, class3_df], [1, 2, 3]):
            corr_matrix = np.zeros((len(class_df.columns), len(class_df.columns)))
            for i in range(len(class_df.columns)):
                for j in range(len(class_df.columns)):
                    corr_matrix[i][j] = corr(class_df[class_df.columns[i]], class_df[class_df.columns[j]], self.train_weights)

            # Plot the correlation matrix for each class
            plt.matshow(corr_matrix, cmap='RdBu')
            plt.xticks(range(len(class_df.columns)), class_df.columns, rotation='vertical')
            plt.yticks(range(len(class_df.columns)), class_df.columns)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f'correlation_features_class{class_num}.png')
            plt.clf()
        