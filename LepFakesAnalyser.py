import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import array
import ROOT

import glob,os,time,logging
import math,random
import tqdm

import warnings
import matplotlib.cbook
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)

pd.options.mode.chained_assignment = None

# Fixing random seed
random.seed(30)

############################################################
###               Load Helper functions                  ###
############################################################
# General
from LFAHelpers import style, ensure_dir
# read/write data
from LFAHelpers import load_data_rootnp, load_data_RDF
# Histos/Plotting/Pie-Charts
from LFAHelpers import HistMaker, Plotter, PieCharter
# Data pre-processing
from LFAHelpers import sort_taus, TauTempAnnotator, LepTempAnnotator,TempCombAnnotator,RegionsAnnotator
# Fake background
from LFAHelpers import TauFakeYieldCorrector, BkgYieldCorrector, BKGCorrector_Tau, BKGCorrector_TauSyst, BKGCorrector,BKGCorrector_DiLepSyst,BKGCorrector_DiLepSyst_1b2b,BKGCorrector_DiLepSyst_vectorized
from LFAHelpers import SystSolver, SumMCHists,GetAbsVariation, GetUncertGraph, UncertaintyMaker
from LFAHelpers import BKGCorrector_TauUnknown_OS_1b,BKGCorrector_TauSyst_OS_1b,BKGCorrector_TauSyst_OS_2b,BKGCorrector_TauSyst_OS_vectorized
from LFAConfig import Configurate, Configurate_VarHistBins,Configurate_Xtitles_2l1tau
from LFAFitter import Fit_Tau, Fit_DiTau

# Logger
logging.basicConfig(filename='mylog.log', filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s :: %(name)s :: %(levelname)s    %(message)s',
                    datefmt='%H:%M:%S')
# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s %(levelname)-8s %(funcName)-20s %(message)s')
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

# ROOT config with ATLAS style
ROOT.gROOT.SetBatch(True) # No graphics displayed
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetStyle("ATLAS")
ROOT.gErrorIgnoreLevel = ROOT.kWarning

# Start time
start_time = time.time()

logging.info(' ')
logging.info(10*'='+' R u n n i n g  2L1TAU  a n a l y s i s '+10*'=')
logging.info(' ')

############################################################
###              Main script configuration               ###
############################################################

tree_name, dict_samples, dict_tmps_lep, dict_tmps_tau, dict_tmps_tau_fine, list_of_branches_leptons, list_of_branches_mc, input_folder = Configurate('lephad')

### For development
USE_FRAC_DATA = False

### Selection Flags
SS_SELECTION  = False
OS_SELECTION  = True
SUBL_LEP_CUT  = 14   # None or value in [GeV]; applied to lep3 before tau removal

### Plot Flags
PLOT_PROCESS  = True
PLOT_LEPTMP   = False
PLOT_COMBTMP  = True
PLOT_TAUTMP   = True

### specific plot comparing SR and CRs
PLOT_CR_SUMMARY = True

### q/g initiated jets split? (default=True)
SPLIT_JETSTMP = True

### BKG Extractions
EXTRACT_TAUFAKE_YIELDMETHOD = False   # IMPORTANT: this has to be used with SPLIT_JETSTMP=False
EXTRACT_TAUFAKE_TMPMETHOD   = False    # IMPORTANT: to run Template Fits set CORRECT_BKG_TAU=False
EXTRACT_DILEPFAKE           = False   #

### Apply BKG Corrections
CORRECT_BKG_TAU   = True              # Corrections are hardcoded in BKGCorrector_Tau()/BKGCorrector_TauSyst()
CORRECT_BKG_DILEP = True             # Corrections are provided in lists of SF_vals and SF_errs

### Perform MVA training: q-jet vs g-jet
TRAIN_NN   = False       # --> NeuralNetsConfig
TRAIN_LGBM = False       # --> LGBMConfig
USE_CWOLA = False

#####################################################################
####                        Read ntuples                          ###
#####################################################################

logging.info('Considering the following processes: ' + style.YELLOW +', '.join(dict_samples['sample'])+style.RESET )

# Load data from root files
data_df = load_data_RDF(input_folder, dict_samples, list_of_branches_leptons, list_of_branches_mc)

# For development
if USE_FRAC_DATA:
    data_df = data_df.sample(frac=0.1, random_state=1) # replace=True

logging.info("Data loaded: --- {t} seconds ---".format(t = round(time.time()-start_time, 3)))

# Apply lep3 pT cut?
if SUBL_LEP_CUT:
    mask = data_df['pt_lep3']>SUBL_LEP_CUT
    data_df = data_df[mask]

# Sort light leptons and taus
data_df = sort_taus(data_df)

# Annotate Tau1 templates
if SPLIT_JETSTMP:
    dict_tmps_tau = dict_tmps_tau_fine
data_df = TauTempAnnotator(data_df, dict_tmps_tau, 'fs_had_tau_true_pdg', 'fs_had_tau_true_partonTruthLabelID', 'tau_1_tmp', SPLIT_JETSTMP)

# Annotate Lep1 and Lep2 templates
data_df = LepTempAnnotator(data_df, dict_tmps_lep, 'TruthIFF_Class_lep1', 'lep_1_tmp')
data_df = LepTempAnnotator(data_df, dict_tmps_lep, 'TruthIFF_Class_lep2', 'lep_2_tmp')

# Combinations+dictionary
#data_df = TempCombAnnotator(data_df, ['tau_1_tmp','lep_1_tmp','lep_2_tmp'])
data_df = TempCombAnnotator(data_df, ['lep_1_tmp_simple','lep_2_tmp_simple'])
dict_tmps_comb = {'sample': data_df['TempCombinations'].unique().tolist(), 
                  'fillcolor': [x+1 for x in data_df['TempCombinationsEncoded'].unique().tolist()]}
dict_tmps_comb = {'sample': ['data,data', 'sim,sim', 'sim,jet', 'jet,sim', 'jet,jet'], 'fillcolor': [1, 2, 3, 4, 5]}

logging.debug(dict_tmps_comb)
logging.info('Printing pandas dataframe:\n\t'+ data_df['TempCombinations'].tail(10).to_string().replace('\n', '\n\t') )

logging.info("Templates created: --- {t} seconds ---".format(t = round(time.time()-start_time, 3)))

#####################################################################
###                        Selections                             ###
#####################################################################
m_0b = data_df['m_nbjets']==0
#m_1b = (data_df['m_nbjets']==1)*(data_df['fs_had_tau_1_nTrack']==1)
#m_1b = (data_df['m_nbjets']==1)*(data_df['fs_had_tau_1_JetTrackWidth']>0)
#m_1b = (data_df['m_nbjets']==1) | (data_df['m_nbjets']==2)
m_1b = data_df['m_nbjets']==1
m_2b = data_df['m_nbjets']==2
m_tau1_tight = data_df['fs_had_tau_1_tight']==1
m_tau1_1p = data_df['fs_had_tau_1_nTrack']==1
m_tau1_3p = data_df['fs_had_tau_1_nTrack']==3
m_SS_leptons = data_df['charge_lep1']*data_df['charge_lep2']>0
m_tau1_pt1 = (data_df['fs_had_tau_1_pt']>=20)*(data_df['fs_had_tau_1_pt']<30)
m_tau1_pt2 = (data_df['fs_had_tau_1_pt']>=30)*(data_df['fs_had_tau_1_pt']<40)
m_tau1_pt3 = data_df['fs_had_tau_1_pt']>=40
m_dummy = data_df['m_nbjets']>=0

# Apply ECIDS only if SS selection
m_lep1_tight = data_df['isTight_lep1']==1
m_lep2_tight = data_df['isTight_lep2']==1
if SS_SELECTION and not OS_SELECTION:
    m_lep1_tight = (data_df['isTight_lep1']==1) & (data_df['ECIDS_lep1']==1) & (data_df['ele_ambiguity_lep1']<=0) & (data_df['ele_AddAmbiguity_lep1']<=0)
    m_lep2_tight = (data_df['isTight_lep2']==1) & (data_df['ECIDS_lep2']==1) & (data_df['ele_ambiguity_lep2']<=0) & (data_df['ele_AddAmbiguity_lep2']<=0)

my_selection = m_dummy
if SS_SELECTION:
    my_selection = data_df['charge_lep1']*data_df['charge_lep2']>0
if OS_SELECTION:
    my_selection = data_df['charge_lep1']*data_df['charge_lep2']<0

#####################################################################
###                    Annotate lepton pairs                      ###
#####################################################################
masks = [m_1b & m_tau1_tight & (m_lep1_tight & ~m_lep2_tight),
         m_1b & m_tau1_tight & (~m_lep1_tight & m_lep2_tight),
         m_1b & m_tau1_tight & (~m_lep1_tight & ~m_lep2_tight) ]
region_names = ['T#bar{T}', '#bar{T}T', '#bar{T}#bar{T}']
data_df = RegionsAnnotator(data_df, masks, region_names)
data_df.name='_INCL'

logging.info("Regions annotated: --- {t} seconds ---".format(t = round(time.time()-start_time, 3)))

#####################################################################
###                        Apply BKG corrections                  ###
#####################################################################

SystVarWeightTags=[]
SystVarWeightTags_Tau = []
SystVarWeightTags_Dilep = []

### Jets faking tau
if CORRECT_BKG_TAU:
    # Produce weights for each systematic uncertainty (Nominal='' must be the last!)
    # Values are hard-coded in BKGCorrector_TauSyst
    # The corresponding weight name will be 'weight_'+systvar, or weight_nominal in case of nominal.
    if OS_SELECTION:
        SystVarWeightTags_Tau = ['tau_shape_up', 'tau_shape_down',
                                'tau_norm_up', 'tau_norm_down',
                                '']
        for systtag in SystVarWeightTags_Tau:
            #data_df = BKGCorrector_TauSyst_OS_1b(data_df, 'QGMethod', '', systtag) # correct both 1b and 2b separately
            data_df = BKGCorrector_TauSyst_OS_vectorized(data_df, 'QGMethod', '', systtag) # correct both 1b and 2b separately

    if SS_SELECTION and not OS_SELECTION:
        SystVarWeightTags_Tau = ['tau_norm_up', 'tau_norm_down', '']
        for systtag in SystVarWeightTags_Tau:
            # If you want corrections from OS, use QGMethod method here
            #data_df = BKGCorrector_TauSyst_OS_1b(data_df, '1Bin', '', systtag) # correct both 1b and 2b inclusively
            data_df = BKGCorrector_TauSyst_OS_vectorized(data_df, '1Bin', '', systtag) # correct both 1b and 2b inclusively

# Unknown correction (for studies)
if 0>1:
    data_df = BKGCorrector_TauUnknown_OS_1b(data_df,'')

### Jets faking one or both light leptons
if CORRECT_BKG_DILEP:
    # Scale factors for leptons pair combinations (tt,tf,ft,ff)
    # SF_vals = [1.0, 0.909, 0.969, 11.590]
    # SF_errs = [0.0, 0.007, 0.059, 1.599]

    # OS version
    #SF_vals = [1.0, 0.981, 1.143, 10.612]
    #SF_errs = [0.0, 0.016, 0.098, 2.707]
    #data_df = BKGCorrector(data_df, 'TempCombinations', dict_tmps_comb, SF_vals, SF_errs)

    # OS 2b version (original jet,sim = 0.117+-7.072)
    SF_vals = [1.0, 1.676, 1.0, 1.0]
    SF_errs = [0.0, 0.071, 1.0, 1.0]

    SystVarWeightTags_Dilep = ['dilep_simjet_up', 'dilep_simjet_down',
                               'dilep_jetsim_up', 'dilep_jetsim_down',
                               '']
    for systtag in SystVarWeightTags_Dilep:
        #data_df = BKGCorrector_DiLepSyst(data_df, 'TempCombinations', dict_tmps_comb, SF_vals, SF_errs, systtag)
        #data_df = BKGCorrector_DiLepSyst_1b2b(data_df, 'TempCombinations', dict_tmps_comb, systtag)
        if OS_SELECTION:
            #data_df = BKGCorrector_DiLepSyst_1b2b(data_df, 'TempCombinations', dict_tmps_comb, systtag)
            data_df = BKGCorrector_DiLepSyst_vectorized(data_df, 'TempCombinations', dict_tmps_comb, systtag)
        if SS_SELECTION and not OS_SELECTION:
            #data_df = BKGCorrector_DiLepSyst_1b2b(data_df, 'TempCombinations', dict_tmps_comb, systtag, SS_Selection=True)
            data_df = BKGCorrector_DiLepSyst_vectorized(data_df, 'TempCombinations', dict_tmps_comb, systtag, SS_Selection=True)

### Combine weight tags
if CORRECT_BKG_TAU:
    SystVarWeightTags += SystVarWeightTags_Tau
if CORRECT_BKG_DILEP:
    if len(SystVarWeightTags):
        SystVarWeightTags = SystVarWeightTags[:-1]
    SystVarWeightTags += SystVarWeightTags_Dilep

logging.info("Weights corrected: --- {t} seconds ---".format(t = round(time.time()-start_time, 3)))
#####################################################################
###                     Select Regions                            ###
#####################################################################

### Data frames for specific selections
data_1b       = data_df[m_1b & my_selection]    # & m_SS_leptons
data_1b_TauSR = data_df[m_1b & m_tau1_tight  & my_selection]
data_SR       = data_df[m_1b & m_tau1_tight  & m_lep1_tight  & m_lep2_tight  & my_selection] # & m_tau1_tight & m_lep1_tight & m_lep2_tight m_SS_leptons
data_TauCR    = data_df[m_1b & ~m_tau1_tight & m_lep1_tight  & m_lep2_tight  & my_selection] # m_tau1_1p
data_Lep1CR   = data_df[m_1b & m_tau1_tight  & ~m_lep1_tight & m_lep2_tight  & my_selection]
data_Lep2CR   = data_df[m_1b & m_tau1_tight  & m_lep1_tight  & ~m_lep2_tight & my_selection]

### Add tags to each data frame
data_1b.name='_1b'
data_1b_TauSR.name='_1b_tauM'
data_SR.name='_SR'
data_TauCR.name='_Tau1CR'
data_Lep1CR.name='_Lep1CR'
data_Lep2CR.name='_Lep2CR'

#####################################################################
###                         Build Pie Charts                      ###
#####################################################################
logging.info(style.YELLOW+"Plotting Pie-Charts."+style.RESET)
logging.info('Pie-charts will be save in PieCharts/')
ensure_dir('PieCharts/')

### Inclusive
logging.info("Making pie-charts for 'data_SR' dataframe according to 'sample_Id', 'tau_1_tmp', 'lep_1_tmp' and 'lep_2_tmp' splits.")
PieCharter(data_SR, 'sample_Id', dict_samples, "SR_lephad.pdf", show_fractions=True)
PieCharter(data_SR, 'tau_1_tmp', dict_tmps_tau, "SR_tau1tmp_lephad.pdf")
PieCharter(data_SR, 'lep_1_tmp', dict_tmps_lep, "SR_lep1tmp_lephad.pdf")
PieCharter(data_SR, 'lep_2_tmp', dict_tmps_lep, "SR_lep2tmp_lephad.pdf")

### Per Process
logging.info("Making pie-charts for 'sample_Id' per process.")
for iproc, process in enumerate(dict_samples['sample']):
    if process == 'data': continue
    process_cleaned = process.replace('/', '_')
    PieCharter(data_SR[data_SR['sample_Id'] == iproc], 'tau_1_tmp', dict_tmps_tau, 'PieCharts/SR_'+process_cleaned+'_tau1tmp_lephad.pdf')
    PieCharter(data_SR[data_SR['sample_Id'] == iproc], 'lep_1_tmp', dict_tmps_lep, 'PieCharts/SR_'+process_cleaned+'_lep1tmp_lephad.pdf')
    PieCharter(data_SR[data_SR['sample_Id'] == iproc], 'lep_2_tmp', dict_tmps_lep, 'PieCharts/SR_'+process_cleaned+'_lep2tmp_lephad.pdf')


#####################################################################
###                       Neural Network                          ###
#####################################################################
if TRAIN_NN:
    from NeuralNetsConfig import prepare_data,ReduceLROnPlateau,auc,roc_curve,keras
    from NeuralNetsConfig import classifier_model,classifier_model_simple
    X_train, X_test, Y_train, Y_test, w_train, w_test = prepare_data(data_SR)

    num_features = X_train.shape[1]
    print('number of inputs: ',num_features)
    input_shape = (num_features,)

    # Building model
    nn_classifier_model = classifier_model((num_features,))
    opt = keras.optimizers.Nadam(learning_rate=0.01)
    nn_classifier_model.compile(optimizer=opt,
                                loss = "binary_crossentropy",
                                metrics = ["accuracy"])

    # Learning Rate Performance Scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7)

    history = nn_classifier_model.fit(x=X_train, y=Y_train.values, 
                                  sample_weight=w_train.values, 
                                  validation_data = (X_test,Y_test.values, w_test.values),
                                  epochs=50, batch_size=1024, # epochs 50
                                  callbacks=[lr_scheduler] )

    #plt.cla()
    plt.close()
    # Training accuracy
    plt.figure(10, figsize=(8, 5))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.grid(True)
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')

    # Training loss
    plt.figure(11, figsize=(8, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid(True)
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig('loss.png')

    # ROC curve
    y_pred_train = nn_classifier_model.predict(X_train).ravel()
    print(y_pred_train[:10])
    fpr_train, tpr_train, thresholds_train = roc_curve(Y_train.values, y_pred_train, sample_weight=w_train.values)
    auc_train = 0# auc(fpr_train, tpr_train)

    y_pred_test = nn_classifier_model.predict(X_test).ravel()
    fpr_test, tpr_test, thresholds_test = roc_curve(Y_test.values, y_pred_test, sample_weight=w_test.values)
    auc_test = 0# auc(fpr_test, tpr_test)

    plt.figure(12)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_train, tpr_train, label='Train (area = {:.3f})'.format(auc_train))
    plt.plot(fpr_test, tpr_test, label='Test (area = {:.3f})'.format(auc_test))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig('roc.png')  
    
#####################################################################
###                           LightGBM                            ###
#####################################################################
if TRAIN_LGBM:

    from LGBMConfig import prepare_data_lgbm,lgbm_class_model, boosters_predict

    # preselect
    #data_bdt = data_1b[m_tau1_3p]

    X_train, X_test, Y_train, Y_test, w_train, w_test, features = prepare_data_lgbm(data_1b)

    optimal_rounds,best_cv_score,best_models = lgbm_class_model(X_train, X_test, Y_train, Y_test, w_train, w_test)

    data_SR['lgbm_score'] = boosters_predict(best_models, data_SR[features])
    data_TauCR['lgbm_score'] = boosters_predict(best_models, data_TauCR[features])


#####################################################################
###                     Plotting Processes                        ###
#####################################################################
dict_hists = Configurate_VarHistBins()
dict_xtitles = Configurate_Xtitles_2l1tau()

if PLOT_PROCESS:
    for var in ['fs_had_tau_1_pt', 'fs_had_tau_1_eta','fs_had_tau_1_RNNScore','m_njets', 'pt_lep1','eta_lep1', 'pt_lep2','eta_lep2','m_met','NNout_tauFakes']:
        bins,xtitle = dict_hists[var],dict_xtitles[var]
        hists = HistMaker(data_SR, 'process', dict_samples, var, bins, xtitle=xtitle)

        # Purity/Significance
        if var=='NNout_tauFakes':
            HistMaker(data_SR, 'process', dict_samples, var, bins, xtitle=xtitle, PlotPurity='tH')

        # Compute syst variations if available (CORRECT_BKG_TAU=True)
        if CORRECT_BKG_TAU or CORRECT_BKG_DILEP:
            UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(data_SR, 'process', dict_samples, var, bins, SystVarWeightTags[:-1])
            HistMaker(data_SR, 'process', dict_samples, var, bins, xtitle=xtitle, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)

#####################################################################
###                   Lep Fake combined Templates                 ###
#####################################################################
if PLOT_COMBTMP:
    hists = HistMaker(data_1b_TauSR,
                      'TempCombinations',
                      dict_tmps_comb,
                      'regions_encoded', [4,0,4],
                      xtitle='Lepton-pair regions',
                      region_names=['SR=TT']+region_names )

    ### Systematic variations
    if CORRECT_BKG_DILEP:
        UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(data_1b_TauSR, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [4,0,4], SystVarWeightTags[:-1])
        HistMaker(data_1b_TauSR,
                  'TempCombinations',
                  dict_tmps_comb,
                  'regions_encoded', [4,0,4],
                  region_names=['SR=TT']+region_names,
                  xtitle='Lepton-pair regions',
                  UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio )


if EXTRACT_DILEPFAKE:
    # For template fit
    hists = HistMaker(data_1b_TauSR, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [3,1,4], region_names=region_names )
    datacor = Fit_DiTau(data_1b_TauSR.copy(), hists, dict_tmps_comb)
    datacor.name = '_cor'
    # Post-Fit plot
    HistMaker(datacor, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [4,0,4], region_names=['SR']+region_names )

    # Pre-Fit plot
    hists = HistMaker(data_1b_TauSR, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [4,0,4], region_names=['SR']+region_names )
    # Exact solution for XCheck
    SystSolver(hists, dict_tmps_comb, ['SR']+region_names)

#####################################################################
###                       All CRs and SR                          ###
#####################################################################
logging.info("Plots produced: --- {t} seconds ---".format(t = round(time.time()-start_time, 3)))

if PLOT_CR_SUMMARY:
    # For illustration of corrections
    masks_CI = [ m_1b & m_tau1_tight  & (m_lep1_tight  & m_lep2_tight)  & my_selection,   # SR
                 m_1b & ~m_tau1_tight & (m_lep1_tight  & m_lep2_tight)  & my_selection,   # Tau CR
                 m_1b & m_tau1_tight  & (m_lep1_tight  & ~m_lep2_tight) & my_selection,
                 m_1b & m_tau1_tight  & (~m_lep1_tight & m_lep2_tight)  & my_selection,
                 m_1b & m_tau1_tight  & (~m_lep1_tight & ~m_lep2_tight) & my_selection]
    region_names_CI = ['SR=TTM', 'TT#bar{M}', 'T#bar{T}M', '#bar{T}TM', '#bar{T}#bar{T}M']
    data_df_CI = RegionsAnnotator(data_df.copy(), masks_CI, region_names_CI)
    data_df_CI.name='_INCL'

    nbins = len(masks_CI)+1
    hists = HistMaker(data_df_CI,
                      #'process',
                      #dict_samples,
                      'tau_1_tmp', dict_tmps_tau,  # Trick
                      'regions_encoded', [5,1,6],
                      xtitle='Fake Control Regions',
                      region_names=region_names_CI )

    ### compute syst variations if available (CORRECT_BKG_TAU=True)
    if CORRECT_BKG_TAU or CORRECT_BKG_DILEP:
        UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(data_df_CI, 'process', dict_samples, 'regions_encoded', [5,1,6], SystVarWeightTags[:-1])
        HistMaker(data_df_CI,
                  'process',
                  dict_samples,
                  'regions_encoded', [5,1,6],
                  region_names=region_names_CI,
                  xtitle='Fake Control Regions',
                  UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)

logging.info("Annotated SR and CRs: --- {t} seconds ---".format(t = round(time.time()-start_time, 3)))

#####################################################################
###                   Tau Fakes Templates                         ###
#####################################################################

if PLOT_TAUTMP:
    histlist = ['fs_had_tau_1_pt', 'fs_had_tau_1_eta', 'fs_had_tau_true_pdg', 'fs_had_tau_true_partonTruthLabelID','m_njets', 
                'pt_lep1','eta_lep1', 'pt_lep2','eta_lep2','pt_jet1','pt_jet2','m_met','fs_had_tau_1_JetTrackWidth','NNout_tauFakes']
    
    if TRAIN_LGBM:
        histlist.append('lgbm_score')

    # Signal Region
    for var in histlist:
        bins,xtitle = dict_hists[var],dict_xtitles[var]
        #bins = dict_hists[var]
        hists = HistMaker(data_SR, 'tau_1_tmp', dict_tmps_tau, var, bins, xtitle=xtitle)

        # Purity/Significance
        if var=='NNout_tauFakes':
            HistMaker(data_SR, 'tau_1_tmp', dict_tmps_tau, var, bins, xtitle=xtitle, PlotPurity='tau')

        ### compute syst variations if available (CORRECT_BKG_TAU=True)
        if CORRECT_BKG_TAU:
            UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(data_SR, 'tau_1_tmp', dict_tmps_tau, var, bins, SystVarWeightTags[:-1])
            HistMaker(data_SR, 'tau_1_tmp', dict_tmps_tau, var, bins, xtitle=xtitle, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)

    # Tau Control Region
    for var in histlist:
        bins,xtitle = dict_hists[var],dict_xtitles[var]
        hists = HistMaker(data_TauCR, 'tau_1_tmp', dict_tmps_tau, var, bins, xtitle=xtitle)

        ### compute syst variations if available (CORRECT_BKG_TAU=True)
        if CORRECT_BKG_TAU:
            UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(data_TauCR, 'tau_1_tmp', dict_tmps_tau, var, bins, SystVarWeightTags[:-1])
            HistMaker(data_TauCR, 'tau_1_tmp', dict_tmps_tau, var, bins, xtitle=xtitle, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)

    # Inclusive region for RNN Score
    histlist = ['fs_had_tau_1_RNNScore']
    data_Incl = pd.concat([data_SR, data_TauCR])
    data_Incl_1p = data_Incl[data_Incl['fs_had_tau_1_nTrack']==1]
    data_Incl_1p.name = '_noID_1p'
    data_Incl_3p = data_Incl[data_Incl['fs_had_tau_1_nTrack']==3]
    data_Incl_3p.name = '_noID_3p'

    for data in [data_Incl_1p, data_Incl_3p]:
        for var in histlist:
            bins,xtitle = dict_hists[var],dict_xtitles[var]
            hists = HistMaker(data, 'tau_1_tmp', dict_tmps_tau, var, bins, xtitle=xtitle)

            ### compute syst variations if available (CORRECT_BKG_TAU=True)
            if CORRECT_BKG_TAU:
                UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(data, 'tau_1_tmp', dict_tmps_tau, var, bins, SystVarWeightTags[:-1])
                HistMaker(data, 'tau_1_tmp', dict_tmps_tau, var, bins, xtitle=xtitle, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)


#####################################################################
###                        Lep1 Templates                         ###
#####################################################################

if PLOT_LEPTMP:
    for var in ['m_njets', 'pt_lep1','eta_lep1','TruthIFF_Class_lep1']:
        bins,xtitle = dict_hists[var],dict_xtitles[var]
        hists = HistMaker(data_SR, 'lep_1_tmp', dict_tmps_lep, var, bins, xtitle=xtitle)

        # Syst variations
        if CORRECT_BKG_DILEP:
            UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(data_SR, 'lep_1_tmp', dict_tmps_lep, var, bins, SystVarWeightTags[:-1])
            HistMaker(data_SR, 'lep_1_tmp', dict_tmps_lep, var, bins, xtitle=xtitle, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)

    for var in ['m_njets', 'pt_lep1','eta_lep1','TruthIFF_Class_lep1']:
        bins,xtitle = dict_hists[var],dict_xtitles[var]
        hists = HistMaker(data_Lep1CR, 'lep_1_tmp', dict_tmps_lep, var, bins, xtitle=xtitle)

        # Syst variations
        if CORRECT_BKG_DILEP:
            UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(data_Lep1CR, 'lep_1_tmp', dict_tmps_lep, var, bins, SystVarWeightTags[:-1])
            HistMaker(data_Lep1CR, 'lep_1_tmp', dict_tmps_lep, var, bins, xtitle=xtitle, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)


#####################################################################
###                        Lep2 Templates                         ###
#####################################################################
if PLOT_LEPTMP:
    for var in ['m_njets', 'pt_lep2','eta_lep2','TruthIFF_Class_lep2']:
        bins,xtitle = dict_hists[var],dict_xtitles[var]
        hists = HistMaker(data_SR, 'lep_2_tmp', dict_tmps_lep, var, bins, xtitle=xtitle)

        # Syst variations
        if CORRECT_BKG_DILEP:
            UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(data_SR, 'lep_2_tmp', dict_tmps_lep, var, bins, SystVarWeightTags[:-1])
            HistMaker(data_SR, 'lep_2_tmp', dict_tmps_lep, var, bins, xtitle=xtitle, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)

    for var in ['m_njets', 'pt_lep2','eta_lep2','TruthIFF_Class_lep2']:
        bins,xtitle = dict_hists[var],dict_xtitles[var]
        hists = HistMaker(data_Lep2CR, 'lep_2_tmp', dict_tmps_lep, var, bins, xtitle=xtitle)

        # Syst variations
        if CORRECT_BKG_DILEP:
            UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(data_Lep2CR, 'lep_2_tmp', dict_tmps_lep, var, bins, SystVarWeightTags[:-1])
            HistMaker(data_Lep2CR, 'lep_2_tmp', dict_tmps_lep, var, bins, xtitle=xtitle, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)

    # This does not moodify dataframe, but prints BKG SFs for X-Check
    # bins = dict_hists['pt_lep2']
    # xtitle = dict_xtitles['pt_lep2']
    # hists_fit = HistMaker(data_Lep2CR, 'lep_2_tmp', dict_tmps_lep, 'pt_lep2', bins, xtitle=xtitle)
    # BkgYieldCorrector(hists_fit)
    # if CORRECT_BKG_DILEP:
    #     UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(data_Lep2CR, 'lep_2_tmp', dict_tmps_lep, var, bins, SystVarWeightTags[:-1])
    #     HistMaker(data_Lep2CR, 'lep_2_tmp', dict_tmps_lep, 'pt_lep2', bins, xtitle=xtitle, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)


#####################################################################
###                      TAU BKG Extraction                       ###
#####################################################################
if EXTRACT_TAUFAKE_YIELDMETHOD:

    #### inclusive 
    print(style.RED+"Inclusive histogram."+style.RESET)
    bins,xtitle = dict_hists['m_njets'],dict_xtitles['m_njets']
    hists_yields = HistMaker(data_TauCR, 'tau_1_tmp', dict_tmps_tau, 'm_njets', bins, xtitle=xtitle)
    TauFakeYieldCorrector(hists_yields, unknown_is_jet=False)

    #### bins
    bins = dict_hists['m_njets']
    for prong in [1, 3]:
        for ptbin in [1,2,3]:
            print(style.RED+str(prong)+'-prong, tau pT'+str(ptbin)+style.RESET)

            prong_mask = m_tau1_1p if prong==1 else m_tau1_3p
            pt_mask = m_tau1_pt1 if ptbin==1 else m_tau1_pt2 if ptbin==2 else m_tau1_pt3

            subdf = data_TauCR[prong_mask & pt_mask]
            subdf.name='_Tau1CR_'+str(prong)+'p_pt'+str(ptbin)
            hists_yields = HistMaker(subdf, 'tau_1_tmp', dict_tmps_tau, 'm_njets', bins)
            TauFakeYieldCorrector(hists_yields, unknown_is_jet=False)

if EXTRACT_TAUFAKE_TMPMETHOD:

    ### Produce Post-Fit plots for JetTrackWidth
    if CORRECT_BKG_TAU:
        print(style.YELLOW+'-----------------Measuring tau background: producing JetTrackWidth post-fit plots after Template Fit------------'+style.RESET)
        for prong in [1, 3]:
            for ptbin in [1,2,3]:
                print(style.GREEN+str(prong)+'-prong, tau pT'+str(ptbin)+style.RESET)
                bins_varBinSize = dict_hists['fs_had_tau_1_JetTrackWidth_p'+str(prong)+'pt'+str(ptbin)+'_pf'] # With optimized binning
                xtitle_varBinSize = dict_xtitles['fs_had_tau_1_JetTrackWidth_p'+str(prong)+'pt'+str(ptbin)+'_pf']

                prong_mask = m_tau1_1p if prong==1 else m_tau1_3p
                pt_mask = m_tau1_pt1 if ptbin==1 else m_tau1_pt2 if ptbin==2 else m_tau1_pt3

                subdf = data_TauCR[prong_mask & pt_mask]
                subdf.name='_Tau1CR_'+str(prong)+'p_pt'+str(ptbin)

                # Draw histograms with optimal binning
                hists_tmp = HistMaker(subdf, 'tau_1_tmp', dict_tmps_tau, 'fs_had_tau_1_JetTrackWidth', bins_varBinSize, xtitle=xtitle_varBinSize)
                NormF, NormF_err, SF_unknown = Fit_Tau(hists_tmp, DEBUG=False)
                # Print fit results
                print("Estimated q-SF: %.3f $\pm$ %.3f" % (NormF[0],NormF_err[0]) )
                print("Estimated g-SF: %.3f $\pm$ %.3f" % (NormF[1],NormF_err[1]) )
                print("Estimated unknown inefficiency SF: %.3f" % (SF_unknown) )

                # Syst variations
                UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(subdf, 'tau_1_tmp', dict_tmps_tau, 'fs_had_tau_1_JetTrackWidth', bins_varBinSize, SystVarWeightTags[:-1])
                HistMaker(subdf, 'tau_1_tmp', dict_tmps_tau, 'fs_had_tau_1_JetTrackWidth', bins_varBinSize,
                          xtitle=xtitle_varBinSize,
                          UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)

    #### Runs the Template Fits (CORRECT_BKG_TAU should be set False!)
    if not CORRECT_BKG_TAU:
        print(style.YELLOW+'-----------------Measuring tau background: running Template Fit---------------------'+style.RESET)
        for prong in [1, 3]:
            for ptbin in [1,2,3]:
                print(style.RED+str(prong)+'-prong, tau pT'+str(ptbin)+style.RESET)
                bins_varBinSize = dict_hists['fs_had_tau_1_JetTrackWidth_p'+str(prong)+'pt'+str(ptbin)]

                prong_mask = m_tau1_1p if prong==1 else m_tau1_3p
                pt_mask = m_tau1_pt1 if ptbin==1 else m_tau1_pt2 if ptbin==2 else m_tau1_pt3

                subdf = data_TauCR[prong_mask & pt_mask]
                subdf.name='_Tau1CR_'+str(prong)+'p_pt'+str(ptbin)
                bins_varBinSize_min = bins_varBinSize.copy()

                # Find optimal binning (partially manual tune)
                sigma_min = 100
                NormF_min, NormF_err_min = None, None
                for i in range(200):
                    n = random.randint(1,10)
                    if prong==3 and ptbin==2:
                        n = random.randint(1,7)
                    bins = sorted(random.sample(bins_varBinSize, len(bins_varBinSize)-n))
                    if 0.0 not in bins: bins=[0.0]+bins
                    hists_tmp = HistMaker(subdf, 'tau_1_tmp', dict_tmps_tau, 'fs_had_tau_1_JetTrackWidth', bins)
                    NormF, NormF_err, _ = Fit_Tau(hists_tmp)
                    sigma = math.sqrt(NormF_err[0]*NormF_err[0] + NormF_err[1]*NormF_err[1])
                    if sigma_min>sigma and NormF[1]>0.6 and NormF[1]<1.3:
                        sigma_min = sigma
                        NormF_min,NormF_err_min = NormF,NormF_err
                        bins_varBinSize_min = bins.copy()

                # Draw histograms with optimal binning
                hists_tmp = HistMaker(subdf, 'tau_1_tmp', dict_tmps_tau, 'fs_had_tau_1_JetTrackWidth', bins_varBinSize_min)
                NormF, NormF_err, SF_unknown = Fit_Tau(hists_tmp, DEBUG=False)

                # If the previous loop fails to find a good fit:
                if not NormF_min:
                    NormF_min = NormF
                    NormF_err_min = NormF_err
                # Print fit results
                print("Estimated q-SF: %.3f $\pm$ %.3f" % (NormF_min[0], NormF_err_min[0]) )
                print("Estimated g-SF: %.3f $\pm$ %.3f" % (NormF_min[1], NormF_err_min[1]) )
                print("Estimated unknown inefficiency SF: %.3f" % (SF_unknown) )
                # Binning to be used for Post-Fit plots
                print('The optimal binning: ', bins_varBinSize_min)

    #### Runs the Template Fits on BDT score (CORRECT_BKG_TAU should be set False!)
    if not CORRECT_BKG_TAU and TRAIN_LGBM:
        print(style.YELLOW+'-----------------Measuring tau background: running Template Fit---------------------'+style.RESET)
        for prong in [1, 3]:
            for ptbin in [1,2,3]:
                print(style.RED+str(prong)+'-prong, tau pT'+str(ptbin)+style.RESET)
                bins_varBinSize = dict_hists['lgbm_score_p'+str(prong)+'pt'+str(ptbin)]

                prong_mask = m_tau1_1p if prong==1 else m_tau1_3p
                pt_mask = m_tau1_pt1 if ptbin==1 else m_tau1_pt2 if ptbin==2 else m_tau1_pt3

                subdf = data_TauCR[prong_mask & pt_mask]
                subdf.name='_Tau1CR_'+str(prong)+'p_pt'+str(ptbin)
                bins_varBinSize_min = bins_varBinSize.copy()

                # Draw histograms with optimal binning
                hists_tmp = HistMaker(subdf, 'tau_1_tmp', dict_tmps_tau, 'lgbm_score', bins_varBinSize_min)
                NormF, NormF_err, SF_unknown = Fit_Tau(hists_tmp, DEBUG=False)

                # Print fit results
                print("Estimated q-SF: %.3f $\pm$ %.3f" % (NormF[0], NormF_err[0]) )
                print("Estimated g-SF: %.3f $\pm$ %.3f" % (NormF[1], NormF_err[1]) )
                print("Estimated unknown inefficiency SF: %.3f" % (SF_unknown) )
                # Binning to be used for Post-Fit plots
                print('The optimal binning: ', bins_varBinSize_min)
    
    if CORRECT_BKG_TAU and TRAIN_LGBM:
        print(style.YELLOW+'-----------------Measuring tau background: running Template Fit---------------------'+style.RESET)
        for prong in [1, 3]:
            for ptbin in [1,2,3]:
                print(style.RED+str(prong)+'-prong, tau pT'+str(ptbin)+style.RESET)
                bins_varBinSize = dict_hists['lgbm_score_p'+str(prong)+'pt'+str(ptbin)]
                xtitle_varBinSize = dict_xtitles['lgbm_score']

                prong_mask = m_tau1_1p if prong==1 else m_tau1_3p
                pt_mask = m_tau1_pt1 if ptbin==1 else m_tau1_pt2 if ptbin==2 else m_tau1_pt3

                subdf = data_TauCR[prong_mask & pt_mask]
                subdf.name='_Tau1CR_'+str(prong)+'p_pt'+str(ptbin)
                bins_varBinSize_min = bins_varBinSize.copy()

                # Draw histograms with optimal binning
                hists_tmp = HistMaker(subdf, 'tau_1_tmp', dict_tmps_tau, 'lgbm_score', bins_varBinSize_min, xtitle=xtitle_varBinSize)
                NormF, NormF_err, SF_unknown = Fit_Tau(hists_tmp, DEBUG=False)

                # Print fit results
                print("Estimated q-SF: %.3f $\pm$ %.3f" % (NormF[0], NormF_err[0]) )
                print("Estimated g-SF: %.3f $\pm$ %.3f" % (NormF[1], NormF_err[1]) )
                print("Estimated unknown inefficiency SF: %.3f" % (SF_unknown) )

                # Syst variations
                UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(subdf, 'tau_1_tmp', dict_tmps_tau, 'lgbm_score', bins_varBinSize, SystVarWeightTags[:-1])
                HistMaker(subdf, 'tau_1_tmp', dict_tmps_tau, 'lgbm_score', bins_varBinSize,
                          xtitle=xtitle_varBinSize,
                          UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)

    # #### Runs the Template Fits on CWOLA score (CORRECT_BKG_TAU should be set False!)
    # if not CORRECT_BKG_TAU and USE_CWOLA:
    #     print(style.YELLOW+'-----------------Measuring tau background: running Template Fit on CWOLA score ---------------------'+style.RESET)
    #     for prong in [1, 3]:
    #         for ptbin in [1,2,3]:
    #             print(style.RED+str(prong)+'-prong, tau pT'+str(ptbin)+style.RESET)
    #             bins_varBinSize = dict_hists['NNout_tauFakes_p'+str(prong)+'pt'+str(ptbin)]

    #             prong_mask = m_tau1_1p if prong==1 else m_tau1_3p
    #             pt_mask = m_tau1_pt1 if ptbin==1 else m_tau1_pt2 if ptbin==2 else m_tau1_pt3

    #             subdf = data_TauCR[prong_mask & pt_mask]
    #             subdf.name='_Tau1CR_'+str(prong)+'p_pt'+str(ptbin)
    #             bins_varBinSize_min = bins_varBinSize.copy()

    #             # Draw histograms with optimal binning
    #             hists_tmp = HistMaker(subdf, 'tau_1_tmp', dict_tmps_tau, 'NNout_tauFakes', bins_varBinSize_min)
    #             NormF, NormF_err, SF_unknown = Fit_Tau(hists_tmp, DEBUG=False)

    #             # Print fit results
    #             print("Estimated q-SF: %.3f $\pm$ %.3f" % (NormF[0], NormF_err[0]) )
    #             print("Estimated g-SF: %.3f $\pm$ %.3f" % (NormF[1], NormF_err[1]) )
    #             print("Estimated unknown inefficiency SF: %.3f" % (SF_unknown) )
    #             # Binning to be used for Post-Fit plots
    #             print('The optimal binning: ', bins_varBinSize_min)


logging.info("Program finished: --- {t} seconds ---".format(t = round(time.time()-start_time, 3)))