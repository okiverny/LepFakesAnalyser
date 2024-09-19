import numpy as np
import pandas as pd
import array
import ROOT
import matplotlib.pyplot as plt

import ctypes
import glob,os,time
import math,random
import logging
import yaml
import argparse

import warnings
import matplotlib.cbook
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
#warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)

pd.options.mode.chained_assignment = None

# Fixing random seed
random.seed(30)

############################################################
###                       Arguments                      ###
############################################################
argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--config", type=str, default='config_2l1tau_nominal.yaml', help="yaml config file", metavar='CONFIG')
args = argParser.parse_args()

print("args=%s" % args)

############################################################
###               Loading Helper functions                  ###
############################################################
# General
from LFAHelpers import style, ensure_dir
# read/write data
from DataReader import load_data_RDF

from LFAConfig import Configurate, Configurate_VarHistBins,Configurate_Xtitles, get_fakesSFs_default
from DataModifiers import TauTempAnnotator, LepTempAnnotator, TempCombAnnotator, RegionsAnnotator, correct_fakes
from Plotter import HistMaker, UncertaintyMaker

############################################################
###                 Setting the Logger                   ###
############################################################
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

############################################################
###              ROOT config with ATLAS style            ###
############################################################
ROOT.gROOT.SetBatch(True) # No graphics displayed
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetStyle("ATLAS")
ROOT.gErrorIgnoreLevel = ROOT.kWarning

def main():
    ############################################################
    ###               Reading Configuration file             ###
    ############################################################
    with open(args.config, 'r') as f:
        # Load the configuration file
        config = yaml.safe_load(f)

    # Convert possible lists of values
    from LFAHelpers import config_parse
    config = config_parse(config)
    print(config)

    # Setup dictionarry with corrections
    from LFAHelpers import get_fakesSFs
    config_fakeSFs = get_fakesSFs(config)


    ############################################################
    ###                 Read ntuples                         ###
    ############################################################
    start_time = time.time()
    logging.info(' ')
    logging.info(10*'='+' R u n n i n g  ' + config['AnalysisChannel'] + '  a n a l y s i s '+10*'=')
    logging.info(' ')

    tree_name, dict_samples, dict_tmps_lep, dict_tmps_tau, dict_tmps_tau_fine, list_of_branches_leptons, list_of_branches_mc, input_folder = Configurate(config['AnalysisChannel'])

    # Overwrite the input folder by the one from config file
    input_folder = config['InputDir']

    logging.info('Considering the following processes: ' + style.YELLOW +', '.join(dict_samples['sample'])+style.RESET )

    # Load data from root files
    data_df = load_data_RDF(input_folder, dict_samples, list_of_branches_leptons, list_of_branches_mc, tree_name=tree_name)

    # Rename the weight_XXX to weight_XXX_fix
    if 'weight_nominalWtau_fix' in data_df.columns:
        data_df = data_df.rename(columns={'weight_nominalWtau_fix': 'weight_nominal'})

    # For code development
    if config['USE_FRAC_DATA']:
        data_df = data_df.sample(frac=0.1, random_state=1) # replace=True

    logging.info("Data loaded: --- {t} seconds ---".format(t = round(time.time()-start_time, 3)))

    # Apply some additional selection?
    if config['AnalysisChannel']=='lephad':
        from DataModifiers import apply_lep3_cut
        data_df = apply_lep3_cut(data_df, config['SUBL_LEP_CUT'])

    # Split taus and light leptons
    if config['AnalysisChannel']=='lephad':
        from DataModifiers import sort_taus
        data_df = sort_taus(data_df)
    elif config['AnalysisChannel']=='hadhad':
        from DataModifiers import sort_taus_ditau
        data_df = sort_taus_ditau(data_df)

    ############################################################
    ###                 Leptons labeling                     ###
    ############################################################
    # Labeling taus and light leptons according to truth information

    if config['AnalysisChannel']=='lephad':
        # Labeling taus
        if config['SPLIT_JETSTMP']:
            data_df = TauTempAnnotator(data_df, dict_tmps_tau_fine, 'fs_had_tau_true_pdg', 'fs_had_tau_true_partonTruthLabelID', 'tau_1_tmp', config['SPLIT_JETSTMP'])
            # create also q-jet and g-jet merged version of templator
            data_df = TauTempAnnotator(data_df, dict_tmps_tau, 'fs_had_tau_true_pdg', 'fs_had_tau_true_partonTruthLabelID', 'tau_1_tmp_qgcomb', False)
            # remove duplicated column
            data_df.drop('tau_1_tmp_qgcomb_simple', axis=1)
        else:
            data_df = TauTempAnnotator(data_df, dict_tmps_tau, 'fs_had_tau_true_pdg', 'fs_had_tau_true_partonTruthLabelID', 'tau_1_tmp', config['SPLIT_JETSTMP'])
        # Labeling Lep1 and Lep2 templates
        data_df = LepTempAnnotator(data_df, dict_tmps_lep, 'TruthIFF_Class_lep1', 'lep_1_tmp')
        data_df = LepTempAnnotator(data_df, dict_tmps_lep, 'TruthIFF_Class_lep2', 'lep_2_tmp')

    if config['AnalysisChannel']=='hadhad':
        # Labeling taus
        if config['SPLIT_JETSTMP']:
            data_df = TauTempAnnotator(data_df, dict_tmps_tau_fine, 'fs_had_tau_true_pdg', 'fs_had_tau_true_partonTruthLabelID', 'tau_1_tmp', config['SPLIT_JETSTMP'])
            data_df = TauTempAnnotator(data_df, dict_tmps_tau_fine, 'fs_had_tau_2_true_pdg', 'fs_had_tau_2_true_partonTruthLabelID', 'tau_2_tmp', config['SPLIT_JETSTMP'])
            # create also q-jet and g-jet merged version of templators
            data_df = TauTempAnnotator(data_df, dict_tmps_tau, 'fs_had_tau_true_pdg', 'fs_had_tau_true_partonTruthLabelID', 'tau_1_tmp_qgcomb', False)
            data_df = TauTempAnnotator(data_df, dict_tmps_tau, 'fs_had_tau_2_true_pdg', 'fs_had_tau_2_true_partonTruthLabelID', 'tau_2_tmp_qgcomb', False)
            # remove duplicated columns
            data_df.drop('tau_1_tmp_qgcomb_simple', axis=1)
            data_df.drop('tau_2_tmp_qgcomb_simple', axis=1)
        else:
            data_df = TauTempAnnotator(data_df, dict_tmps_tau, 'fs_had_tau_true_pdg', 'fs_had_tau_true_partonTruthLabelID', 'tau_1_tmp', config['SPLIT_JETSTMP'])
            data_df = TauTempAnnotator(data_df, dict_tmps_tau, 'fs_had_tau_2_true_pdg', 'fs_had_tau_2_true_partonTruthLabelID', 'tau_2_tmp', config['SPLIT_JETSTMP'])
        # Annotate Lep1 templates
        data_df = LepTempAnnotator(data_df, dict_tmps_lep, 'TruthIFF_Class_lep1', 'lep_1_tmp')

    # Set the default splitting
    if config['SPLIT_JETSTMP']:
        dict_tmps_qgcomb = dict_tmps_tau
        dict_tmps_tau = dict_tmps_tau_fine

    ############################################################
    ###         Lepton-pairs/tau-pairs labeling              ###
    ############################################################
    pair = ['lep_1_tmp_simple','lep_2_tmp_simple'] if config['AnalysisChannel']=='lephad' else ['tau_1_tmp_simple','tau_2_tmp_simple']
    data_df = TempCombAnnotator(data_df, pair)
    # Force the correct order ['data,data', 'sim,sim', 'sim,jet', 'jet,sim', 'jet,jet']!
    dict_tmps_comb = {'sample': data_df['TempCombinations'].unique().tolist(),
                  'fillcolor': [x+1 for x in data_df['TempCombinationsEncoded'].unique().tolist()]}
    dict_tmps_comb = {'sample': ['data,data', 'sim,sim', 'sim,jet', 'jet,sim', 'jet,jet'], 'fillcolor': [1, 4, 800, 800-5, 800+9]}

    logging.debug(dict_tmps_comb)
    logging.info("Templates created: --- {t} seconds ---".format(t = round(time.time()-start_time, 3)))


    #####################################################################
    ###                        Selections                             ###
    #####################################################################
    # mask_jt = data_df.fs_had_tau_1_JetTrackWidth < 0
    # data_df.loc[mask_jt, 'fs_had_tau_1_JetTrackWidth'] = -0.01

    # b-jets
    m_bjets = data_df['m_nbjets'].isin(config['BJETS'])
    logging.info('Selecting events with the following number of b-jets: {}'.format(config['BJETS']))
    m_1bjet = data_df['m_nbjets'] == 1
    m_2bjet = data_df['m_nbjets'] == 2

    # Tau ID
    m_tau1_tight, m_tau2_tight = data_df['fs_had_tau_1_tight']==1, None

    # Tau prongs
    m_tau1_1p, m_tau1_3p = data_df['fs_had_tau_1_nTrack']==1, data_df['fs_had_tau_1_nTrack']==3
    m_tau2_1p, m_tau2_3p = None, None

    # Leading tau pT ranges
    m_tau1_pt1 = (data_df['fs_had_tau_1_pt']>=20)*(data_df['fs_had_tau_1_pt']<30)
    m_tau1_pt2 = (data_df['fs_had_tau_1_pt']>=30)*(data_df['fs_had_tau_1_pt']<40)
    m_tau1_pt3 = data_df['fs_had_tau_1_pt']>=40

    # Light leptons ID + Iso
    m_lep1_tight, m_lep2_tight = data_df['isTight_lep1']==1, None

    # Dummy DF for dilep charge selection
    dicharge_selection = data_df['m_nbjets']>=0

    if config['AnalysisChannel']=='lephad':
        m_lep2_tight = data_df['isTight_lep2']==1

        logging.info('Applying {} selection for lepton pair charges.'.format(config['DilepSelection']))
        if config['DilepSelection']=='SS':
            # Dilep SS charge requirement
            dicharge_selection = data_df['charge_lep1']*data_df['charge_lep2']>0
            # Apply ECIDS only if SS selection
            logging.info("Applying 'ECIDS=True', 'ele_ambiguity<=0' and 'ele_AddAmbiguity<=0' cuts to both leptons.")
            dicharge_selection = dicharge_selection & (data_df['ECIDS_lep1']==1) & (data_df['ele_ambiguity_lep1']<=0) & (data_df['ele_AddAmbiguity_lep1']<=0)
            dicharge_selection = dicharge_selection & (data_df['ECIDS_lep2']==1) & (data_df['ele_ambiguity_lep2']<=0) & (data_df['ele_AddAmbiguity_lep2']<=0)

            # tau charge opposite to light leptons?
            dicharge_selection = dicharge_selection & (data_df['had_tau_1_charge'] == -data_df['charge_lep1'])

        elif config['DilepSelection']=='OS':
            # Dilep OS charge requirement
            dicharge_selection = data_df['charge_lep1']*data_df['charge_lep2']<0
        
    elif config['AnalysisChannel']=='hadhad':
        m_tau2_tight = data_df['fs_had_tau_2_tight']==1
        m_tau2_1p = data_df['fs_had_tau_2_nTrack']==1
        m_tau2_3p = data_df['fs_had_tau_2_nTrack']==3

    #####################################################################
    ###      Annotate Regions according to lepton pairs/tau pairs     ###
    #####################################################################
    if config['AnalysisChannel']=='lephad':
        masks = [m_bjets & m_tau1_tight & (m_lep1_tight & ~m_lep2_tight),
                 m_bjets & m_tau1_tight & (~m_lep1_tight & m_lep2_tight),
                 m_bjets & m_tau1_tight & (~m_lep1_tight & ~m_lep2_tight) ]
        #region_names = ['T#bar{T}', '#bar{T}T', '#bar{T}#bar{T}']
        region_names = ['#it{lep}^{pass}#it{lep}^{fail}','#it{lep}^{fail}#it{lep}^{pass}','#it{lep}^{fail}#it{lep}^{fail}']
        data_df = RegionsAnnotator(data_df, masks, region_names)
        data_df.name='_INCL'
    elif config['AnalysisChannel']=='hadhad':
        masks = [m_bjets & m_lep1_tight & (m_tau1_tight & ~m_tau2_tight),
                 m_bjets & m_lep1_tight & (~m_tau1_tight & m_tau2_tight),
                 m_bjets & m_lep1_tight & (~m_tau1_tight & ~m_tau2_tight)]
        #region_names = ['[M,#bar{M}]', '[#bar{M},M]','[#bar{M},#bar{M}]']
        region_names = ['#tau_{had}^{pass}#tau_{had}^{fail}', '#tau_{had}^{fail}#tau_{had}^{pass}','#tau_{had}^{fail}#tau_{had}^{fail}']
        data_df = RegionsAnnotator(data_df, masks, region_names)
        data_df.name='_INCL'


    #####################################################################
    ###                        Apply BKG corrections                  ###
    #####################################################################

    data_df, SystVarWeightTags = correct_fakes(data_df, dict_tmps_comb, config)
    logging.info("Weights corrected: --- {t} seconds ---".format(t = round(time.time()-start_time, 3)))

    #####################################################################
    ###                     Define Regions                            ###
    #####################################################################
    nbjets_tag = ''.join([str(b) for b in config['BJETS']])
    data_dict = {}

    if config['AnalysisChannel']=='lephad':
        data_dict['data_nb']       = data_df[m_bjets & dicharge_selection]
        data_dict['data_nb_TauSR'] = data_df[m_bjets & m_tau1_tight  & dicharge_selection]
        data_dict['data_SR']       = data_df[m_bjets & m_tau1_tight  & m_lep1_tight  & m_lep2_tight  & dicharge_selection]
        data_dict['data_Tau1CR']    = data_df[m_bjets & ~m_tau1_tight & m_lep1_tight  & m_lep2_tight  & dicharge_selection]
        data_dict['data_Lep1CR']   = data_df[m_bjets & m_tau1_tight  & ~m_lep1_tight & m_lep2_tight  & dicharge_selection]
        data_dict['data_Lep2CR']   = data_df[m_bjets & m_tau1_tight  & m_lep1_tight  & ~m_lep2_tight & dicharge_selection]

        # pdg_mask = (data_df.process=='data')+((data_df.process!='data')*(data_df['fs_had_tau_true_pdg']==0))
        # data_dict['data_Tau1CR_1b_1p'] = data_df[m_bjets & ~m_tau1_tight & m_lep1_tight  & m_lep2_tight  & dicharge_selection & m_tau1_1p & m_1bjet & m_tau1_pt1 & pdg_mask]
        # data_dict['data_Tau1CR_1b_1p'].name = '_Tau1CR_1b_1p_pt1'

        ### Add tags to each data frame
        data_dict['data_nb'].name='_'+nbjets_tag+'b'
        data_dict['data_nb_TauSR'].name='_'+nbjets_tag+'b_tauM'
        data_dict['data_SR'].name='_SR'
        data_dict['data_Tau1CR'].name='_Tau1CR'
        data_dict['data_Lep1CR'].name='_Lep1CR'
        data_dict['data_Lep2CR'].name='_Lep2CR'

    elif config['AnalysisChannel']=='hadhad':
        data_dict['data_nb']       = data_df[m_bjets]
        data_dict['data_nb_LepSR'] = data_df[m_bjets & m_lep1_tight]
        data_dict['data_SR']       = data_df[m_bjets & m_lep1_tight & m_tau1_tight & m_tau2_tight]
        data_dict['data_Tau1CR']   = data_df[m_bjets & ~m_tau1_tight & m_tau2_tight & m_lep1_tight]
        data_dict['data_Tau2CR']   = data_df[m_bjets & m_tau1_tight & ~m_tau2_tight & m_lep1_tight]
        data_dict['data_Lep1CR']   = data_df[m_bjets & m_tau1_tight & m_tau2_tight & ~m_lep1_tight]

        # Names of data frames as tags
        data_dict['data_nb'].name='_'+nbjets_tag+'b'
        data_dict['data_nb_LepSR'].name='_'+nbjets_tag+'b_lepT'
        data_dict['data_SR'].name='_SR'
        data_dict['data_Tau1CR'].name='_Tau1CR'
        data_dict['data_Tau2CR'].name='_Tau2CR'
        data_dict['data_Lep1CR'].name='_Lep1CR'

    #####################################################################
    ###                         Plot Pie Charts                       ###
    #####################################################################
    if config['PIECHARTS']:
        logging.info(style.YELLOW+"Plotting Pie-Charts."+style.RESET)
        logging.info('Pie-charts will be saved to PieCharts/')
        ensure_dir('PieCharts/')

        from Plotter import PieCharter

        data_SR = data_dict['data_SR']

        if config['AnalysisChannel']=='lephad':
            logging.info("Making pie-charts for 'data_SR' dataframe according to 'sample_Id', 'tau_1_tmp', 'lep_1_tmp' and 'lep_2_tmp' splits.")
        elif config['AnalysisChannel']=='hadhad':
            logging.info("Making pie-charts for 'data_SR' dataframe according to 'sample_Id', 'tau_1_tmp', 'tau_2_tmp' and 'lep_1_tmp' splits.")

        PieCharter(data_SR, 'sample_Id', dict_samples, "PieCharts/SR_"+config['AnalysisChannel']+".pdf", show_fractions=True)
        PieCharter(data_SR, 'tau_1_tmp', dict_tmps_tau, "PieCharts/SR_tau1tmp_"+config['AnalysisChannel']+".pdf")
        PieCharter(data_SR, 'lep_1_tmp', dict_tmps_lep, "PieCharts/SR_lep1tmp_"+config['AnalysisChannel']+".pdf")
        if config['AnalysisChannel']=='hadhad': PieCharter(data_SR, 'tau_2_tmp', dict_tmps_tau, "PieCharts/SR_tau2tmp_"+config['AnalysisChannel']+".pdf")
        if config['AnalysisChannel']=='lephad': PieCharter(data_SR, 'lep_2_tmp', dict_tmps_lep, "PieCharts/SR_lep2tmp_"+config['AnalysisChannel']+".pdf")

        if config['AnalysisChannel']=='hadhad': PieCharter(data_SR, 'TempCombinations', dict_tmps_comb, "PieCharts/SR_comb_"+config['AnalysisChannel']+".pdf")

        ### Per Process
        logging.info("Making pie-charts for 'sample_Id' per process.")
        for iproc, process in enumerate(dict_samples['sample']):
            if process == 'data': continue
            process_cleaned = process.replace('/', '_')
            PieCharter(data_SR[data_SR['sample_Id'] == iproc], 'tau_1_tmp', dict_tmps_tau, 'PieCharts/SR_'+process_cleaned+'_tau1tmp_'+config['AnalysisChannel']+'.pdf')
            PieCharter(data_SR[data_SR['sample_Id'] == iproc], 'lep_1_tmp', dict_tmps_lep, 'PieCharts/SR_'+process_cleaned+'_lep1tmp_'+config['AnalysisChannel']+'.pdf')
            if config['AnalysisChannel']=='hadhad':
                PieCharter(data_SR[data_SR['sample_Id'] == iproc], 'tau_2_tmp', dict_tmps_tau, 'PieCharts/SR_'+process_cleaned+'_tau2tmp_'+config['AnalysisChannel']+'.pdf')
                PieCharter(data_SR[data_SR['sample_Id'] == iproc], 'TempCombinations', dict_tmps_comb, 'PieCharts/SR_comb_'+process_cleaned+'_'+config['AnalysisChannel']+".pdf")
            if config['AnalysisChannel']=='lephad':
                PieCharter(data_SR[data_SR['sample_Id'] == iproc], 'lep_2_tmp', dict_tmps_lep, 'PieCharts/SR_'+process_cleaned+'_lep2tmp_'+config['AnalysisChannel']+'.pdf')


    #####################################################################
    ###                        Print q/g ratios                       ###
    #####################################################################
    # from DataModifiers import qgratio, qgratio_plot
    #
    # ensure_dir('QGplots/')
    # if config['AnalysisChannel']=='lephad':
    #     logging.info('Leading tau (CR)')
    #     qgratio(data_dict['data_SR'], 1)
    #     qgratio_plot(pd.concat([data_dict['data_SR'], data_dict['data_Tau1CR']], keys=['SR','CR']), 1, plt, sample='all')
    # elif config['AnalysisChannel']=='hadhad':
    #     logging.info('Leading tau (CR1)')
    #     data_dict['data_SR']=qgratio(data_dict['data_SR'], 1, None)
    #     logging.info('Sun-leading tau (CR2)')
    #     data_dict['data_SR']=qgratio(data_dict['data_SR'], 2, None)

    #     qgratio_plot(pd.concat([data_dict['data_SR'], data_dict['data_Tau1CR']], keys=['SR','CR']), 1, plt, sample='all')
    #     qgratio_plot(pd.concat([data_dict['data_SR'], data_dict['data_Tau2CR']], keys=['SR','CR']), 2, plt, sample='all')

    #####################################################################
    ###                       MVA Methods (tHq)                       ###
    #####################################################################
    if config['AnalysisChannel']=='lephad' and config['TRAIN_THQ']:
        logging.info("MVA start: --- {t} seconds ---".format(t = round(time.time()-start_time, 3)))

        if config['MVA_METHOD']=='LGBM': logging.info(style.YELLOW+'Train LGBM to classify tH, tZ and others'+style.RESET)
        elif config['MVA_METHOD']=='NN': logging.info(style.YELLOW+'Train NN to classify tH, tZ and others'+style.RESET)

        list_of_branches_mva = ['eta_jf','phi_jf','pt_jf', 'M_b_jf',
                            'eta_b','phi_b','pt_b','MMC_out_1',
                            'HvisEta','HvisPt','m_sumet','m_met',
                            'TvisEta','TvisMass','TvisPt',
                            #'HT_all',
                            'lep_Top_eta','lep_Top_pt','lep_Top_phi','deltaRTau',
                            'pt_jet1', 'phi_jet1','eta_jet1','m_phi_met',
                            'deltaPhiTau', 'had_tau_pt','had_tau_eta',
                            'lep_Higgs_pt','lep_Higgs_eta','lep_Higgs_phi',
                            'm_njets','m_nbjets',
                            'fs_had_tau_1_nTrack',
                            'fs_had_tau_1_RNNScore','fs_had_tau_1_JetTrackWidth','fs_had_tau_1_JetCaloWidth',
                            'weight_nominal',
                            'process']
        mask_mc = data_dict['data_SR'].process!='data'
        df_MVA = data_dict['data_SR'][mask_mc].copy()
        df_MVA = df_MVA[list_of_branches_mva]

        from MVAMethods import ClassifyProcess
        classifier = ClassifyProcess(df_MVA, 'process', 'weight_nominal')
        classifier.label_classes()
        classifier.balance_classes()
        classifier.split_data()
        classifier.scale_features()
        #classifier.plot_correlation_with_process()

        if config['TUNE_PARS']=='Grid':
            ## Find parameters
            params_grid = {
                'boosting_type': ['dart'], # 'gbdt'
                'num_leaves': [12, 20, 32],
                'max_depth': [-1],
                'learning_rate': [0.01, 0.05],
                'n_estimators': [150, 300],
                'feature_fraction': [1.0, 0.8, 0.5],
                'subsample': [0, 0.2],
                'reg_alpha': [0],
                'reg_lambda': [0]
            }
            classifier.tune_hyperparameters(params_grid)

        # Bayesian tune
        elif config['TUNE_PARS']=='Bayesian':
            if config['MVA_METHOD']=='LGBM':
                best_score, best_params = classifier.bayes_parameter_tune()
            elif config['MVA_METHOD']=='NN':
                best_score, best_params = classifier.bayes_parameter_tune_nn()
            print('opt_score',best_score)
            print('opt_params',best_params)

        if config['DilepSelection']=='SS':
            params = {'boosting_type': 'gbdt',
                  'objective': 'multiclass',
                  'num_class': 3,
                  'is_unbalance': 'false',
                  'metric': 'multi_logloss', # auc_mu or multi_logloss
                  'num_leaves': 24,
                  'max_depth': -1,
                  'learning_rate': 0.02,
                  'min_data_in_leaf': 512,
                  'feature_fraction': 0.6,
                  'subsample': 0.5,
                  'subsample_freq': 20,
                  'max_bin': 30,
                  'extra_trees': True,
            }
        elif config['DilepSelection']=='OS':
            params = {'boosting_type': 'gbdt',
                  'objective': 'multiclass',
                  'num_class': 3,
                  'is_unbalance': 'false',
                  'metric': 'multi_logloss', # auc_mu or multi_logloss
                  'num_leaves': 32,
                  'max_depth': -1,
                  'learning_rate': 0.025,
                  'min_data_in_leaf': 512,
                  'feature_fraction': 0.6,
                  'subsample': 0.8,
                  'subsample_freq': 20,
                  'max_bin': 30,
                  'extra_trees': True,
            }


        # Model building
        if config['MVA_METHOD']=='LGBM':
            classifier.train_model(params, do_early_stopping=True, num_boost_round=500)
        elif config['MVA_METHOD']=='NN':
            params = {
                'dropout_rate_1': 0.06,
                'dropout_rate_2': 0.19,
                'dropout_rate_3': 0.12,
                'dense_size_1': 75,
                'dense_size_2': 58,
                'dense_size_3': 62,
                'learning_rate': 0.038,
                'batch_size': 1254,
            }
            classifier.train_model_nn(params)

        if config['EVAL_ARGMAX']: classifier.evaluate_model()

        # Plotting train/test ROC curves
        classifier.plot_roc_curves_OvR()
        classifier.plot_roc_curves_OvR(subsample='train')

        if config['EVAL_LGBM_IMPORTANCE']: classifier.get_importance()

        if config['EVAL_SHAP'] and not len(classifier.cat_features):
            classifier.explain_model()
        if config['EVAL_TERNARY']: classifier.plot_ternary()

        # Add predictions
        logging.info("Start predict: --- {t} seconds ---".format(t = round(time.time()-start_time, 3)))
        predictions = classifier.predict(data_dict['data_SR'][list_of_branches_mva[:-2]].copy())
        logging.info("End predict: --- {t} seconds ---".format(t = round(time.time()-start_time, 3)))
        data_dict['data_SR'] = pd.concat([data_dict['data_SR'], predictions], axis=1)
        data_dict['data_SR'].name='_SR'
        logging.info("Concat: --- {t} seconds ---".format(t = round(time.time()-start_time, 3)))

        from MVAMethods import plot_weight, plot_probs, get_LH_discriminant, get_discriminant, plot_scores
        plot_probs(data_dict['data_SR'][mask_mc])
        plot_weight(data_dict['data_SR'])

        # Construct individual likelihood discriminants
        data_dict['data_SR'] = get_LH_discriminant(data_dict['data_SR'], sig_name='tH_prob', bkg_names=['tZ_prob', 'others_prob'], k_val=0.05, out_name='tH_LHscore')
        data_dict['data_SR'] = get_LH_discriminant(data_dict['data_SR'], sig_name='tZ_prob', bkg_names=['others_prob','tH_prob'], k_val=0.05, out_name='tZ_LHscore')
        data_dict['data_SR'] = get_LH_discriminant(data_dict['data_SR'], sig_name='others_prob', bkg_names=['tZ_prob','tH_prob'], k_val=0.05, out_name='others_LHscore')
        plot_scores(data_dict['data_SR'][mask_mc], varname='tH_LHscore')
        plot_scores(data_dict['data_SR'][mask_mc], varname='tZ_LHscore')
        plot_scores(data_dict['data_SR'][mask_mc], varname='others_LHscore')

        data_dict['data_SR'] = get_discriminant(data_dict['data_SR'], sig_name='tH_prob', bkg_names=['tZ_prob','others_prob'], out_name='tH_score')
        data_dict['data_SR'] = get_discriminant(data_dict['data_SR'], sig_name='tZ_prob', bkg_names=['tH_prob','others_prob'], out_name='tZ_score')
        data_dict['data_SR'] = get_discriminant(data_dict['data_SR'], sig_name='others_prob', bkg_names=['tZ_prob','tH_prob'], out_name='others_score')
        plot_scores(data_dict['data_SR'][mask_mc], varname='tH_score')
        plot_scores(data_dict['data_SR'][mask_mc], varname='tZ_score')
        plot_scores(data_dict['data_SR'][mask_mc], varname='others_score')

        # 2-dim scatter plot of scores
        from MVAMethods import plot_scatter
        plot_scatter(data_dict['data_SR'][mask_mc].copy(), xname='tH_LHscore', yname='others_LHscore', classname='process')

        # Apply a BDT score cut?
        if config['DilepSelection']=='SS':
            mask_BDTscore = data_dict['data_SR']['others_score'] < 0
            data_dict['data_SR_tight'] = data_dict['data_SR'][mask_BDTscore]
            data_dict['data_SR_tight'].name='_SR_tight'
        elif config['DilepSelection']=='OS':
            mask_BDTscore = data_dict['data_SR']['others_LHscore'] < -1
            data_dict['data_SR_tight'] = data_dict['data_SR'][mask_BDTscore]
            data_dict['data_SR_tight'].name='_SR_tight'

    #####################################################################
    ###                  MVA Methods (QG-Tagger)                      ###
    #####################################################################
    if config['AnalysisChannel']=='lephad' and config['TRAIN_QGTAGGER']:
        logging.info("MVA QGTagger start: --- {t} seconds ---".format(t = round(time.time()-start_time, 3)))

        logging.info(style.YELLOW+'Train LGBM to classify quark/gluon initiated tau fakes'+style.RESET)

        list_of_branches_mva = ['m_nbjets', 'm_njets',
                            'fs_had_tau_1_nTrack','fs_had_tau_1_pt','fs_had_tau_1_eta',
                            'fs_had_tau_1_RNNScore',
                            'fs_had_tau_1_JetTrackWidth','fs_had_tau_1_JetCaloWidth',
                            'm_sumet','m_met',
                            'weight_nominal',
                            'tau_1_tmp']
        mask_mc = data_dict['data_Tau1CR'].process!='data'
        df_MVA = data_dict['data_Tau1CR'][mask_mc].copy()
        df_MVA = df_MVA[list_of_branches_mva]

        from MVAMethods import ClassifyProcess
        QGTagger = ClassifyProcess(df_MVA, 'tau_1_tmp', 'weight_nominal', classes=['q-jet', 'g-jet'])
        QGTagger.label_classes()
        QGTagger.balance_classes(reweight_dims=['m_nbjets','fs_had_tau_1_nTrack','fs_had_tau_1_pt','fs_had_tau_1_eta'])
        QGTagger.split_data()
        QGTagger.scale_features()

        params = {
            'application': 'binary',
            'objective': 'cross_entropy', # binary cross_entropy
            'metric': 'auc', # auc or binary_logloss
            'is_unbalance': 'false',
            'boosting': 'gbdt', # methods: gbdt, rf, dart, goss
            'num_leaves': 32, # default: 32
            'max_depth': -1,
            'learning_rate': 0.035,
            'min_data_in_leaf': 512,
            'feature_fraction': 0.6,
            'subsample': 0.8,
            'subsample_freq': 20,
            'max_bin': 10,
            'extra_trees': True,
            'verbose': 0,
            'seed': 42
        }

        QGTagger.train_model(params, do_early_stopping=True, num_boost_round=500, metric_name='auc')
        QGTagger.plot_roc_curves()
        QGTagger.get_importance()

        # Add predictions
        data_dict['data_SR']['lgbm_score']     = QGTagger.predicit_binary(data_dict['data_SR'][list_of_branches_mva[:-2]].copy())
        data_dict['data_Tau1CR']['lgbm_score'] = QGTagger.predicit_binary(data_dict['data_Tau1CR'][list_of_branches_mva[:-2]].copy())

        # Modify predictions for 'unknown' determination
        mask_unknown = data_dict['data_Tau1CR']['fs_had_tau_1_JetTrackWidth'] < 0
        data_dict['data_Tau1CR']['lgbm_score'][mask_unknown] = -1.0

        mask_unknown = data_dict['data_SR']['fs_had_tau_1_JetTrackWidth'] < 0
        data_dict['data_SR']['lgbm_score'][mask_unknown] = -1.0

        # data_dict['data_SR'].name='_SR'
        logging.info("Concat: --- {t} seconds ---".format(t = round(time.time()-start_time, 3)))


    #####################################################################
    ###                     Plotting 2 Histograms                     ###
    #####################################################################
    if -1>0:
        from Plotter import plot_histogram
        dict_hists = Configurate_VarHistBins()
        dict_xtitles = Configurate_Xtitles()

        histnames = ['m_njets',
                     'm_nbjets',
                     'fs_had_tau_true_pdg',  'fs_had_tau_true_partonTruthLabelID',
                     #'fs_had_tau_2_true_pdg','fs_had_tau_2_true_partonTruthLabelID',
                     'fs_had_tau_1_pt','fs_had_tau_1_eta','fs_had_tau_1_nTrack','fs_had_tau_1_RNNScore',
                     #'fs_had_tau_2_pt','fs_had_tau_2_eta','fs_had_tau_2_nTrack','fs_had_tau_2_RNNScore',
                     'eta_jf','phi_jf','pt_jf', 'M_b_jf',
                    #'eta_b','phi_b','pt_b','MMC_out_1',
                    'HvisEta','HvisPt','m_sumet','m_met',
                    'TvisEta','TvisMass','TvisPt'
        ]
        mask = data_dict['data_SR'].process == 'Wjets'
        mask_new = data_dict['data_SR'].process == 'Wjetsnew'
        for var in histnames:
            bins,xtitle = dict_hists[var],dict_xtitles[var]
            plot_histogram(data_dict['data_SR'], var, bins, mask, mask_new, weight_name='weight_nominal', xtitle=xtitle)

        from Plotter import plot_weight
        plot_weight(data_dict['data_SR'], mask, mask_new)

        print('======================')
        print('Old W+jets raw events: ',len(data_dict['data_SR'][mask]))
        print('New W+jets raw events: ',len(data_dict['data_SR'][mask_new]))


    #####################################################################
    ###                     Plotting Processes                        ###
    #####################################################################
    dict_hists = Configurate_VarHistBins()
    dict_xtitles = Configurate_Xtitles()

    if config['PLOTPROCESSES']:
        logging.info('Making PLOTPROCESSES plots for {}'.format(config['PLOTPROCESSES_VARS']))
        ensure_dir('Plots/')

        for var in config['PLOTPROCESSES_VARS']:
            bins,xtitle = dict_hists[var],dict_xtitles[var]

            for region_name in config['PLOTPROCESSES_REGIONS']:
                hists = HistMaker(data_dict[region_name], 'process', dict_samples, var, bins, xtitle=xtitle)
                # Compute syst variations if available (CORRECT_BKG_TAU=True)
                if config['ADD_UNCERTAINTY']:
                    UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(data_dict[region_name], 'process', dict_samples, var, bins, SystVarWeightTags[:-1])
                    HistMaker(data_dict[region_name], 'process', dict_samples, var, bins, xtitle=xtitle, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)

                # Purity/Significance
                if var=='NNout_tauFakes' and region_name=='data_SR':
                    HistMaker(data_dict[region_name], 'process', dict_samples, var, bins, xtitle=xtitle, PlotPurity='tH')

    #####################################################################
    ###                   Plot Tau Fakes Templates                    ###
    #####################################################################
    if config['PLOTTAU1TMP']:
        logging.info('Making PLOTTAU1TMP plots for {}'.format(config['PLOTTAU1TMP_VARS']))
        ensure_dir('Plots/')

        for var in config['PLOTTAU1TMP_VARS']:
            bins,xtitle = dict_hists[var],dict_xtitles[var]

            for region_name in config['PLOTTAU1TMP_REGIONS']:
                hists = HistMaker(data_dict[region_name], 'tau_1_tmp', dict_tmps_tau, var, bins, xtitle=xtitle)
                # Compute syst variations if available (CORRECT_BKG_TAU=True)
                if config['ADD_UNCERTAINTY']:
                    UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(data_dict[region_name], 'tau_1_tmp', dict_tmps_tau, var, bins, SystVarWeightTags[:-1])
                    HistMaker(data_dict[region_name], 'tau_1_tmp', dict_tmps_tau, var, bins, xtitle=xtitle, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)

                # Purity/Significance
                if var=='NNout_tauFakes' and region_name=='data_SR':
                    HistMaker(data_dict[region_name], 'tau_1_tmp', dict_tmps_tau, var, bins, xtitle=xtitle, PlotPurity='tau')

    if config['PLOTTAU2TMP']:
        logging.info('Making PLOTTAU2TMP plots for {}'.format(config['PLOTTAU2TMP_VARS']))
        ensure_dir('Plots/')

        for var in config['PLOTTAU2TMP_VARS']:
            bins,xtitle = dict_hists[var],dict_xtitles[var]

            for region_name in config['PLOTTAU2TMP_REGIONS']:
                hists = HistMaker(data_dict[region_name], 'tau_2_tmp', dict_tmps_tau, var, bins, xtitle=xtitle)
                # Compute syst variations if available (CORRECT_BKG_TAU=True)
                if config['ADD_UNCERTAINTY']:
                    UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(data_dict[region_name], 'tau_2_tmp', dict_tmps_tau, var, bins, SystVarWeightTags[:-1])
                    HistMaker(data_dict[region_name], 'tau_2_tmp', dict_tmps_tau, var, bins, xtitle=xtitle, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)

    #####################################################################
    ###                         Lep Templates                         ###
    #####################################################################
    if config['PLOTLEP1TMP']:
        logging.info('Making PLOTLEP1TMP plots for {}'.format(config['PLOTLEP1TMP_VARS']))
        ensure_dir('Plots/')

        for var in config['PLOTLEP1TMP_VARS']:
            bins,xtitle = dict_hists[var],dict_xtitles[var]

            for region_name in config['PLOTLEP1TMP_REGIONS']:
                hists = HistMaker(data_dict[region_name], 'lep_1_tmp', dict_tmps_lep, var, bins, xtitle=xtitle)
                if config['ADD_UNCERTAINTY']:
                    UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(data_dict[region_name], 'lep_1_tmp', dict_tmps_lep, var, bins, SystVarWeightTags[:-1])
                    HistMaker(data_dict[region_name], 'lep_1_tmp', dict_tmps_lep, var, bins, xtitle=xtitle, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)


    if config['PLOTLEP2TMP']:
        logging.info('Making PLOTLEP2TMP plots for {}'.format(config['PLOTLEP2TMP_VARS']))
        ensure_dir('Plots/')

        for var in config['PLOTLEP2TMP_VARS']:
            bins,xtitle = dict_hists[var],dict_xtitles[var]

            for region_name in config['PLOTLEP2TMP_REGIONS']:
                hists = HistMaker(data_dict[region_name], 'lep_2_tmp', dict_tmps_lep, var, bins, xtitle=xtitle)
                if config['ADD_UNCERTAINTY']:
                    UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(data_dict[region_name], 'lep_2_tmp', dict_tmps_lep, var, bins, SystVarWeightTags[:-1])
                    HistMaker(data_dict[region_name], 'lep_2_tmp', dict_tmps_lep, var, bins, xtitle=xtitle, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)

    #####################################################################
    ###                    di-lep/di-tau Templates                    ###
    #####################################################################
    if config['PLOTPAIRSTMP']:
        mypairs = 'di-lep' if config['AnalysisChannel']=='lephad' else 'di-tau'
        logging.info('Making {} combined regions plots'.format(mypairs))
        ensure_dir('Plots/')

        if config['AnalysisChannel']=='lephad':
            mypairs = 'di-lep'
            thisdata = data_dict['data_nb_TauSR']
            #rnames = ['SR=TT']+region_names
            rnames = ['SR=#it{lep}^{pass}#it{lep}^{pass}']+region_names
        elif config['AnalysisChannel']=='hadhad':
            mypairs = 'di-tau'
            thisdata = data_dict['data_nb_LepSR']
            #rnames = ['SR=MM']+region_names
            rnames = ['SR=#tau_{had}^{pass}#tau_{had}^{pass}']+region_names

        histnames = ['regions_encoded','fs_had_tau_1_RNNScore','fs_had_tau_2_RNNScore']  # 'fs_had_tau_1_RNNScore', 'fs_had_tau_2_RNNScore', 'regions_encoded'
        for var in histnames:
            if var=='regions_encoded':
                bins = [4,0,4]
                xtitle = mypairs+' regions'
            else:
                bins,xtitle = dict_hists[var],dict_xtitles[var]
                rnames = None

            hists = HistMaker(thisdata, 'TempCombinations', dict_tmps_comb, var, bins, xtitle=xtitle, region_names=rnames)
            if config['ADD_UNCERTAINTY']:
                UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(thisdata, 'TempCombinations', dict_tmps_comb, var, bins, SystVarWeightTags[:-1])
                hists = HistMaker(thisdata, 'TempCombinations', dict_tmps_comb, var, bins, xtitle=xtitle, region_names=rnames, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio )

        # hists = HistMaker(thisdata, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [4,0,4], xtitle=mypairs+' regions', region_names=rnames)
        # if config['ADD_UNCERTAINTY']:
        #     UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(thisdata, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [4,0,4], SystVarWeightTags[:-1])
        #     hists = HistMaker(thisdata, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [4,0,4], xtitle=mypairs+' regions', region_names=rnames, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio )


        # Extended CRs + SR
        logging.info("Making combined regions plot for 3 decay products templated by '{}' ".format(config['PLOTPAIRSTMP_TMPNAME']))
        if config['AnalysisChannel']=='lephad':
            masks_CI = [ m_bjets & m_tau1_tight  & (m_lep1_tight  & m_lep2_tight)  & dicharge_selection,   # SR
                         m_bjets & ~m_tau1_tight & (m_lep1_tight  & m_lep2_tight)  & dicharge_selection,   # Tau CR
                         m_bjets & m_tau1_tight  & (m_lep1_tight  & ~m_lep2_tight) & dicharge_selection,
                         m_bjets & m_tau1_tight  & (~m_lep1_tight & m_lep2_tight)  & dicharge_selection,
                         m_bjets & m_tau1_tight  & (~m_lep1_tight & ~m_lep2_tight) & dicharge_selection]
            region_names_CI = ['SR=TTM', 'TT#bar{M}', 'T#bar{T}M', '#bar{T}TM', '#bar{T}#bar{T}M']
        elif config['AnalysisChannel']=='hadhad':
            masks_CI = [ m_bjets & m_lep1_tight & (m_tau1_tight & m_tau2_tight), # SR
                         m_bjets & ~m_lep1_tight & (m_tau1_tight & m_tau2_tight), # Lep CR
                         m_bjets & m_lep1_tight & (m_tau1_tight & ~m_tau2_tight), 
                         m_bjets & m_lep1_tight & (~m_tau1_tight & m_tau2_tight),
                         m_bjets & m_lep1_tight & (~m_tau1_tight & ~m_tau2_tight)]
            region_names_CI = ['SR=TMM', '#bar{T}MM', 'TM#bar{M}', 'T#bar{M}M', 'T#bar{M}#bar{M}']

        data_df_CI = RegionsAnnotator(data_df.copy(), masks_CI, region_names_CI)
        data_df_CI.name='_INCL'

        nbins = len(masks_CI)+1
        tmp_options = {'process': dict_samples,
                       'tau_1_tmp': dict_tmps_tau,
                       'TempCombinations': dict_tmps_comb,
        }

        hists = HistMaker(data_df_CI,
                      config['PLOTPAIRSTMP_TMPNAME'], tmp_options[config['PLOTPAIRSTMP_TMPNAME']],     # Templated by processes
                      'regions_encoded', [5,1,6],
                      xtitle='Fake Control Regions ({})'.format(mypairs),
                      region_names=region_names_CI )
        if config['ADD_UNCERTAINTY']:
            UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(data_df_CI, config['PLOTPAIRSTMP_TMPNAME'], tmp_options[config['PLOTPAIRSTMP_TMPNAME']], 'regions_encoded', [5,1,6], SystVarWeightTags[:-1])
            HistMaker(data_df_CI,
                      config['PLOTPAIRSTMP_TMPNAME'], tmp_options[config['PLOTPAIRSTMP_TMPNAME']],
                      'regions_encoded', [5,1,6],
                      region_names=region_names_CI,
                      xtitle='Fake Control Regions ({})'.format(mypairs),
                      UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)

    #####################################################################
    ###                      TAU BKG Extraction                       ###
    #####################################################################
    if config['EXTRACT_TAUBKG']:

        if config['AnalysisChannel']=='lephad':
            from Plotter import plot_histogram

            #####################################################################
            if 'Counting' in config['EXTRACT_TAUBKG_METHOD']:
                from BkgFitter import TauFakeYieldCorrector
                logging.info(style.YELLOW+"Computing Scale-Factors for jet-faking-tau with Counting method (m_njets histogram)."+style.RESET)

                results = {
                    'name': 'Counting',
                }

                thistemplator = 'tau_1_tmp_qgcomb' if config['SPLIT_JETSTMP'] else 'tau_1_tmp'
                thisdict_tmps_tau = dict_tmps_qgcomb if config['SPLIT_JETSTMP'] else dict_tmps_tau

                #### inclusive estimate
                #bins,xtitle = dict_hists['m_njets'],dict_xtitles['m_njets']
                #hists_yields = HistMaker(data_dict['data_Tau1CR'], thistemplator, thisdict_tmps_tau, 'm_njets', bins, xtitle=xtitle)
                #TauFakeYieldCorrector(hists_yields, unknown_is_jet=False)

                # write results to latex file
                latexfile = open('taufakes_yields_method.tex', 'w', encoding='utf-8')
                from BkgFitter import print_to_latex_start, print_to_latex_values
                print_to_latex_start(latexfile, 'Counting')

                #### nbjets, nTracks and pT bins
                bins,xtitle = dict_hists['m_njets'],dict_xtitles['m_njets']
                counter = 0
                # Loop over b-jets ( config['BJETS'] or [1,2] )
                for nb in config['BJETS']:
                    # If SS channel, consider 1b+2b together
                    if config['DilepSelection'] == 'SS':
                        if nb>1: continue

                    # Loop over tau nTracks
                    for prong in [1, 3]:
                        # to latex table
                        latexfile.write("    \\midrule\n")
                        latexfile.write("    \\multicolumn{7}{c}{"+str(prong)+"-prong, "+str(nb)+" b-jet}, \\\\\n")
                        latexfile.write("    \\midrule\n")

                        catname = 'SFs_'+str(nb)+'b_'+str(prong)+'prong'
                        results[catname] = {
                            'nb': nb,
                            'nTracks': prong,
                            'pT_low': [],
                            'pT_high':[],
                            'SFs': [],
                            'SFs_err': [],
                        }

                        # Loop over tau pT bins
                        for ptbin in [1,2,3]:
                            tau_pT_dict = {
                                '1': '20--30',
                                '2': '30--40',
                                '3': '$>$40',
                            }
                            if not config['DilepSelection'] == 'SS':
                                mymsg = style.CYAN+str(nb)+' bjets, '+str(prong)+'-prong, tau pT = '+tau_pT_dict[str(ptbin)].replace('$','')+" GeV"+style.RESET
                            else:
                                mymsg = style.CYAN+'1+2 bjets, '+str(prong)+'-prong, tau pT = '+tau_pT_dict[str(ptbin)].replace('$','')+" GeV"+style.RESET

                            nb_mask = m_1bjet if nb==1 else m_2bjet
                            prong_mask = m_tau1_1p if prong==1 else m_tau1_3p
                            pt_mask = m_tau1_pt1 if ptbin==1 else m_tau1_pt2 if ptbin==2 else m_tau1_pt3

                            # For SS channel only:
                            if config['DilepSelection'] == 'SS':
                                nb_mask = m_1bjet | m_2bjet
                            
                            subdf = data_dict['data_Tau1CR'][nb_mask & prong_mask & pt_mask]
                            subdf.name='_Tau1CR_'+str(prong)+'p_pt'+str(ptbin)
                            if len(subdf)==0: continue
                            hists_yields = HistMaker(subdf, thistemplator, thisdict_tmps_tau, 'm_njets', bins, xtitle=xtitle)
                            result = TauFakeYieldCorrector(hists_yields, unknown_is_jet=False, mymsg=mymsg)

                            result['tau_pT_range'] = tau_pT_dict[str(ptbin)]

                            print_to_latex_values(latexfile, result, 'Counting')

                            ##########################################
                            # Computing fakes in SR
                            subdf_SR = data_dict['data_SR'][nb_mask & prong_mask & pt_mask]
                            subdf_SR.name='_SR_'+str(prong)+'p_pt'+str(ptbin)
                            hists_SR_yields = HistMaker(subdf_SR, thistemplator, thisdict_tmps_tau, 'm_njets', bins, xtitle=xtitle)
                            jet_SR = None
                            for ihist, hist in enumerate(hists_SR_yields):
                                if '_jet_' in hist.GetName():
                                    jet_SR = ROOT.TH1D( hist.Clone("jet") )
                            # Compute integral
                            Njet_err = ctypes.c_double(0.)
                            Njet = jet_SR.IntegralAndError(0, jet_SR.GetNbinsX()+1, Njet_err, "")
                            Njet_SR = Njet*result['SF_jets_est']
                            Njet_err_SR = Njet_SR*math.sqrt( (Njet_err.value/Njet)**2 + (result['SF_jets_err_est']/result['SF_jets_est'])**2 )
                            logging.info('      Estimated jets in SR Njets = {x} \u00B1 {dx}'.format(x=round(Njet_SR,1), dx=round(Njet_err_SR, 1)))
                            ##########################################

                            # append results
                            pT_low = 20 if ptbin==1 else 30 if ptbin==2 else 40
                            pT_high = 30 if ptbin==1 else 40 if ptbin==2 else 100
                            results[catname]['pT_low'].append(pT_low)
                            results[catname]['pT_high'].append(pT_high)

                            results[catname]['SFs'].append( result['SF_jets_est'] )
                            results[catname]['SFs_err'].append( result['SF_jets_err_est'] )

                        counter += 1

                # Finish the latex table
                latexfile.write('    \\bottomrule\n')
                latexfile.write('  \\end{tabular}\n')
                latexfile.write('\\end{table}')

                if config['EXTRACT_TAUBKG_SAVE']:
                    from BkgFitter import result_to_root
                    outfilename = 'taufakes_yields_method.root'
                    logging.info("Saving results to {}".format(outfilename))
                    result_to_root(results, outfilename)

            #####################################################################
            if 'TemplateFit' in config['EXTRACT_TAUBKG_METHOD']:
                from BkgFitter import Fit_Tau
                logging.info(style.YELLOW+'Computing Scale-Factors for jet-faking-tau with Template Fit method'+style.RESET)

                results = {
                    'name': 'TemplateFit',
                }

                # write results to latex file
                latexfile = open('taufakes_tmpfit_method.tex', 'w', encoding='utf-8')
                from BkgFitter import print_to_latex_start, print_to_latex_values
                print_to_latex_start(latexfile, 'TemplateFit')

                # Find optimal binning?
                #tmphistname = 'fs_had_tau_1_JetTrackWidth'
                tmphistname = 'lgbm_score' if config['TRAIN_QGTAGGER'] else 'fs_had_tau_1_JetTrackWidth'
                optimize_binning = True

                #tmphistname = 'fs_had_tau_1_RNNScore'

                try:
                    if dict_hists[tmphistname+'_1b_p1pt1_opt']:  ## change here to '_1b_p1pt1_opt'
                        logging.info("Optimized binning for '{}' is in the config file. Skipping the optimization".format(tmphistname+'_1b_p1pt1_opt'))
                        optimize_binning = False
                except KeyError:
                    logging.info("Will optimize the binning of the '{}' histogram.".format(tmphistname))

                # Loop over b-jets ( config['BJETS'] or [1,2] )
                for nb in config['BJETS']:
                    # Loop over tau nTracks
                    for prong in [1, 3]:
                        # to latex table
                        latexfile.write("    \\midrule\n")
                        latexfile.write("    \\multicolumn{5}{c}{"+str(prong)+"-prong, "+str(nb)+" b-jet} \\\\\n")
                        latexfile.write("    \\midrule\n")

                        catname = 'SFs_'+str(nb)+'b_'+str(prong)+'prong'
                        results[catname] = {
                            'nb': nb,
                            'nTracks': prong,
                            'pT_low': [],
                            'pT_high':[],
                            'SFs_q': [],
                            'SFs_q_err': [],
                            'SFs_g': [],
                            'SFs_g_err': [],
                            'SFs_unknown': [],
                            'SFs_unknown_err': [],
                        }
  

                        # Loop over tau pT bins
                        for ptbin in [1,2,3]:
                            tau_pT_dict = {
                                '1': '20--30',
                                '2': '30--40',
                                '3': '$>$40',
                            }
                            mymsg = style.CYAN+str(nb)+' bjets, '+str(prong)+'-prong, tau pT = '+tau_pT_dict[str(ptbin)].replace('$','')+" GeV"+style.RESET
                            
                            # Optimiza the binning?
                            if optimize_binning:
                                bins_varBinSize = dict_hists[tmphistname+'_p'+str(prong)+'pt'+str(ptbin)]
                            else:
                                bins_varBinSize = dict_hists[tmphistname+'_'+str(nb)+'b_p'+str(prong)+'pt'+str(ptbin)+'_opt']

                            nb_mask = m_1bjet if nb==1 else m_2bjet
                            prong_mask = m_tau1_1p if prong==1 else m_tau1_3p
                            pt_mask = m_tau1_pt1 if ptbin==1 else m_tau1_pt2 if ptbin==2 else m_tau1_pt3

                            #subdf = data_dict['data_Tau1CR'][nb_mask & prong_mask & pt_mask]
                            subdf = pd.concat([data_dict['data_Tau1CR'][nb_mask & prong_mask & pt_mask], data_dict['data_SR'][nb_mask & prong_mask & pt_mask]])
                            subdf.name='_Tau1CR_'+str(nb)+'b_'+str(prong)+'p_pt'+str(ptbin)
                            if len(subdf)==0: continue

                            if optimize_binning:
                                bins_varBinSize_min = bins_varBinSize.copy()

                                # Find optimal binning (partially manual tune)
                                sigma_min = 100
                                result_best = None
                                for i in range(200):
                                    n = random.randint(1,10)
                                    if prong==3 and ptbin==2:
                                        n = random.randint(1,7)
                                    bins = sorted(random.sample(bins_varBinSize, len(bins_varBinSize)-n))
                                    if 0.0 not in bins: bins=[0.0]+bins
                                    hists_tmp = HistMaker(subdf, 'tau_1_tmp', dict_tmps_tau, tmphistname, bins)
                                    thisresult = Fit_Tau(hists_tmp)
                                    sigma = math.sqrt(thisresult['SF_q_err_est']*thisresult['SF_q_err_est'] + thisresult['SF_g_err_est']*thisresult['SF_g_err_est'])
                                    if sigma_min>sigma and thisresult['SF_g_est']>0.6 and thisresult['SF_g_est']<1.3:
                                        sigma_min = sigma
                                        result_best = thisresult
                                        bins_varBinSize_min = bins.copy()

                                #logging.info('The optimal binning: ', bins_varBinSize_min)

                                # Draw histograms with optimal binning
                                hists_tmp = HistMaker(subdf, 'tau_1_tmp', dict_tmps_tau, tmphistname, bins_varBinSize_min)
                                result = Fit_Tau(hists_tmp, DEBUG=False, mymsg=mymsg)

                            # No binning optimization
                            else:
                                logging.info('No binning optimization is requested')
                                xtitle = dict_xtitles[tmphistname+'_'+str(nb)+'b_p'+str(prong)+'pt'+str(ptbin)]
                                hists_tmp = HistMaker(subdf, 'tau_1_tmp', dict_tmps_tau, tmphistname, bins_varBinSize, xtitle=xtitle)
                                result = Fit_Tau(hists_tmp, DEBUG=False, mymsg=mymsg)
                                if config['ADD_UNCERTAINTY']:
                                    UncertaintyBand,UncertaintyBandRatio = UncertaintyMaker(subdf, 'tau_1_tmp', dict_tmps_tau, tmphistname, bins_varBinSize, SystVarWeightTags[:-1])
                                    hists = HistMaker(subdf, 'tau_1_tmp', dict_tmps_tau, tmphistname, bins_varBinSize, xtitle=xtitle, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio )


                                ## Comparison of quark and gluon templates for ttbar and Zjets
                                mask_ttbar_qjets = (subdf.process == 'ttbar') & (subdf.tau_1_tmp == 'q-jet')
                                mask_ttbar_gjets = (subdf.process == 'ttbar') & (subdf.tau_1_tmp == 'g-jet')
                                mask_ttbar_qjets.name = 't#bar{t} q-jet'
                                mask_ttbar_gjets.name = 't#bar{t} g-jet'
                                plot_histogram(subdf, tmphistname, [30, 0, 0.18], mask_ttbar_qjets, mask_ttbar_gjets, weight_name='weight_nominal', xtitle=xtitle, region_tag=None, figname=tmphistname+'_'+str(nb)+'b_p'+str(prong)+'_pt'+str(ptbin)+'_ttbar.pdf', norm=True)

                                mask_Zjets_qjets = (subdf.process == 'Zjets') & (subdf.tau_1_tmp == 'q-jet')
                                mask_Zjets_gjets = (subdf.process == 'Zjets') & (subdf.tau_1_tmp == 'g-jet')
                                mask_Zjets_qjets.name = 'Zjets q-jet'
                                mask_Zjets_gjets.name = 'Zjets g-jet'
                                plot_histogram(subdf, tmphistname, [30, 0, 0.18], mask_Zjets_qjets, mask_Zjets_gjets, weight_name='weight_nominal', xtitle=xtitle, region_tag=None, figname=tmphistname+'_'+str(nb)+'b_p'+str(prong)+'_pt'+str(ptbin)+'_Zjets.pdf', norm=True)



                            # Check raw events
                            mask_qjets_raw = subdf['tau_1_tmp'] == 'q-jet'
                            mask_gjets_raw = subdf['tau_1_tmp'] == 'g-jet'
                            logging.info('     Raw events: q-jets={qj}, g-jets={gj}'.format(qj=sum(mask_qjets_raw), gj=sum(mask_gjets_raw)))

                            # Complete result dictionary
                            result['tau_pT_range'] = tau_pT_dict[str(ptbin)]

                            print_to_latex_values(latexfile, result, 'TemplateFit')

                            ##########################################
                            # Computing fakes in SR
                            subdf_SR = data_dict['data_SR'][nb_mask & prong_mask & pt_mask]
                            subdf_SR.name='_SR_'+str(nb)+'b_'+str(prong)+'p_pt'+str(ptbin)
                            bins, xtitle = dict_hists['m_njets'], dict_xtitles['m_njets']
                            hists_SR_yields = HistMaker(subdf_SR, 'tau_1_tmp', dict_tmps_tau, 'm_njets', bins, xtitle=xtitle)
                            qjet_SR, gjet_SR, unknown_SR = None, None, None
                            for ihist, hist in enumerate(hists_SR_yields):
                                if '_q-jet_' in hist.GetName():
                                    qjet_SR = ROOT.TH1D( hist.Clone("qjet_SR") )
                                elif '_g-jet_' in hist.GetName():
                                    gjet_SR = ROOT.TH1D( hist.Clone("gjet_SR") )
                                elif '_unknown_' in hist.GetName():
                                    unknown_SR = ROOT.TH1D( hist.Clone("unknown_SR") )
                            # Compute integral
                            Nqjet_err, Ngjet_err, Nunknown_err = ctypes.c_double(0.), ctypes.c_double(0.), ctypes.c_double(0.)
                            Nqjet = qjet_SR.IntegralAndError(0, qjet_SR.GetNbinsX()+1, Nqjet_err, "")
                            Ngjet = gjet_SR.IntegralAndError(0, gjet_SR.GetNbinsX()+1, Ngjet_err, "")
                            Nunknown = unknown_SR.IntegralAndError(0, unknown_SR.GetNbinsX()+1, Nunknown_err, "")
                            Njet_SR = Nqjet*result['SF_q_est'] + Ngjet*result['SF_g_est']
                            Njet_err_SR = 0.0
                            logging.info('      Estimated jets in SR Njets = {x} \u00B1 {dx}'.format(x=round(Njet_SR,1), dx=round(Njet_err_SR, 1)))
                            ##########################################

                            # append to results
                            pT_low = 20 if ptbin==1 else 30 if ptbin==2 else 40
                            pT_high = 30 if ptbin==1 else 40 if ptbin==2 else 100
                            results[catname]['pT_low'].append(pT_low)
                            results[catname]['pT_high'].append(pT_high)

                            results[catname]['SFs_q'].append( result['SF_q_est'] )
                            results[catname]['SFs_q_err'].append( result['SF_q_err_est'] )
                            results[catname]['SFs_g'].append( result['SF_g_est'] )
                            results[catname]['SFs_g_err'].append( result['SF_g_err_est'] )
                            results[catname]['SFs_unknown'].append( result['SF_unknown_est'] )
                            results[catname]['SFs_unknown_err'].append( 0.0001 )

                # Finish the latex table
                latexfile.write('    \\bottomrule\n')
                latexfile.write('  \\end{tabular}\n')
                latexfile.write('\\end{table}')

                if config['EXTRACT_TAUBKG_SAVE']:
                    from BkgFitter import result_to_root
                    outfilename = 'taufakes_tmpfit_method.root'
                    logging.info("Saving results to {}".format(outfilename))
                    result_to_root(results, outfilename)

        elif config['AnalysisChannel']=='hadhad':
            logging.info(style.YELLOW+'Computing Scale-Factors for di-tau fakes with Template Fit method'+style.RESET)
            from BkgFitter import Fit_DiTau

            results = {
                'name': 'di-tau',
                'SFs_1b_simjet': [],
                'SFs_1b_jetsim': [],
                'SFs_1b_jetjet': [],
                'SFs_2b_simjet': [],
                'SFs_2b_jetsim': [],
                'SFs_2b_jetjet': [],

                'SFs_1b_simjet_err': [],
                'SFs_1b_jetsim_err': [],
                'SFs_1b_jetjet_err': [],
                'SFs_2b_simjet_err': [],
                'SFs_2b_jetsim_err': [],
                'SFs_2b_jetjet_err': [],

                'bins': [0, 1, 2, 3, 4],
            }

            # Loop over b-jets ( config['BJETS'] or [1,2] )
            for nb in config['BJETS']:
                for prong1 in [1, 3]:
                    for prong2 in [1,3]:
                        mymsg = style.CYAN+str(nb)+' bjets, '+str(prong1)+'-prong,'+str(prong2)+'-prong'+style.RESET

                        nb_mask = m_1bjet if nb==1 else m_2bjet
                        prong1_mask = m_tau1_1p if prong1==1 else m_tau1_3p
                        prong2_mask = m_tau2_1p if prong2==1 else m_tau2_3p

                        subdf = data_dict['data_nb_LepSR'][nb_mask & prong1_mask & prong2_mask]
                        subdf.name='_Tau1CR_'+str(nb)+'b_'+str(prong1)+'p_'+str(prong2)+'p'

                        hists = HistMaker(subdf, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [3,1,4], region_names=region_names )
                        result = Fit_DiTau(subdf.copy(), hists, dict_tmps_comb, mymsg=mymsg)

                        # store results to dict
                        results['SFs_'+str(nb)+'b_simjet'].append(result['SF_simjet_est'])
                        results['SFs_'+str(nb)+'b_simjet_err'].append(result['SF_simjet_err_est'])
                        results['SFs_'+str(nb)+'b_jetsim'].append(result['SF_jetsim_est'])
                        results['SFs_'+str(nb)+'b_jetsim_err'].append(result['SF_jetsim_err_est'])
                        results['SFs_'+str(nb)+'b_jetjet'].append(result['SF_jetjet_est'])
                        results['SFs_'+str(nb)+'b_jetjet_err'].append(result['SF_jetjet_err_est'])

            if config['EXTRACT_TAUBKG_SAVE']:
                from BkgFitter import result_to_root
                outfilename = 'taufakes_ditau.root'
                logging.info("Saving results to {}".format(outfilename))
                result_to_root(results, outfilename)

            

    #####################################################################
    ###                      TAU BKG Extraction                       ###
    #####################################################################
    if config['EXTRACT_LEPBKG']:
        if config['AnalysisChannel']=='lephad':
            logging.info(style.YELLOW+'Computing Scale-Factors for di-lep fakes with Template Fit method'+style.RESET)

            from BkgFitter import Fit_DiTau

            results = {
                'name': 'di-lep',
                'SFs_1b': [],
                'SFs_2b': [],
                'SFs_1b_err': [],
                'SFs_2b_err': [],
                'labels': ['simjet', 'jetsim', 'jetjet'],
            }

            # write results to latex file
            latexfile = open('taufakes_dilep.tex', 'w', encoding='utf-8')
            from BkgFitter import print_to_latex_start, print_to_latex_values
            print_to_latex_start(latexfile, 'di-lep')

            # Loop over b-jets ( config['BJETS'] or [1,2] )
            for nb in config['BJETS']:
                mymsg = style.CYAN+'Di-lep fakes for '+str(nb)+' bjets'+style.RESET

                nb_mask = m_1bjet if nb==1 else m_2bjet

                subdf = data_dict['data_nb_TauSR'][nb_mask]
                subdf.name='_TauSR_'+str(nb)+'b'

                hists = HistMaker(subdf, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [3,1,4], region_names=region_names )
                result = Fit_DiTau(subdf.copy(), hists, dict_tmps_comb, mymsg=mymsg)
                result['nb'] = nb

                # to latex table
                print_to_latex_values(latexfile, result, 'di-lep')

                # store results to dict
                results['SFs_'+str(nb)+'b'] = [result['SF_simjet_est'], result['SF_jetsim_est'], result['SF_jetjet_est']]
                results['SFs_'+str(nb)+'b_err'] = [result['SF_simjet_err_est'], result['SF_jetsim_err_est'], result['SF_jetjet_err_est']]

            # Finish the latex table
            latexfile.write('    \\bottomrule\n')
            latexfile.write('  \\end{tabular}\n')
            latexfile.write('\\end{table}')

            if config['EXTRACT_LEPBKG_SAVE']:
                print(results)
                from BkgFitter import result_to_root
                outfilename = 'taufakes_dilep.root'
                logging.info("Saving results to {}".format(outfilename))
                result_to_root(results, outfilename)



        elif config['AnalysisChannel']=='hadhad':
            logging.info(style.YELLOW+'Computing Scale-Factors for lepton fakes with Counting method'+style.RESET)
            from BkgFitter import TauFakeYieldCorrector
            bins = dict_hists['pt_lep1']

            # Loop over b-jets ( config['BJETS'] or [1,2] )
            for nb in config['BJETS']:

                nb_mask = m_1bjet if nb==1 else m_2bjet

                subdf = data_dict['data_Lep1CR'][nb_mask]
                subdf.name='_Lep1CR_'+str(nb)+'b'

                hists_fit = HistMaker(subdf, 'lep_1_tmp', dict_tmps_lep, 'pt_lep1', bins)

                mymsg = style.CYAN+'Light lepton fake: '+str(nb)+' bjets'+style.RESET
                TauFakeYieldCorrector(hists_fit, mymsg=mymsg)

                #from LFAHelpers import BkgYieldCorrector
                #NF, NF_err = BkgYieldCorrector(hists_fit)





    logging.info("Done: --- {t} seconds ---".format(t = round(time.time()-start_time, 3)))

if __name__ == "__main__":
    main()