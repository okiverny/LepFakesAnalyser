import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import ROOT
import root_numpy as rootnp
import glob
import os
import ctypes

from LFAHelpers import style

############################################################
###                  Helper functions                    ###
############################################################
# General
from LFAHelpers import style, ensure_dir
# read/write data
from LFAHelpers import load_data_rootnp
# Histos/Plotting/Pie-Charts
from LFAHelpers import HistMaker, Plotter, PieCharter
# Data pre-processing
from LFAHelpers import sort_taus_ditau, TauTempAnnotator, LepTempAnnotator,TempCombAnnotator,RegionsAnnotator
# Fake background
from LFAHelpers import BkgYieldCorrector, BKGCorrector, BKGCorrector_DiTau,LepBKGCorrector,BKGCorrector_DiTauSyst,BKGCorrector_TauSyst_OS_1b
from LFAHelpers import UncertantyMaker
from LFAHelpers import SystSolver
from LFAConfig import Configurate, Configurate_VarHistBins
from LFAFitter import Fit_DiTau

import warnings
import matplotlib.cbook
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)

# ROOT config
ROOT.gROOT.SetBatch(True) # No graphics displayed
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetStyle("ATLAS")
ROOT.gErrorIgnoreLevel = ROOT.kWarning


############################################################
###              Main script configuration               ###
############################################################

tree_name, dict_samples, dict_tmps_lep, dict_tmps_tau, dict_tmps_tau_fine, list_of_branches_leptons, list_of_branches_mc, input_folder = Configurate('hadhad')

### For development
USE_FRAC_DATA = False

### Plot Flags
PLOT_PROCESS  = True
PLOT_LEPTMP   = True
PLOT_TAUTMP   = True
PLOT_COMBTMP  = True

### specific plot comparing SR and CRs
PLOT_CR_SUMMARY = True

### q/g initiated jets split? (default=False)
SPLIT_JETSTMP = False

### BKG Extractions
EXTRACT_DITAUFAKE = False
EXTRACT_LEPFAKE   = False

### Apply BKG Corrections
CORRECT_BKG_DITAU = True
CORRECT_BKG_LEP   = True

#####################################################################
###                          Read ntuples                         ###
#####################################################################
print(len(dict_samples['sample']))
print(len(dict_samples['fillcolor']))
print(dict_samples['sample'])


# Load data from root files
data_df = load_data_rootnp(input_folder, dict_samples, list_of_branches_leptons, list_of_branches_mc)

# For development
if USE_FRAC_DATA:
    data_df = data_df.sample(frac=0.1, random_state=1)

# Sort light leptons and taus
data_df = sort_taus_ditau(data_df)

# Annotate Tau1 and Tau2 templates
if SPLIT_JETSTMP:
    dict_tmps_tau = dict_tmps_tau_fine
data_df = TauTempAnnotator(data_df, dict_tmps_tau, 'fs_had_tau_true_pdg', 'fs_had_tau_true_partonTruthLabelID', 'tau_1_tmp', SPLIT_JETSTMP)
data_df = TauTempAnnotator(data_df, dict_tmps_tau, 'fs_had_tau_2_true_pdg', 'fs_had_tau_2_true_partonTruthLabelID', 'tau_2_tmp', SPLIT_JETSTMP)

# Annotate Lep1 templates
data_df = LepTempAnnotator(data_df, dict_tmps_lep, 'TruthIFF_Class_lep1', 'lep_1_tmp')

# Template Combinations + dictionary
data_df = TempCombAnnotator(data_df, ['tau_1_tmp_simple','tau_2_tmp_simple'])
dict_tmps_comb = {'sample': data_df['TempCombinations'].unique().tolist(), 
                  'fillcolor': [x+1 for x in data_df['TempCombinationsEncoded'].unique().tolist()]}
### Force the correct order!
dict_tmps_comb = {'sample': ['data,data', 'sim,sim', 'sim,jet', 'jet,sim', 'jet,jet'], 'fillcolor': [1, 2, 3, 4, 5]}

print(data_df[['TempCombinations','TempCombinationsEncoded']].tail(10))
print(dict_tmps_comb)
print(data_df['TempCombinations'].tail(20))

#####################################################################
###                        Selections                             ###
#####################################################################
m_0b = data_df['m_nbjets']==0
m_1b = (data_df['m_nbjets']==1) | (data_df['m_nbjets']==2)
#m_1b = data_df['m_nbjets']==1
m_2b = data_df['m_nbjets']==2
m_2j = data_df['m_njets']==2
m_tau1_tight = data_df['fs_had_tau_1_tight']==1
m_tau2_tight = data_df['fs_had_tau_2_tight']==1
m_lep1_tight = data_df['isTight_lep1']==1
m_tau1_1p = data_df['fs_had_tau_1_nTrack']==1
m_tau2_1p = data_df['fs_had_tau_2_nTrack']==1
m_tau1_3p = data_df['fs_had_tau_1_nTrack']==3
m_tau2_3p = data_df['fs_had_tau_2_nTrack']==3

#####################################################################
###                   Ditau charge requirement                    ###
#####################################################################

m_charge = data_df['had_tau_1_charge']*data_df['had_tau_2_charge']==-1
data_df = data_df[m_charge]

#####################################################################
###                    Annotate tau pairs                         ###
#####################################################################
# Annotate tau-pair regions
masks = [m_1b & m_lep1_tight & (m_tau1_tight & ~m_tau2_tight),
         m_1b & m_lep1_tight & (~m_tau1_tight & m_tau2_tight),
         m_1b & m_lep1_tight & (~m_tau1_tight & ~m_tau2_tight)]
region_names = ['[M,nM]', '[nM,M]','[nM,nM]']
data_df = RegionsAnnotator(data_df, masks, region_names)
data_df.name='_INCL'

#####################################################################
###                   Application of corrections                  ###
#####################################################################
SystVarWeightTags=[]
SystVarWeightTags_DiTau = []
SystVarWeightTags_Lep = []

if CORRECT_BKG_DITAU:

    # Tau-pairs
    ###################### 1-bjet #######################################################################
    # # 1-prong, 1-prong
    # SF_vals = [1.0, 1.266, 1.846, 0.667]
    # SF_errs = [0.0, 0.154, 0.305, 0.122]
    # data_df = BKGCorrector_DiTau(data_df, 'TempCombinations', dict_tmps_comb, SF_vals, SF_errs, 1, 1, 1)

    # # 1-prong, 3-prong
    # SF_vals = [1.0, 1.031, 2.150, 0.976]
    # SF_errs = [0.0, 0.471, 1.267, 0.364]
    # data_df = BKGCorrector_DiTau(data_df, 'TempCombinations', dict_tmps_comb, SF_vals, SF_errs, 1, 3, 1)

    # # 3-prong, 1-prong
    # SF_vals = [1.0, 0.941, 0.798, 1.360]
    # SF_errs = [0.0, 0.194, 0.598, 0.264]
    # data_df = BKGCorrector_DiTau(data_df, 'TempCombinations', dict_tmps_comb, SF_vals, SF_errs, 3, 1, 1)

    # # 3-prong, 3-prong
    # SF_vals = [1.0, 1.187, 1.540, 0.681]
    # SF_errs = [0.0, 0.297, 0.338, 0.136]
    # data_df = BKGCorrector_DiTau(data_df, 'TempCombinations', dict_tmps_comb, SF_vals, SF_errs, 3, 3, 1)

    # ###################### 2-bjet (TBD)#######################################################################
    # # 1-prong, 1-prong
    # SF_vals = [1.0, 1.253, 1.107, 0.641]
    # SF_errs = [0.0, 0.228, 0.520, 0.219]
    # data_df = BKGCorrector_DiTau(data_df, 'TempCombinations', dict_tmps_comb, SF_vals, SF_errs, 1, 1, 2)

    # # 1-prong, 3-prong
    # SF_vals = [1.0, 0.893, 0.135, 1.184]
    # SF_errs = [0.0, 0.240, 0.491, 0.255]
    # data_df = BKGCorrector_DiTau(data_df, 'TempCombinations', dict_tmps_comb, SF_vals, SF_errs, 1, 3, 2)

    # # 3-prong, 1-prong
    # SF_vals = [1.0, 1.554, 1.958, 0.616]
    # SF_errs = [0.0, 0.406, 0.598, 0.323]
    # data_df = BKGCorrector_DiTau(data_df, 'TempCombinations', dict_tmps_comb, SF_vals, SF_errs, 3, 1, 2)

    # # 3-prong, 3-prong
    # SF_vals = [1.0, 0.772, 1.615, 1.317]
    # SF_errs = [0.0, 0.417, 1.271, 0.391]
    # data_df = BKGCorrector_DiTau(data_df, 'TempCombinations', dict_tmps_comb, SF_vals, SF_errs, 3, 3, 2)

    ######################################################################################################

    SystVarWeightTags_DiTau = ['ditau_simjet_up', 'ditau_simjet_down',
                             'ditau_jetsim_up', 'ditau_jetsim_down',
                             '']
    for systtag in SystVarWeightTags_DiTau:
        data_df = BKGCorrector_DiTauSyst(data_df, systtag) # correct both 1b and 2b

    ###################### 1D X 1D method ########################
    # SystVarWeightTags_DiTau = ['tau_shape_up', 'tau_shape_down',
    #                          'tau_norm_up', 'tau_norm_down',
    #                          '']
    # for systtag in SystVarWeightTags_DiTau:
    #     data_df = BKGCorrector_TauSyst_OS_1b(data_df, 'QGMethod', '', systtag) # leading tau
    # #data_df = BKGCorrector_TauSyst_OS_1b(data_df, 'QGMethod', '', '') # leading tau

    # rename_dict = {'weight_tau_shape_up': 'weight_tau_shape_up_1',
    #                'weight_tau_shape_down': 'weight_tau_shape_down_1',
    #                'weight_tau_norm_up': 'weight_tau_norm_up_1',
    #                'weight_tau_norm_down': 'weight_tau_norm_down_1'}
    # SystVarWeightTags_Tau1=['tau_shape_up_1', 'tau_shape_down_1', 'tau_norm_up_1', 'tau_norm_down_1']
    # data_df.rename(columns=rename_dict, inplace=True)

    # for systtag in SystVarWeightTags_DiTau:
    #     data_df = BKGCorrector_TauSyst_OS_1b(data_df, 'QGMethod', '_2', systtag) # leading tau

    # # Final list of systematics
    # SystVarWeightTags_DiTau = SystVarWeightTags_Tau1+SystVarWeightTags_DiTau
    # print(SystVarWeightTags_DiTau)

    ###################### end of 1D X 1D method ########################

    #data_df = BKGCorrector_Marvin(data_df, 'QGMethod', '')
    #data_df = BKGCorrector_Marvin(data_df, 'QGMethod', '_2')

if CORRECT_BKG_LEP:
    # Lepton
    data_df = LepBKGCorrector(data_df, 'lep_1_tmp')

    #data_df = BKGCorrector_Oleh(data_df)
    #data_df = BKGCorrector_Marvin(data_df, 'QGMethod', '') # 1Bin
    #data_df = BKGCorrector_Marvin(data_df, 'QGMethod', '_2')

### Combine weight tags
if CORRECT_BKG_DITAU:
    SystVarWeightTags += SystVarWeightTags_DiTau

#####################################################################
###                     Region Selections                         ###
#####################################################################

data_1b = data_df[m_1b]
data_1b_LepSR = data_df[m_1b & m_lep1_tight]
data_SR = data_df[ m_1b & m_lep1_tight & m_tau1_tight & m_tau2_tight] # m_tau1_tight m_tau2_tight    & m_tau1_tight & m_tau2_tight       & m_tau1_1p & m_tau2_1p
data_Tau1CR = data_df[ m_1b & ~m_tau1_tight & m_tau2_tight & m_lep1_tight]
data_Tau2CR = data_df[ m_1b & m_tau1_tight & ~m_tau2_tight & m_lep1_tight]
data_Lep1CR = data_df[ m_1b & m_tau1_tight & m_tau2_tight & ~m_lep1_tight]

# Names of data frames as tags
data_1b.name='_1b'
data_1b_LepSR.name='_1b_lepT'
data_SR.name='_SR'
data_Tau1CR.name='_Tau1CR'
data_Tau2CR.name='_Tau2CR'
data_Lep1CR.name='_Lep1CR'


#####################################################################
###                          Pie Charts                           ###
#####################################################################
print(style.YELLOW+"Plotting Pie-Chart."+style.RESET)
ensure_dir('PieCharts/')

PieCharter(data_SR, 'sample_Id', dict_samples, "SR_hadhad.pdf", show_fractions=True)
PieCharter(data_SR, 'tau_1_tmp', dict_tmps_tau, "SR_tau1tmp_hadhad.pdf")
PieCharter(data_SR, 'tau_2_tmp', dict_tmps_tau, "SR_tau2tmp_hadhad.pdf")
PieCharter(data_SR, 'lep_1_tmp', dict_tmps_lep, "SR_lep1tmp_hadhad.pdf")

for iproc, process in enumerate(dict_samples['sample']):
    if process == 'data': continue
    process_cleaned = process.replace('/', '_')
    PieCharter(data_SR[data_SR['sample_Id'] == iproc], 'tau_1_tmp', dict_tmps_tau, 'PieCharts/SR_'+process_cleaned+'_tau1tmp_hadhad.pdf')
    PieCharter(data_SR[data_SR['sample_Id'] == iproc], 'tau_2_tmp', dict_tmps_tau, 'PieCharts/SR_'+process_cleaned+'_tau2tmp_hadhad.pdf')
    PieCharter(data_SR[data_SR['sample_Id'] == iproc], 'lep_1_tmp', dict_tmps_lep, 'PieCharts/SR_'+process_cleaned+'_lep1tmp_hadhad.pdf')
    PieCharter(data_SR[data_SR['sample_Id'] == iproc], 'TempCombinations', dict_tmps_comb, 'PieCharts/SR_'+process_cleaned+'_combtmp_hadhad.pdf')


#####################################################################
###                         Plot Processes                        ###
#####################################################################
dict_hists = Configurate_VarHistBins()

if PLOT_PROCESS:
    for var in ['fs_had_tau_1_pt','fs_had_tau_2_pt','fs_had_tau_1_RNNScore','fs_had_tau_2_RNNScore','m_njets', 'pt_lep1','eta_lep1', 'fs_had_tau_1_eta','fs_had_tau_2_eta']:
        bins = dict_hists[var]
        hists = HistMaker(data_SR, 'process', dict_samples, var, bins)

        # Compute syst variations if available (CORRECT_BKG_TAU=True)
        if CORRECT_BKG_DITAU: # or CORRECT_BKG_LEP:
            UncertaintyBand,UncertaintyBandRatio = UncertantyMaker(data_SR, 'process', dict_samples, var, bins, SystVarWeightTags[:-1])
            HistMaker(data_SR, 'process', dict_samples, var, bins, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)


    for var in ['pt_lep1','eta_lep1']:
        bins = dict_hists[var]
        hists = HistMaker(data_Lep1CR, 'process', dict_samples, var, bins)

#####################################################################
###                         SR+CRs SUMMARY                        ###
#####################################################################
if PLOT_CR_SUMMARY:
    # For correction illustration
    masks_CI = [ m_1b & m_lep1_tight & (m_tau1_tight & m_tau2_tight), # SR
                 m_1b & ~m_lep1_tight & (m_tau1_tight & m_tau2_tight), # Lep CR
                 m_1b & m_lep1_tight & (m_tau1_tight & ~m_tau2_tight), 
                 m_1b & m_lep1_tight & (~m_tau1_tight & m_tau2_tight),
                 m_1b & m_lep1_tight & (~m_tau1_tight & ~m_tau2_tight)]
    region_names_CI = ['SR=TMM', '#bar{T}MM', 'TM#bar{M}', 'T#bar{M}M', 'T#bar{M}#bar{M}']
    data_df_CI = RegionsAnnotator(data_df.copy(), masks_CI, region_names_CI)
    data_df_CI.name='_INCL'

    nbins = len(masks_CI)+1
    hists = HistMaker(data_df_CI, 'process', dict_samples, 'regions_encoded', [5,1,6], region_names=region_names_CI )
    #hists = HistMaker(data_df_CI, 'tau_1_tmp', dict_tmps_tau, 'regions_encoded', [5,1,6], region_names=region_names_CI )
    #hists = HistMaker(data_df_CI, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [5,1,6], region_names=region_names_CI )

    ### compute syst variations if available (CORRECT_BKG_DITAU=True)
    if CORRECT_BKG_DITAU:
        UncertaintyBand,UncertaintyBandRatio = UncertantyMaker(data_df_CI, 'process', dict_samples, 'regions_encoded', [5,1,6], SystVarWeightTags[:-1])
        HistMaker(data_df_CI, 'process', dict_samples, 'regions_encoded', [5,1,6], region_names=region_names_CI, UncertaintyBand=UncertaintyBand, UncertaintyBandRatio=UncertaintyBandRatio)


#####################################################################
###                  Tau Combination Templates                    ###
#####################################################################

if PLOT_COMBTMP:
    hists = HistMaker(data_1b_LepSR, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [4,0,4], region_names=['SR']+region_names)

if  EXTRACT_DITAUFAKE:
    hists = HistMaker(data_1b_LepSR, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [3,1,4], region_names=region_names )
    datacor = Fit_DiTau(data_1b_LepSR.copy(), hists, dict_tmps_comb)
    datacor.name = '_cor'
    # Post-Fit plot
    HistMaker(datacor, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [4,0,4], region_names=['SR']+region_names )

    # Pre-Fit plot
    hists = HistMaker(data_1b_LepSR, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [4,0,4], region_names=['SR']+region_names )
    # Exact solution for XCheck
    SystSolver(hists, dict_tmps_comb, ['SR']+region_names)

    # data_df = Fit_DiTau(data_df, hists, dict_tmps_comb)
    # # Plot Corrected MC (rewrites)
    # hists = HistMaker(data_df, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [3,1,4], region_names)


    for prong1 in [1, 3]:
        for prong2 in [1,3]:
            print(style.RED+str(prong1)+'-prong,'+str(prong2)+'-prong'+style.RESET)

            prong1_mask = m_tau1_1p if prong1==1 else m_tau1_3p
            prong2_mask = m_tau2_1p if prong2==1 else m_tau2_3p

            subdf = data_1b_LepSR[prong1_mask & prong2_mask]
            subdf.name='_Tau1CR_'+str(prong1)+'p_'+str(prong2)+'p'

            hists = HistMaker(subdf, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [3,1,4], region_names=region_names )
            datacor = Fit_DiTau(subdf.copy(), hists, dict_tmps_comb)
            datacor.name = '_Tau1CR_'+str(prong1)+'p_'+str(prong2)+'p_cor'

            # Post-Fit plot
            HistMaker(datacor, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [4,0,4], region_names=['SR']+region_names )
            # Pre-Fit plot
            hists = HistMaker(subdf, 'TempCombinations', dict_tmps_comb, 'regions_encoded', [4,0,4], region_names=['SR']+region_names )
            # Exact analytical solution for X-check
            SystSolver(hists, dict_tmps_comb, ['SR']+region_names)

#####################################################################
###               PLOTTING BLOCK: Tau Templates                   ###
#####################################################################

if PLOT_TAUTMP:
    #------------------------Signal Region---------------------------
    for var in ['fs_had_tau_1_pt','fs_had_tau_1_eta','m_njets','fs_had_tau_1_RNNScore','fs_had_tau_true_pdg','fs_had_tau_true_partonTruthLabelID']:
        bins = dict_hists[var]
        hists = HistMaker(data_SR, 'tau_1_tmp', dict_tmps_tau, var, bins)

    for var in ['fs_had_tau_2_pt','fs_had_tau_2_eta','m_njets','fs_had_tau_2_RNNScore','fs_had_tau_2_true_pdg','fs_had_tau_2_true_partonTruthLabelID']:
        bins = dict_hists[var]
        hists = HistMaker(data_SR, 'tau_2_tmp', dict_tmps_tau, var, bins)

    #------------------------Control Regions---------------------------
    for var in ['fs_had_tau_1_pt','fs_had_tau_1_RNNScore','m_njets','fs_had_tau_true_pdg','fs_had_tau_true_partonTruthLabelID', 'fs_had_tau_1_eta']:
        bins = dict_hists[var]
        hists = HistMaker(data_Tau1CR, 'tau_1_tmp', dict_tmps_tau, var, bins)

    for var in ['fs_had_tau_2_pt','fs_had_tau_2_RNNScore','m_njets','fs_had_tau_2_true_pdg','fs_had_tau_2_true_partonTruthLabelID', 'fs_had_tau_2_eta']:
        bins = dict_hists[var]
        hists = HistMaker(data_Tau2CR, 'tau_2_tmp', dict_tmps_tau, var, bins)


#####################################################################
###              PLOTTING BLOCK: Lep1 Templates                   ###
#####################################################################

if PLOT_LEPTMP:
    for var in ['pt_lep1','eta_lep1','m_njets','TruthIFF_Class_lep1','fs_had_tau_true_partonTruthLabelID']:
        bins = dict_hists[var]
        hists = HistMaker(data_SR, 'lep_1_tmp', dict_tmps_lep, var, bins)

    #---------------------------------------------------
    for var in ['pt_lep1','eta_lep1','m_njets','TruthIFF_Class_lep1','fs_had_tau_true_partonTruthLabelID']:
        bins = dict_hists[var]
        hists = HistMaker(data_Lep1CR, 'lep_1_tmp', dict_tmps_lep, var, bins)


#####################################################################
###                PLOTTING BLOCK: Combined Templates             ###
#####################################################################
# bins = dict_hists['TempCombinationsEncoded']
# hists = HistMaker(data_SR, 'TempCombinations', dict_tmps_comb, 'TempCombinationsEncoded', bins)

# bins = dict_hists['m_njets']
# hists = HistMaker(data_SR, 'TempCombinations', dict_tmps_comb, 'm_njets', bins)

# bins = dict_hists['fs_had_tau_2_pt']
# hists = HistMaker(data_Tau2CR, 'TempCombinations', dict_tmps_comb, 'fs_had_tau_2_pt', bins)

#####################################################################
###        Analytical N-factor estimate for lepton fake           ###
#####################################################################
if EXTRACT_LEPFAKE:
    bins = dict_hists['pt_lep1']
    hists_fit = HistMaker(data_Lep1CR, 'lep_1_tmp', dict_tmps_lep, 'pt_lep1', bins)
    NF, NF_err = BkgYieldCorrector(hists_fit)




#HistFitter(hists, dict_tmps_comb)