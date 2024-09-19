import numpy as np
import pandas as pd
import logging
import math

from LFAHelpers import style

####################################################################
###                    Manipulations with Data                   ###
####################################################################

# For lephad channel only
def apply_lep3_cut(data_df, ptcut_value):
    '''
    Apply lep3 > ptcut_value cut
    input:
        data_df: dataframe with all variables
        ptcut_value: cut value in GeV
    output:
        data_df: dataframe with lep3 cut applied
    '''
    logging.info(" ")
    logging.info("======================= Applying lep3 > {} GeV cut ======================= ".format(ptcut_value))
    logging.info(style.YELLOW+"Applying lep3 > {} GeV cut".format(ptcut_value)+style.RESET)
    mask = data_df['pt_lep3']>ptcut_value
    data_df = data_df[mask]
    return data_df

# Split taus and light leptons
def sort_taus(dframe):
    '''
    Sort taus and light leptons according to truth information. Only for single tau channel.
    input:
        dframe: dataframe with all variables
    output:
        dframe: dataframe with taus and light leptons sorted
    '''

    # Some print outs
    logging.info(" ")
    logging.info("======================= Splitting light leptons and taus ======================= ")
    logging.info(style.YELLOW+"Sorting lepX variables and separating taus."+style.RESET)

    # Store tau charge (not implemented in had_tau variables)
    dframe['had_tau_1_charge'] = dframe['charge_lep2'] # dummy initialization

    # if lep1 is tau: move lep2 to lep1 and lep3 to lep2
    mask = dframe['type_lep1']==3
    dframe.loc[mask, 'pt_lep1'] = dframe[mask]['pt_lep2']
    dframe.loc[mask, 'eta_lep1'] = dframe[mask]['eta_lep2']
    dframe.loc[mask, 'had_tau_1_charge'] = dframe[mask]['charge_lep1']
    dframe.loc[mask, 'charge_lep1'] = dframe[mask]['charge_lep2']
    dframe.loc[mask, 'isTight_lep1'] = dframe[mask]['isTight_lep2']
    dframe.loc[mask, 'type_lep1'] = dframe[mask]['type_lep2']
    dframe.loc[mask, 'TruthIFF_Class_lep1'] = dframe[mask]['TruthIFF_Class_lep2']
    dframe.loc[mask, 'ECIDS_lep1'] = dframe[mask]['ECIDS_lep2']
    dframe.loc[mask, 'ele_ambiguity_lep1'] = dframe[mask]['ele_ambiguity_lep2']
    dframe.loc[mask, 'ele_AddAmbiguity_lep1'] = dframe[mask]['ele_AddAmbiguity_lep2']

    dframe.loc[mask, 'pt_lep2'] = dframe[mask]['pt_lep3']
    dframe.loc[mask, 'eta_lep2'] = dframe[mask]['eta_lep3']
    dframe.loc[mask, 'charge_lep2'] = dframe[mask]['charge_lep3']
    dframe.loc[mask, 'isTight_lep2'] = dframe[mask]['isTight_lep3']
    dframe.loc[mask, 'type_lep2'] = dframe[mask]['type_lep3']
    dframe.loc[mask, 'TruthIFF_Class_lep2'] = dframe[mask]['TruthIFF_Class_lep3']
    dframe.loc[mask, 'ECIDS_lep2'] = dframe[mask]['ECIDS_lep3']
    dframe.loc[mask, 'ele_ambiguity_lep2'] = dframe[mask]['ele_ambiguity_lep3']
    dframe.loc[mask, 'ele_AddAmbiguity_lep2'] = dframe[mask]['ele_AddAmbiguity_lep3']

    # if lep2 is tau: move lep3 to lep2
    mask = dframe['type_lep2']==3
    dframe.loc[mask, 'pt_lep2'] = dframe[mask]['pt_lep3']
    dframe.loc[mask, 'eta_lep2'] = dframe[mask]['eta_lep3']
    dframe.loc[mask, 'had_tau_1_charge'] = dframe[mask]['charge_lep2']
    dframe.loc[mask, 'charge_lep2'] = dframe[mask]['charge_lep3']
    dframe.loc[mask, 'isTight_lep2'] = dframe[mask]['isTight_lep3']
    dframe.loc[mask, 'type_lep2'] = dframe[mask]['type_lep3']
    dframe.loc[mask, 'TruthIFF_Class_lep2'] = dframe[mask]['TruthIFF_Class_lep3']
    dframe.loc[mask, 'ECIDS_lep2'] = dframe[mask]['ECIDS_lep3']
    dframe.loc[mask, 'ele_ambiguity_lep2'] = dframe[mask]['ele_ambiguity_lep3']
    dframe.loc[mask, 'ele_AddAmbiguity_lep2'] = dframe[mask]['ele_AddAmbiguity_lep3']

    # to move lep3_charge to had_tau_charge
    mask = dframe['type_lep3']==3
    dframe.loc[mask, 'had_tau_1_charge'] = dframe[mask]['charge_lep3']
    
    # drop lep3 columns
    dframe = dframe.drop(['pt_lep3','eta_lep3','charge_lep3','type_lep3','isTight_lep3', 'TruthIFF_Class_lep3',
                          'ECIDS_lep3', 'ele_ambiguity_lep3', 'ele_AddAmbiguity_lep3'], axis=1)

    return dframe

def sort_taus_ditau(dframe):
    '''
    Sort taus and light leptons according to truth information. Only for ditau channel.
    input:
        dframe: dataframe with all variables
    output:
        dframe: dataframe with taus and light leptons sorted
    '''

    # Some print outs
    logging.info("======================= Splitting light leptons and taus ======================= ")
    logging.info(style.YELLOW+"Sorting lepX variables and removing taus."+style.RESET)

    # if lep2 is elec: move lep2 to lep1
    mask = dframe['type_lep2']==1
    dframe.loc[mask, 'pt_lep1'] = dframe[mask]['pt_lep2']
    dframe.loc[mask, 'eta_lep1'] = dframe[mask]['eta_lep2']
    dframe.loc[mask, 'charge_lep1'] = dframe[mask]['charge_lep2']
    dframe.loc[mask, 'isTight_lep1'] = dframe[mask]['isTight_lep2']
    dframe.loc[mask, 'type_lep1'] = dframe[mask]['type_lep2']
    dframe.loc[mask, 'TruthIFF_Class_lep1'] = dframe[mask]['TruthIFF_Class_lep2']

    # if lep3 is elec: move lep3 to lep1
    mask = dframe['type_lep3']==1
    dframe.loc[mask, 'pt_lep1'] = dframe[mask]['pt_lep3']
    dframe.loc[mask, 'eta_lep1'] = dframe[mask]['eta_lep3']
    dframe.loc[mask, 'charge_lep1'] = dframe[mask]['charge_lep3']
    dframe.loc[mask, 'isTight_lep1'] = dframe[mask]['isTight_lep3']
    dframe.loc[mask, 'type_lep1'] = dframe[mask]['type_lep3']
    dframe.loc[mask, 'TruthIFF_Class_lep1'] = dframe[mask]['TruthIFF_Class_lep3']

    # if lep2 is muon: move lep2 to lep1
    mask = dframe['type_lep2']==2
    dframe.loc[mask, 'pt_lep1'] = dframe[mask]['pt_lep2']
    dframe.loc[mask, 'eta_lep1'] = dframe[mask]['eta_lep2']
    dframe.loc[mask, 'charge_lep1'] = dframe[mask]['charge_lep2']
    dframe.loc[mask, 'isTight_lep1'] = dframe[mask]['isTight_lep2']
    dframe.loc[mask, 'type_lep1'] = dframe[mask]['type_lep2']
    dframe.loc[mask, 'TruthIFF_Class_lep1'] = dframe[mask]['TruthIFF_Class_lep2']

    # if lep3 is muon: move lep3 to lep1
    mask = dframe['type_lep3']==2
    dframe.loc[mask, 'pt_lep1'] = dframe[mask]['pt_lep3']
    dframe.loc[mask, 'eta_lep1'] = dframe[mask]['eta_lep3']
    dframe.loc[mask, 'charge_lep1'] = dframe[mask]['charge_lep3']
    dframe.loc[mask, 'isTight_lep1'] = dframe[mask]['isTight_lep3']
    dframe.loc[mask, 'type_lep1'] = dframe[mask]['type_lep3']
    dframe.loc[mask, 'TruthIFF_Class_lep1'] = dframe[mask]['TruthIFF_Class_lep3']
    
    # drop lep3 columns
    dframe = dframe.drop(['pt_lep3','eta_lep3','charge_lep3','type_lep3','isTight_lep3', 'TruthIFF_Class_lep3'], axis=1)
    # drop lep2 columns
    dframe = dframe.drop(['pt_lep2','eta_lep2','charge_lep2','type_lep2','isTight_lep2', 'TruthIFF_Class_lep2'], axis=1)

    return dframe

####################################################################
###                Labeling events based on truth                ###
####################################################################

def TauTempAnnotator(data, dict_tmps_tau, tmp_var_lep, tmp_var_jet, new_var, splitJets=False):
    logging.info(style.YELLOW+"Labeling tau templates based on truth matching to leptons and jets."+style.RESET)
    logging.info("      Input variables: '{v1}' and '{v2}' ".format(v1 = tmp_var_lep, v2 = tmp_var_jet))
    logging.info("      Output variable: '{v}' with values from {t}".format(v = new_var, t = dict_tmps_tau['sample']) )
    logging.info("      Output variable: '{v}' with values from {t} with 'jet'={j}".format(v = new_var+'_simple', t = ['data','jet','sim'], j='jet/q-jet/g-jet/unknown') )

    def annotate_tau(x_lep, x_jet, dict_tmps_tau):

        if math.isnan(x_lep):
            return dict_tmps_tau['sample'][0]    # data
        elif abs(x_lep) == 15 and x_jet<0:
            return dict_tmps_tau['sample'][1]    # tau
        elif abs(x_lep) == 11 and x_jet<0:
            return dict_tmps_tau['sample'][2]    # elec
        elif abs(x_lep) == 13 and x_jet<0:
            return dict_tmps_tau['sample'][3]    # muon

        elif x_jet>0:
            if not splitJets:
                return dict_tmps_tau['sample'][4]    # jet
            else:
                if x_jet<21:
                    return dict_tmps_tau['sample'][4]    # q-jet
                elif x_jet==21:
                    return dict_tmps_tau['sample'][5]    # g-jet
        else:
            if not splitJets:
                return dict_tmps_tau['sample'][5]    # unknown
            else:
                return dict_tmps_tau['sample'][6]    # unknown

    data[new_var] = data.apply(
        lambda row: annotate_tau(row[tmp_var_lep], row[tmp_var_jet], dict_tmps_tau),
        axis=1
    )
    data[new_var+'_simple'] = data[new_var].apply(lambda x: 'jet' if (x=='jet' or x=='q-jet' or x=='g-jet' or x=='unknown') else 'data' if x=='data' else 'sim')
    #data[new_var+'_simple'] = data[new_var].apply(lambda x: 'jet' if (x=='jet' or x=='q-jet' or x=='g-jet') else 'data' if x=='data' else 'unknown' if x=='unknown' else 'sim')

    return data

def LepTempAnnotator(data, dict_tmps_lep, tmp_var, new_var):
    logging.info(style.YELLOW+"Labeling light lepton templates."+style.RESET)
    logging.info("      Input variable: '{v}' ".format(v = tmp_var) )
    logging.info("      Output variable: '{v}' with values from {t}".format(v = new_var, t = dict_tmps_lep['sample']) )
    logging.info("      Output variable: '{v}' with values from {t} with 'sim' = {s}".format(v = new_var+'_simple', t=['data','jet','sim'], s=[i for i in dict_tmps_lep['sample'] if i!='data' and i!='jet']) )

    def annotate_lep(x, dict_tmps_lep):
        if math.isnan(x):
            return dict_tmps_lep['sample'][0]    # data
        elif x == 2:
            return dict_tmps_lep['sample'][1]    # elec
        elif x == 4:
            return dict_tmps_lep['sample'][2]    # muon
        elif x == 3:
            return dict_tmps_lep['sample'][3]    # c-flips
        elif x >= 8:
            return dict_tmps_lep['sample'][4]    # jet
        else:
            return dict_tmps_lep['sample'][5]    # rest

    data[new_var] = data[tmp_var].apply(
        lambda row: annotate_lep(row, dict_tmps_lep),
    )
    data[new_var+'_simple'] = data[new_var].apply(lambda x: x if x=='jet' else 'data' if x=='data' else 'sim')

    return data

def TempCombAnnotator(data, templators):
    logging.info(style.YELLOW+"Labeling light lepton or tau pairs."+style.RESET)
    logging.info("      Input variables: {vars}' ".format(vars = templators))

    data['TempCombinations'] = data[templators].agg(lambda row: ','.join(row), axis=1)
    data['TempCombinationsEncoded'] = data['TempCombinations'].factorize()[0]

    logging.info("      Output variable: '{v}' = {t}".format(v = 'TempCombinations', t=data['TempCombinations'].unique().tolist()))
    logging.info("      Output variable: '{v}' = {t}".format(v = 'TempCombinationsEncoded', t=data['TempCombinationsEncoded'].unique().tolist()))

    return data

def RegionsAnnotator(data, masks, region_names):
    '''
    Annotate regions according to the masks.
    Input:
        data: dataframe
        masks: list of masks
        region_names: list of region names
    Output:
        data: dataframe with new columns 'regions_encoded' and 'regions'
    '''
    logging.info((style.YELLOW+"Annotating regions:"+style.RESET+" {}".format(region_names)))
    data['regions_encoded'] = 0
    data['regions'] = ''
    for i, mask in enumerate(masks):
        data.loc[mask, 'regions_encoded'] = i+1
        data.loc[mask, 'regions'] = region_names[i]
    return data


######################################################################
#####          Correct the single tau fakes with syst            #####
######################################################################
def BKGCorrector_Tau(data, cortype, which_tau, sys):

    if (sys==''):
        logging.info(style.YELLOW+"Correcting single tau background templates derived with Template Fit."+style.RESET)
    else:
        logging.info(style.YELLOW+"Computing weight for single tau background templates derived with Template Fit ("+str(sys)+")."+style.RESET)

    # Init syst flags
    is_tau_shape    = 0
    is_tau_norm     = 0
    is_ditau_simjet = 0
    is_ditau_jetsim = 0
    is_lep_sys      = 0
    is_dilep_simjet = 0
    is_dilep_jetsim = 0

    if sys=='tau_shape_up': is_tau_shape = 1
    elif (sys=="tau_shape_down"): is_tau_shape = -1
    elif (sys=="tau_norm_up"): is_tau_norm = 1
    elif (sys=="tau_norm_down"): is_tau_norm = -1
    elif (sys=="ditau_simjet_up"): is_ditau_simjet = 1
    elif (sys=="ditau_simjet_down"): is_ditau_simjet = -1
    elif (sys=="ditau_jetsim_up"): is_ditau_jetsim = 1
    elif (sys=="ditau_jetsim_down"): is_ditau_jetsim = -1

    elif (sys=="lep_sys_up"): is_lep_sys = 1
    elif (sys=="lep_sys_down"): is_lep_sys = -1
    elif (sys=="dilep_simjet_up"): is_dilep_simjet = 1
    elif (sys=="dilep_simjet_down"): is_dilep_simjet = -1
    elif (sys=="dilep_jetsim_up"): is_dilep_jetsim = 1
    elif (sys=="dilep_jetsim_down"): is_dilep_jetsim = -1

    # Construct a tag for association of the tau with the corresponding histo name
    which_tau_cor=''
    if which_tau=='':
        which_tau_cor = '_1'
    else:
        which_tau_cor = which_tau

    # Vectorized implementation
    if cortype=='QGMethod':
        tag = sys
        if sys!='':             # if syst variation, creat new weight coulmn
            if 'weight_'+sys not in data.columns:
                data['weight_'+sys] = data['weight_nominal']
        else:
            tag = 'nominal'

        # if considering the subleading tau
        weight_ref = 'weight_nominal'
        if which_tau_cor=='_2':
            weight_ref = 'weight_'+tag

        # Key variables
        sample = data['process']
        nTracks = data['fs_had_tau'+which_tau_cor+'_nTrack']
        pt = data['fs_had_tau'+which_tau_cor+'_pt']
        partonTruthLabelID = data['fs_had_tau'+which_tau+'_true_partonTruthLabelID']
        pdg = data['fs_had_tau'+which_tau+'_true_pdg']
        nbjets = data['m_nbjets']

        ############################################################################
        ####                    1b / 1-Tack (Default)
        ############################################################################
        ## 1b / 1-Tack / gluon-jet / pt1
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=20) & (pt<30)
        normfactor = 2.184 - is_tau_shape*0.180 + is_tau_norm*0.028
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 1b / 1-Tack / gluon-jet / pt2
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=30) & (pt<40)
        normfactor = 2.235 - is_tau_shape*0.455 + is_tau_norm*0.039
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 1b / 1-Tack / gluon-jet / pt3
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=40)
        normfactor = 1.972 - is_tau_shape*0.474 + is_tau_norm*0.032
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        ############################################################################
        ## 1b / 1-Tack / quark-jet / pt1
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=20) & (pt<30)
        normfactor = 0.613 + is_tau_shape*0.064 + is_tau_norm*0.028
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 1b / 1-Tack / quark-jet / pt2
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=30) & (pt<40)
        normfactor = 0.666 + is_tau_shape*0.096 + is_tau_norm*0.039
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 1b / 1-Tack / quark-jet / pt3
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=40)
        normfactor = 0.710 + is_tau_shape*0.056 + is_tau_norm*0.032
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        ############################################################################
        ## 1b / 1-Tack / unknown / pt1
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=20) & (pt<30)
        normfactor = 0.830
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 1b / 1-Tack / unknown / pt2
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=30) & (pt<40)
        normfactor = 0.902
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 1b / 1-Tack / unknown / pt3
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=40)
        normfactor = 0.952
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        ############################################################################
        ####                    1b / 3-Tack (Default)
        ############################################################################
        ## 1b / 3-Tack / gluon-jet / pt1
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=20) & (pt<30)
        normfactor = 1.684 - is_tau_shape*0.267 + is_tau_norm*0.027
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 1b / 3-Tack / gluon-jet / pt2
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=30) & (pt<40)
        normfactor = 0.936 - is_tau_shape*0.511 + is_tau_norm*0.037
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 1b / 3-Tack / gluon-jet / pt3
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=40)
        normfactor = 1.444 - is_tau_shape*0.726 + is_tau_norm*0.035
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        ############################################################################
        ## 1b / 3-Tack / quark-jet / pt1
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=20) & (pt<30)
        normfactor=0.829 + is_tau_shape*0.057 + is_tau_norm*0.027
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 1b / 3-Tack / quark-jet / pt2
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=30) & (pt<40)
        normfactor = 0.891 + is_tau_shape*0.081 + is_tau_norm*0.037
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 1b / 3-Tack / quark-jet / pt3
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=40)
        normfactor = 0.706 + is_tau_shape*0.067 + is_tau_norm*0.035
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        ############################################################################
        ## 1b / 3-Tack / unknown / pt1
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=20) & (pt<30)
        normfactor = 0.909
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 1b / 3-Tack / unknown / pt2
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=30) & (pt<40)
        normfactor = 1.167
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 1b / 3-Tack / unknown / pt3
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=40)
        normfactor = 1.113
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        # ############################################################################
        # ####                    1b / 1-Tack
        # ############################################################################
        # ## 1b / 1-Tack / gluon-jet / pt1
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=20) & (pt<30)
        # normfactor = 1.763 - is_tau_shape*0.241 + is_tau_norm*0.028
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 1-Tack / gluon-jet / pt2
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=30) & (pt<40)
        # normfactor = 1.723 - is_tau_shape*0.641 + is_tau_norm*0.039
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 1-Tack / gluon-jet / pt3
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=40)
        # normfactor = 1.228 - is_tau_shape*0.591 + is_tau_norm*0.032
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        # ############################################################################
        # ## 1b / 1-Tack / quark-jet / pt1
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=20) & (pt<30)
        # normfactor = 0.761 + is_tau_shape*0.085 + is_tau_norm*0.028
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 1-Tack / quark-jet / pt2
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=30) & (pt<40)
        # normfactor = 0.797 + is_tau_shape*0.135 + is_tau_norm*0.039
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 1-Tack / quark-jet / pt3
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=40)
        # normfactor = 0.797 + is_tau_shape*0.070 + is_tau_norm*0.032
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        # ############################################################################
        # ## 1b / 1-Tack / unknown / pt1
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=20) & (pt<30)
        # normfactor = 0.830
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 1-Tack / unknown / pt2
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=30) & (pt<40)
        # normfactor = 0.902
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 1-Tack / unknown / pt3
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=40)
        # normfactor = 0.952
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        # ############################################################################
        # ####                    1b / 3-Tack
        # ############################################################################
        # ## 1b / 3-Tack / gluon-jet / pt1
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=20) & (pt<30)
        # normfactor = 1.005 - is_tau_shape*0.310 + is_tau_norm*0.027
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 3-Tack / gluon-jet / pt2
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=30) & (pt<40)
        # normfactor = 0.362 - is_tau_shape*0.642 + is_tau_norm*0.037
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 3-Tack / gluon-jet / pt3
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=40)
        # normfactor = 1.041 - is_tau_shape*0.823 + is_tau_norm*0.035
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        # ############################################################################
        # ## 1b / 3-Tack / quark-jet / pt1
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=20) & (pt<30)
        # normfactor=0.973 + is_tau_shape*0.066 + is_tau_norm*0.027
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 3-Tack / quark-jet / pt2
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=30) & (pt<40)
        # normfactor = 0.982 + is_tau_shape*0.102 + is_tau_norm*0.037
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 3-Tack / quark-jet / pt3
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=40)
        # normfactor = 0.743 + is_tau_shape*0.076 + is_tau_norm*0.035
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        # ############################################################################
        # ## 1b / 3-Tack / unknown / pt1
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=20) & (pt<30)
        # normfactor = 0.909
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 3-Tack / unknown / pt2
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=30) & (pt<40)
        # normfactor = 1.167
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 3-Tack / unknown / pt3
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=40)
        # normfactor = 1.113
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        ############################################################################
        ####                    2b / 1-Tack (Default)
        ############################################################################
        ## 2b / 1-Tack / gluon-jet / pt1
        mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=20) & (pt<30)
        normfactor = 2.002 - is_tau_shape*0.225 + is_tau_norm*0.049
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 2b / 1-Tack / gluon-jet / pt2
        mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=30) & (pt<40)
        normfactor = 2.381 - is_tau_shape*0.491 + is_tau_norm*0.073
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 2b / 1-Tack / gluon-jet / pt3
        mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=40)
        normfactor = 1.526 - is_tau_shape*0.637 + is_tau_norm*0.058
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        ############################################################################
        ## 2b / 1-Tack / quark-jet / pt1
        mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=20) & (pt<30)
        normfactor = 0.705 + is_tau_shape*0.096 + is_tau_norm*0.049
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 2b / 1-Tack / quark-jet / pt2
        mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=30) & (pt<40)
        normfactor = 0.701 + is_tau_shape*0.125 + is_tau_norm*0.073
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 2b / 1-Tack / quark-jet / pt3
        mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=40)
        normfactor = 0.775 + is_tau_shape*0.076 + is_tau_norm*0.058
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        ############################################################################
        # ## 2b / 1-Tack / unknown / pt1
        mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=20) & (pt<30)
        normfactor = 0.832
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 2b / 1-Tack / unknown / pt2
        mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=30) & (pt<40)
        normfactor = 0.818
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 2b / 1-Tack / unknown / pt3
        mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=40)
        normfactor = 0.299
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        ############################################################################
        ####                    2b / 3-Tack(Default)
        ############################################################################
        ## 2b / 1-Tack / gluon-jet / pt1
        mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=20) & (pt<30)
        normfactor = 1.665 - is_tau_shape*0.344 + is_tau_norm*0.050
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 2b / 1-Tack / gluon-jet / pt2
        mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=30) & (pt<40)
        normfactor = 1.041 - is_tau_shape*0.575 + is_tau_norm*0.060
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 2b / 1-Tack / gluon-jet / pt3
        mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=40)
        normfactor = 1.334 - is_tau_shape*1.225 + is_tau_norm*0.068
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        ############################################################################
        ## 2b / 1-Tack / quark-jet / pt1
        mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=20) & (pt<30)
        normfactor = 0.994 + is_tau_shape*0.062 + is_tau_norm*0.050
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 2b / 1-Tack / quark-jet / pt2
        mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=30) & (pt<40)
        normfactor = 0.820 + is_tau_shape*0.075 + is_tau_norm*0.060
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        ## 2b / 1-Tack / quark-jet / pt3
        mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=40)
        normfactor = 0.877 + is_tau_shape*0.084 + is_tau_norm*0.068
        data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        ############################################################################
        ## 2b / 1-Tack / unknown / pt1
        mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=20) & (pt<30)
        normfactor = 1.008
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 2b / 1-Tack / unknown / pt2
        mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=30) & (pt<40)
        normfactor = 0.566
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 2b / 1-Tack / unknown / pt3
        mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=40)
        normfactor = 2.018
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        # ##### Previous values (before April 2023)
        # ############################################################################
        # ####                    1b / 1-Tack
        # ############################################################################
        # ## 1b / 1-Tack / gluon-jet / pt1
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=20) & (pt<30)
        # normfactor = 0.611 - is_tau_shape*0.297 + is_tau_norm*0.050
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 1-Tack / gluon-jet / pt2
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=30) & (pt<40)
        # normfactor = 0.993 - is_tau_shape*0.473 + is_tau_norm*0.057
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 1-Tack / gluon-jet / pt3
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=40)
        # normfactor = 1.122 - is_tau_shape*0.977 + is_tau_norm*0.042
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        # ############################################################################
        # ## 1b / 1-Tack / quark-jet / pt1
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=20) & (pt<30)
        # normfactor = 1.360 + is_tau_shape*0.106 + is_tau_norm*0.050
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 1-Tack / quark-jet / pt2
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=30) & (pt<40)
        # normfactor = 1.058 + is_tau_shape*0.103 + is_tau_norm*0.057
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 1-Tack / quark-jet / pt3
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=40)
        # normfactor = 0.909 + is_tau_shape*0.106 + is_tau_norm*0.042
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        # ############################################################################
        # ## 1b / 1-Tack / unknown / pt1
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=20) & (pt<30)
        # normfactor = 0.958
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 1-Tack / unknown / pt2
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=30) & (pt<40)
        # normfactor = 0.848
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 1-Tack / unknown / pt3
        # mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=40)
        # normfactor = 0.953
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        # ############################################################################
        # ####                    1b / 3-Tack
        # ############################################################################
        # ## 1b / 3-Tack / gluon-jet / pt1
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=20) & (pt<30)
        # normfactor = 0.719 - is_tau_shape*0.439 + is_tau_norm*0.050
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 3-Tack / gluon-jet / pt2
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=30) & (pt<40)
        # normfactor = 0.605 - is_tau_shape*0.455 + is_tau_norm*0.062
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 3-Tack / gluon-jet / pt3
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=40)
        # normfactor = 0.841 - is_tau_shape*0.784 + is_tau_norm*0.062
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        # ############################################################################
        # ## 1b / 3-Tack / quark-jet / pt1
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=20) & (pt<30)
        # normfactor=1.159 + is_tau_shape*0.097 + is_tau_norm*0.050
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 3-Tack / quark-jet / pt2
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=30) & (pt<40)
        # normfactor = 0.932 + is_tau_shape*0.070 + is_tau_norm*0.062
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 3-Tack / quark-jet / pt3
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=40)
        # normfactor = 0.897 + is_tau_shape*0.056 + is_tau_norm*0.062
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        # ############################################################################
        # ## 1b / 3-Tack / unknown / pt1
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=20) & (pt<30)
        # normfactor = 0.925
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 3-Tack / unknown / pt2
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=30) & (pt<40)
        # normfactor = 1.397
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 1b / 3-Tack / unknown / pt3
        # mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=40)
        # normfactor = 2.696
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        # ############################################################################
        # ####                    2b / 1-Tack
        # ############################################################################
        # ## 2b / 1-Tack / gluon-jet / pt1
        # mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=20) & (pt<30)
        # normfactor = 1.747 - is_tau_shape*0.234 + is_tau_norm*0.059
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 2b / 1-Tack / gluon-jet / pt2
        # mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=30) & (pt<40)
        # normfactor = 1.560 - is_tau_shape*0.596 + is_tau_norm*0.096
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 2b / 1-Tack / gluon-jet / pt3
        # mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=40)
        # normfactor = 0.918 - is_tau_shape*0.680 + is_tau_norm*0.059
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        # ############################################################################
        # ## 2b / 1-Tack / quark-jet / pt1
        # mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=20) & (pt<30)
        # normfactor = 0.810 + is_tau_shape*0.102 + is_tau_norm*0.059
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 2b / 1-Tack / quark-jet / pt2
        # mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=30) & (pt<40)
        # normfactor = 0.856 + is_tau_shape*0.157 + is_tau_norm*0.096
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 2b / 1-Tack / quark-jet / pt3
        # mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=40)
        # normfactor = 0.848 + is_tau_shape*0.081 + is_tau_norm*0.059
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        # ############################################################################
        # # ## 2b / 1-Tack / unknown / pt1
        # # mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=20) & (pt<30)
        # # normfactor = 1.0
        # # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        # # ## 2b / 1-Tack / unknown / pt2
        # # mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=30) & (pt<40)
        # # normfactor = 1.0
        # # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        # # ## 2b / 1-Tack / unknown / pt3
        # # mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=40)
        # # normfactor = 1.0
        # # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        # ############################################################################
        # ####                    2b / 3-Tack
        # ############################################################################
        # ## 2b / 1-Tack / gluon-jet / pt1
        # mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=20) & (pt<30)
        # normfactor = 1.623 - is_tau_shape*0.331 + is_tau_norm*0.061
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 2b / 1-Tack / gluon-jet / pt2
        # mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=30) & (pt<40)
        # normfactor = 1.696 - is_tau_shape*0.556 + is_tau_norm*0.066
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 2b / 1-Tack / gluon-jet / pt3
        # mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=40)
        # normfactor = 1.168 - is_tau_shape*1.168 + is_tau_norm*0.070
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        # ############################################################################
        # ## 2b / 1-Tack / quark-jet / pt1
        # mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=20) & (pt<30)
        # normfactor = 0.996 + is_tau_shape*0.063 + is_tau_norm*0.061
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 2b / 1-Tack / quark-jet / pt2
        # mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=30) & (pt<40)
        # normfactor = 0.743 + is_tau_shape*0.065 + is_tau_norm*0.066
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor
        # ## 2b / 1-Tack / quark-jet / pt3
        # mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=40)
        # normfactor = 0.897 + is_tau_shape*0.073 + is_tau_norm*0.070
        # data.loc[mask, 'weight_'+tag] = data[weight_ref] * normfactor

        # # ############################################################################
        # # ## 2b / 1-Tack / unknown / pt1
        # # mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=20) & (pt<30)
        # # normfactor = 1.0
        # # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        # # ## 2b / 1-Tack / unknown / pt2
        # # mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=30) & (pt<40)
        # # normfactor = 1.0
        # # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        # # ## 2b / 1-Tack / unknown / pt3
        # # mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=40)
        # # normfactor = 1.0
        # # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    ### Numbers for SS selection
    if cortype=='1Bin':
        tag = sys
        if sys!='':             # if syst variation, creat new weight coulmn
            data['weight_'+sys] = data['weight_nominal']
        else:
            tag = 'nominal'

        sample = data['process']
        nTracks = data['fs_had_tau'+which_tau_cor+'_nTrack']
        pt = data['fs_had_tau'+which_tau_cor+'_pt']
        nbjets = data['m_nbjets']
        pdg = data['fs_had_tau'+which_tau+'_true_pdg']

        ############################################################################
        ####                    1-Tack
        ############################################################################
        ## 1b+2b / 1-Tack / pt1
        mask = (sample!='data') & (nTracks==1) & (pdg==0) & (pt>=20) & (pt<30)
        normfactor = 3.624 + is_tau_norm*1.211
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b+2b / 1-Tack / pt2
        mask = (sample!='data') & (nTracks==1) & (pdg==0) & (pt>=30) & (pt<40)
        normfactor = 1.664 + is_tau_norm*0.960
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b+2b / 1-Tack / pt3
        mask = (sample!='data') & (nTracks==1) & (pdg==0) & (pt>=40)
        normfactor = 1.557 + is_tau_norm*0.812 
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        ############################################################################
        ####                    3-Tack
        ############################################################################
        ## 1b+2b / 1-Tack / pt1
        mask = (sample!='data') & (nTracks==3) & (pdg==0) & (pt>=20) & (pt<30)
        normfactor = 1.117 + is_tau_norm*0.668
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b+2b / 1-Tack / pt2
        mask = (sample!='data') & (nTracks==3) & (pdg==0) & (pt>=30) & (pt<40)
        normfactor = 3.203 + is_tau_norm*2.283
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b+2b / 1-Tack / pt3
        mask = (sample!='data') & (nTracks==3) & (pdg==0) & (pt>=40)
        normfactor = 1.0 + is_tau_norm*1.0  # max(0.196 + is_tau_norm*1.402, 0)
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        # Previous values before April 2023
        # ############################################################################
        # ####                    1-Tack
        # ############################################################################
        # ## 1b+2b / 1-Tack / pt1
        # mask = (sample!='data') & (nTracks==1) & (pdg==0) & (pt>=20) & (pt<30)
        # normfactor = 2.903 + is_tau_norm*0.804
        # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        # ## 1b+2b / 1-Tack / pt2
        # mask = (sample!='data') & (nTracks==1) & (pdg==0) & (pt>=30) & (pt<40)
        # normfactor = 2.695 + is_tau_norm*1.093
        # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        # ## 1b+2b / 1-Tack / pt3
        # mask = (sample!='data') & (nTracks==1) & (pdg==0) & (pt>=40)
        # normfactor = 1.427 + is_tau_norm*0.697 
        # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        # ############################################################################
        # ####                    3-Tack
        # ############################################################################
        # ## 1b+2b / 1-Tack / pt1
        # mask = (sample!='data') & (nTracks==3) & (pdg==0) & (pt>=20) & (pt<30)
        # normfactor = 1.234 + is_tau_norm*0.558
        # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        # ## 1b+2b / 1-Tack / pt2
        # mask = (sample!='data') & (nTracks==3) & (pdg==0) & (pt>=30) & (pt<40)
        # normfactor = 1.659 + is_tau_norm*0.960
        # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        # ## 1b+2b / 1-Tack / pt3
        # mask = (sample!='data') & (nTracks==3) & (pdg==0) & (pt>=40)
        # normfactor = 1.0 + is_tau_norm*1.0  # max(0.066 + is_tau_norm*1.000, 0)
        # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    return data


def BKGCorrector_DiTau(data, sys):

    if (sys==''):
        logging.info(style.YELLOW+"Correcting ditau background templates derived with Template Fit."+style.RESET)
    else:
        logging.info(style.YELLOW+"Computing weight for ditau background templates derived with Template Fit ("+str(sys)+")."+style.RESET)

    # Init syst flags
    is_ditau_simjet = 0
    is_ditau_jetsim = 0
    is_lep_sys      = 0


    if (sys=="ditau_simjet_up"): is_ditau_simjet = 1
    elif (sys=="ditau_simjet_down"): is_ditau_simjet = -1
    elif (sys=="ditau_jetsim_up"): is_ditau_jetsim = 1
    elif (sys=="ditau_jetsim_down"): is_ditau_jetsim = -1

    elif (sys=="lep_sys_up"): is_lep_sys = 1
    elif (sys=="lep_sys_down"): is_lep_sys = -1

    # Create a tag to differentiate the syst variantions
    tag = sys
    if sys!='':             # if syst variation, creat new weight column
        data['weight_'+sys] = data['weight_nominal']
    else:
        tag = 'nominal'

    sample = data['process']
    nbjets = data['m_nbjets']
    nTracks_tau1 = data['fs_had_tau_1_nTrack']
    nTracks_tau2 = data['fs_had_tau_2_nTrack']
    TempCombinations = data['TempCombinations']

    ############################################################################
    ####                    1-prong, 1-prong
    ############################################################################
    # 1b / 1-track,1-track / sim,jet
    mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==1) & (TempCombinations=='sim,jet') & (nbjets==1)
    normfactor = 1.276 + is_ditau_simjet*0.213
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # 2b / 1-track,1-track / sim,jet
    mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==1) & (TempCombinations=='sim,jet') & (nbjets==2)
    normfactor = 1.183 + is_ditau_simjet*0.222
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # 1b / 1-track,1-track / jet,sim
    mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==1) & (TempCombinations=='jet,sim') & (nbjets==1)
    normfactor = 1.535 + is_ditau_jetsim*0.584
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # 2b / 1-track,1-track / jet,sim
    mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==1) & (TempCombinations=='jet,sim') & (nbjets==2)
    normfactor = 1.049 + is_ditau_jetsim*0.489
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # 1b / 1-track,1-track / jet,jet
    mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==1) & (TempCombinations=='jet,jet') & (nbjets==1)
    normfactor = 0.650 - (is_ditau_simjet+is_ditau_jetsim)*0.168/1.414
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # 2b / 1-track,1-track / jet,jet
    mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==1) & (TempCombinations=='jet,jet') & (nbjets==2)
    normfactor = 0.711 - (is_ditau_simjet+is_ditau_jetsim)*0.209/1.414
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    ############################################################################
    ####                    1-prong, 3-prong
    ############################################################################
    # 1b / 1-track,3-track / sim,jet
    mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==3) & (TempCombinations=='sim,jet') & (nbjets==1)
    normfactor = 1.578 + is_ditau_simjet*0.629
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # 2b / 1-track,3-track / sim,jet
    mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==3) & (TempCombinations=='sim,jet') & (nbjets==2)
    normfactor = 0.922 + is_ditau_simjet*0.242
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # 1b / 1-track,3-track / jet,sim
    mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==3) & (TempCombinations=='jet,sim') & (nbjets==1)
    normfactor = 4.365 + is_ditau_jetsim*2.765
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # 2b / 1-track,3-track / jet,sim
    mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==3) & (TempCombinations=='jet,sim') & (nbjets==2)
    normfactor = 0.117 + is_ditau_jetsim*0.489
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # 1b / 1-track,3-track / jet,jet
    mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==3) & (TempCombinations=='jet,jet') & (nbjets==1)
    normfactor = 0.116 - (is_ditau_simjet+is_ditau_jetsim)*0.630/1.414
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # 2b / 1-track,3-track / jet,jet
    mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==3) & (TempCombinations=='jet,jet') & (nbjets==2)
    normfactor = 1.109 - (is_ditau_simjet+is_ditau_jetsim)*0.245/1.414
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    ############################################################################
    ####                    3-prong, 1-prong
    ############################################################################
    # 1b / 3-track,1-track / sim,jet
    mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==1) & (TempCombinations=='sim,jet') & (nbjets==1)
    normfactor = 1.025 + is_ditau_simjet*0.292
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # 2b / 3-track,1-track / sim,jet
    mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==1) & (TempCombinations=='sim,jet') & (nbjets==2)
    normfactor = 1.575 + is_ditau_simjet*0.426
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # 1b / 3-track,1-track / jet,sim
    mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==1) & (TempCombinations=='jet,sim') & (nbjets==1)
    normfactor = 1.734 + is_ditau_jetsim*0.447
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # 2b / 3-track,1-track / jet,sim
    mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==1) & (TempCombinations=='jet,sim') & (nbjets==2)
    normfactor = 1.929 + is_ditau_jetsim*0.689
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # 1b / 3-track,1-track / jet,jet
    mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==1) & (TempCombinations=='jet,jet') & (nbjets==1)
    normfactor = 0.737 - (is_ditau_simjet+is_ditau_jetsim)*0.158/1.414
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # 2b / 3-track,1-track / jet,jet
    mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==1) & (TempCombinations=='jet,jet') & (nbjets==2)
    normfactor = 0.576 - (is_ditau_simjet+is_ditau_jetsim)*0.348/1.414
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    ############################################################################
    ####                    3-prong, 3-prong
    ############################################################################
    # 1b / 3-track,3-track / sim,jet
    mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==3) & (TempCombinations=='sim,jet') & (nbjets==1)
    normfactor = 0.637 + is_ditau_simjet*0.378
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # 2b / 3-track,3-track / sim,jet
    mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==3) & (TempCombinations=='sim,jet') & (nbjets==2)
    normfactor = 0.759 + is_ditau_simjet*0.372
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # 1b / 3-track,3-track / jet,sim
    mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==3) & (TempCombinations=='jet,sim') & (nbjets==1)
    normfactor = 2.924 + is_ditau_jetsim*1.515
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # 2b / 3-track,3-track / jet,sim
    mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==3) & (TempCombinations=='jet,sim') & (nbjets==2)
    normfactor = 1.658 + is_ditau_jetsim*1.279
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # 1b / 3-track,3-track / jet,jet
    mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==3) & (TempCombinations=='jet,jet') & (nbjets==1)
    normfactor = 0.848 - (is_ditau_simjet+is_ditau_jetsim)*0.272/1.414
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # 2b / 3-track,3-track / jet,jet
    mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==3) & (TempCombinations=='jet,jet') & (nbjets==2)
    normfactor = 1.227 - (is_ditau_simjet+is_ditau_jetsim)*0.370/1.414
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    ######  Previous values before April 2023
    # ############################################################################
    # ####                    1-prong, 1-prong
    # ############################################################################
    # # 1b / 1-track,1-track / sim,jet
    # mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==1) & (TempCombinations=='sim,jet') & (nbjets==1)
    # normfactor = 1.266 + is_ditau_simjet*0.154
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # # 2b / 1-track,1-track / sim,jet
    # mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==1) & (TempCombinations=='sim,jet') & (nbjets==2)
    # normfactor = 1.253 + is_ditau_simjet*0.228
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # # 1b / 1-track,1-track / jet,sim
    # mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==1) & (TempCombinations=='jet,sim') & (nbjets==1)
    # normfactor = 1.846 + is_ditau_jetsim*0.305
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # # 2b / 1-track,1-track / jet,sim
    # mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==1) & (TempCombinations=='jet,sim') & (nbjets==2)
    # normfactor = 1.107 + is_ditau_jetsim*0.520
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # # 1b / 1-track,1-track / jet,jet
    # mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==1) & (TempCombinations=='jet,jet') & (nbjets==1)
    # normfactor = 0.667 - (is_ditau_simjet+is_ditau_jetsim)*0.122/1.414
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # # 2b / 1-track,1-track / jet,jet
    # mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==1) & (TempCombinations=='jet,jet') & (nbjets==2)
    # normfactor = 0.641 - (is_ditau_simjet+is_ditau_jetsim)*0.219/1.414
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # ############################################################################
    # ####                    1-prong, 3-prong
    # ############################################################################
    # # 1b / 1-track,3-track / sim,jet
    # mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==3) & (TempCombinations=='sim,jet') & (nbjets==1)
    # normfactor = 1.031 + is_ditau_simjet*0.471
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # # 2b / 1-track,3-track / sim,jet
    # mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==3) & (TempCombinations=='sim,jet') & (nbjets==2)
    # normfactor = 0.893 + is_ditau_simjet*0.240
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # # 1b / 1-track,3-track / jet,sim
    # mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==3) & (TempCombinations=='jet,sim') & (nbjets==1)
    # normfactor = 2.150 + is_ditau_jetsim*1.267
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # # 2b / 1-track,3-track / jet,sim
    # mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==3) & (TempCombinations=='jet,sim') & (nbjets==2)
    # normfactor = 0.135 + is_ditau_jetsim*0.491
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # # 1b / 1-track,3-track / jet,jet
    # mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==3) & (TempCombinations=='jet,jet') & (nbjets==1)
    # normfactor = 0.976 - (is_ditau_simjet+is_ditau_jetsim)*0.364/1.414
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # # 2b / 1-track,3-track / jet,jet
    # mask = (sample!='data') & (nTracks_tau1==1) & (nTracks_tau2==3) & (TempCombinations=='jet,jet') & (nbjets==2)
    # normfactor = 1.184 - (is_ditau_simjet+is_ditau_jetsim)*0.255/1.414
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # ############################################################################
    # ####                    3-prong, 1-prong
    # ############################################################################
    # # 1b / 3-track,1-track / sim,jet
    # mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==1) & (TempCombinations=='sim,jet') & (nbjets==1)
    # normfactor = 0.941 + is_ditau_simjet*0.194
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # # 2b / 3-track,1-track / sim,jet
    # mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==1) & (TempCombinations=='sim,jet') & (nbjets==2)
    # normfactor = 1.554 + is_ditau_simjet*0.406
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # # 1b / 3-track,1-track / jet,sim
    # mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==1) & (TempCombinations=='jet,sim') & (nbjets==1)
    # normfactor = 0.798 + is_ditau_jetsim*0.598
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # # 2b / 3-track,1-track / jet,sim
    # mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==1) & (TempCombinations=='jet,sim') & (nbjets==2)
    # normfactor = 1.958 + is_ditau_jetsim*0.598
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # # 1b / 3-track,1-track / jet,jet
    # mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==1) & (TempCombinations=='jet,jet') & (nbjets==1)
    # normfactor = 1.360 - (is_ditau_simjet+is_ditau_jetsim)*0.264/1.414
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # # 2b / 3-track,1-track / jet,jet
    # mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==1) & (TempCombinations=='jet,jet') & (nbjets==2)
    # normfactor = 0.616 - (is_ditau_simjet+is_ditau_jetsim)*0.323/1.414
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # ############################################################################
    # ####                    3-prong, 3-prong
    # ############################################################################
    # # 1b / 3-track,3-track / sim,jet
    # mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==3) & (TempCombinations=='sim,jet') & (nbjets==1)
    # normfactor = 1.187 + is_ditau_simjet*0.297
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # # 2b / 3-track,3-track / sim,jet
    # mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==3) & (TempCombinations=='sim,jet') & (nbjets==2)
    # normfactor = 0.772 + is_ditau_simjet*0.417
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # # 1b / 3-track,3-track / jet,sim
    # mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==3) & (TempCombinations=='jet,sim') & (nbjets==1)
    # normfactor = 1.540 + is_ditau_jetsim*0.338
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # # 2b / 3-track,3-track / jet,sim
    # mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==3) & (TempCombinations=='jet,sim') & (nbjets==2)
    # normfactor = 1.615 + is_ditau_jetsim*1.271
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # # 1b / 3-track,3-track / jet,jet
    # mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==3) & (TempCombinations=='jet,jet') & (nbjets==1)
    # normfactor = 0.681 - (is_ditau_simjet+is_ditau_jetsim)*0.136/1.414
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
    # # 2b / 3-track,3-track / jet,jet
    # mask = (sample!='data') & (nTracks_tau1==3) & (nTracks_tau2==3) & (TempCombinations=='jet,jet') & (nbjets==2)
    # normfactor = 1.317 - (is_ditau_simjet+is_ditau_jetsim)*0.391/1.414
    # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    return data

######################################################################
###    Correct the single lepton fakes (lepditau channel)          ###
######################################################################
def BKGCorrector_Lep(data, temp_var, sys):
    logging.info(style.YELLOW+"Correcting jet faking a lepton."+style.RESET)

    # Init syst flags
    is_lep_syst = 0
    if (sys=="lep_up"): is_lep_syst = 1
    elif (sys=="lep_down"): is_lep_syst = -1

    # Create a tag to differentiate the syst variantions
    tag = sys
    if sys!='':             # if syst variation, creat new weight column
        data['weight_'+sys] = data['weight_nominal']
    else:
        tag = 'nominal'

    sample = data['process']
    template = data[temp_var]  # 'lep_1_tmp'
    TruthIFF_Class_lep1 = data['TruthIFF_Class_lep1']
    nbjets = data['m_nbjets']

    # 1b 
    mask = (sample!='data') & (nbjets==1) & (template=='jet')
    #normfactor = 1.396 + is_lep_syst*0.310 (before April 2023)
    normfactor = 1.436 + is_lep_syst*0.281
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # 2b 
    mask = (sample!='data') & (nbjets==2) & (template=='jet')
    #normfactor = 3.956 + is_lep_syst*2.034 (before April 2023)
    normfactor = 3.429 + is_lep_syst*1.902
    data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    # Summary print out
    if sys=='':
        logging.info(6*' '+'Template | NormFactor applied for 1b')
        logging.info(6*' '+' jet  |  %4.3f pm %4.3f' % (1.436, 0.281))
        logging.info(6*' '+'Template | NormFactor applied for 2b')
        logging.info(6*' '+' jet  |  %4.3f pm %4.3f' % (3.429, 1.902))

    return data

######################################################################
#####          Correct the dileptons fakes with syst             #####
######################################################################
def BKGCorrector_DiLep(data, templator, dict_tmps_comb, sys, SS_Selection=False):

    ##### Hard-coded SFs
    # OS 1b
    fbkg_vals_1b = [1.0, 1.087, 1.261, 14.166]
    fbkg_errs_1b = [0.0, 0.017, 0.106, 3.081]
    # OS 2b
    fbkg_vals_2b = [1.0, 1.605, 1.0, 1.0]
    fbkg_errs_2b = [0.0, 0.092, 1.0, 1.0]
    # Estimated simjet SF = 1.605 ± 0.092
    # Estimated jetsim SF = 0.146 ± 1.353
    # Estimated jetjet SF = 18.969 ± 39.389

    ###### Before April 2023
    # # OS 1b
    # fbkg_vals_1b = [1.0, 0.981, 1.143, 10.612]
    # fbkg_errs_1b = [0.0, 0.016, 0.098, 2.707]
    # # OS 2b
    # fbkg_vals_2b = [1.0, 1.676, 1.0, 1.0]
    # fbkg_errs_2b = [0.0, 0.071, 1.0, 1.0]

    if SS_Selection:
        fbkg_vals_1b = [1.0, 0.635, 0.520, 6.420]
        fbkg_errs_1b = [0.0, 0.008, 0.042, 3.353]

        fbkg_vals_2b = [1.0, 0.627, 0.940, 1.000]
        fbkg_errs_2b = [0.0, 0.042, 0.231, 1.000]

        ##### Older version (April 2023)
        # fbkg_vals_1b = [1.0, 0.612, 0.527, 4.693]
        # fbkg_errs_1b = [0.0, 0.008, 0.041, 2.623]

        # fbkg_vals_2b = [1.0, 0.612, 0.527, 4.693]
        # fbkg_errs_2b = [0.0, 0.008, 0.041, 2.623]


    if (sys==''):
        logging.info(style.YELLOW+"Correcting dilepton background templates derived with Template Fit."+style.RESET)
        logging.info("The nominal weight '{}' is modified with the following scale-factors:".format('weight_nominal'))
    else:
        logging.info(style.YELLOW+"Computing weight for dilepton background templates ("+str('weight_'+sys)+")."+style.RESET)

    # Init syst flags
    is_dilep_simjet = 0
    is_dilep_jetsim = 0

    if (sys=="dilep_simjet_up"): is_dilep_simjet = 1
    elif (sys=="dilep_simjet_down"): is_dilep_simjet = -1
    elif (sys=="dilep_jetsim_up"): is_dilep_jetsim = 1
    elif (sys=="dilep_jetsim_down"): is_dilep_jetsim = -1
    

    temp_list = [x for x in dict_tmps_comb['sample'] if 'data' not in x]
    if sys=='':
        logging.info(6*' '+'Template | NormFactor applied for 1b')
        for template, normfactor, normfactorerr in zip(temp_list, fbkg_vals_1b, fbkg_errs_1b):
            logging.info(6*' '+ template +  " | %4.3f pm %4.3f" % (normfactor,normfactorerr))
        logging.info(6*' '+'Template | NormFactor applied for 2b')
        for template, normfactor, normfactorerr in zip(temp_list, fbkg_vals_2b, fbkg_errs_2b):
            logging.info(6*' '+ template +  " | %4.3f pm %4.3f" % (normfactor,normfactorerr))


    # syst tag for weight variable
    tag = 'nominal'
    if sys!='':
        tag = sys
        data['weight_'+sys] = data['weight_nominal']

    # b-jet masks
    mask_1b = data['m_nbjets']==1
    mask_2b = data['m_nbjets']==2

    for template, normfactor, normfactorerr in zip(temp_list, fbkg_vals_1b, fbkg_errs_1b):
        mask = mask_1b & (data[templator]==template)
        data.loc[mask, 'weight_'+tag] = data['weight_nominal']*(normfactor + is_dilep_simjet*normfactorerr + is_dilep_jetsim*normfactorerr)

    for template, normfactor, normfactorerr in zip(temp_list, fbkg_vals_2b, fbkg_errs_2b):
        mask = mask_2b & (data[templator]==template)
        data.loc[mask, 'weight_'+tag] = data['weight_nominal']*(normfactor + is_dilep_simjet*normfactorerr + is_dilep_jetsim*normfactorerr)

    return data

######################################################################
###           Combined function for BKG correction                 ###
######################################################################
def correct_fakes(data_df, dict_tmps_comb, config):

    SystVarWeightTags=[]

    if config['AnalysisChannel']=='lephad':
        SystVarWeightTags_Tau, SystVarWeightTags_Dilep = [], []

        ### Jets faking tau
        if config['CORRECT_BKG_TAU']:
            # Produce weights for each systematic uncertainty (Nominal='' must be the last!)
            # Values are hard-coded in BKGCorrector_Tau
            # The corresponding weight name will be 'weight_'+systvar, or weight_nominal in case of nominal.
            if config['DilepSelection']=='OS':
                SystVarWeightTags_Tau = ['tau_shape_up', 'tau_shape_down',
                                        'tau_norm_up', 'tau_norm_down',
                                        '']
                for systtag in SystVarWeightTags_Tau:
                    data_df = BKGCorrector_Tau(data_df, 'QGMethod', '', systtag)

            if config['DilepSelection']=='SS' and not config['DilepSelection']=='OS':
                SystVarWeightTags_Tau = ['tau_norm_up', 'tau_norm_down',
                                         '']
                for systtag in SystVarWeightTags_Tau:
                    # If you want corrections from OS, use QGMethod method here
                    if config['CORRECT_BKG_TAU_METHOD'] == 'TemplateFit':
                        data_df = BKGCorrector_Tau(data_df, 'QGMethod', '', systtag)
                    elif config['CORRECT_BKG_TAU_METHOD'] == 'Counting':
                        data_df = BKGCorrector_Tau(data_df, '1Bin', '', systtag)
                    else:
                        break

        ### Jets faking one or both light leptons
        if config['CORRECT_BKG_DILEP']:
            SystVarWeightTags_Dilep = ['dilep_simjet_up', 'dilep_simjet_down',
                                       'dilep_jetsim_up', 'dilep_jetsim_down',
                                        '']
            for systtag in SystVarWeightTags_Dilep:
                is_SS_Selection = config['DilepSelection']=='SS'
                data_df = BKGCorrector_DiLep(data_df, 'TempCombinations', dict_tmps_comb, systtag, SS_Selection=is_SS_Selection)

        ### Combine weight tags
        if config['CORRECT_BKG_TAU']:
            SystVarWeightTags += SystVarWeightTags_Tau
        if config['CORRECT_BKG_DILEP']:
            if len(SystVarWeightTags):
                SystVarWeightTags = SystVarWeightTags[:-1]
            SystVarWeightTags += SystVarWeightTags_Dilep

    elif config['AnalysisChannel']=='hadhad':
        SystVarWeightTags_DiTau, SystVarWeightTags_Lep = [], []

        ### Jets faking tau pair
        if config['CORRECT_BKG_DITAU']:
            # Apply corrections measured in this channel?
            if config['USE_SF_FROM_OS']==False:
                SystVarWeightTags_DiTau = ['ditau_simjet_up', 'ditau_simjet_down',
                                           'ditau_jetsim_up', 'ditau_jetsim_down',
                                           '']

                for systtag in SystVarWeightTags_DiTau:
                    data_df = BKGCorrector_DiTau(data_df, systtag)
            # Apply corrections from OS channel?
            if config['USE_SF_FROM_OS']==True:
                SystVarWeightTags_DiTau = ['tau_shape_up', 'tau_shape_down',
                                        'tau_norm_up', 'tau_norm_down',
                                        '']
                # Leading tau
                for systtag in SystVarWeightTags_DiTau:
                    data_df = BKGCorrector_Tau(data_df, 'QGMethod', '', systtag)
                # Subleading tau
                for systtag in SystVarWeightTags_DiTau:
                    data_df = BKGCorrector_Tau(data_df, 'QGMethod', '_2', systtag)

        if config['CORRECT_BKG_LEP']:
            SystVarWeightTags_Lep = ['']
            for systtag in SystVarWeightTags_Lep:
                data_df = BKGCorrector_Lep(data_df, 'lep_1_tmp', systtag)

        ### Combine weight tags
        if config['CORRECT_BKG_DITAU']:
            SystVarWeightTags += SystVarWeightTags_DiTau
        ## TODO: complete variations for LEP

    return data_df, SystVarWeightTags


######################################################################
###  Check q-jets and g-jets fractions and correct if needed       ###
######################################################################
def qgratio(df, tau_number, gfracs=None):
    which_tau = {1: '', 2: '_2'}
    counter = 0
    for nb in [1,2]:
        for prong in [1,3]:
            for pT in [1,2,3]:
                pt_min_map = {1: 20, 2: 30, 3: 40}
                pt_max_map = {1: 30, 2: 40, 3: 2000}
                logging.info(style.CYAN+'q/g ratio within '+str(nb)+'b-jet, '+str(prong)+'-tracks, pT='+str(pt_min_map[pT])+'--'+str(pt_max_map[pT])+' GeV'+style.RESET)

                mask = (df['m_nbjets']==nb) & (df['fs_had_tau_'+str(tau_number)+'_nTrack']==prong)
                mask =  mask & (df['fs_had_tau_'+str(tau_number)+'_pt']>=pt_min_map[pT]) & (df['fs_had_tau_'+str(tau_number)+'_pt']<pt_max_map[pT])
                mask_qjets = mask & (df['tau_'+str(tau_number)+'_tmp']=='q-jet')
                mask_gjets = mask & (df['tau_'+str(tau_number)+'_tmp']=='g-jet')
                mask_bjets = mask & (df['tau_'+str(tau_number)+'_tmp']=='q-jet') & (df['fs_had_tau'+which_tau[tau_number]+'_true_partonTruthLabelID']==5)

                Nqjets = df[mask_qjets]['weight_nominal'].values.sum()
                Ngjets = df[mask_gjets]['weight_nominal'].values.sum()
                Nbjets = df[mask_bjets]['weight_nominal'].values.sum()

                logging.info('   Ngjets/Nqjets = {g:.3f}/{q:.3f}; G/(G+Q) = {gfrac:.3f}'.format(g=Ngjets,q=Nqjets, gfrac=Ngjets/(Nqjets+Ngjets)))
                logging.info('                                  b/(G+Q) = {:.3f}'.format(Nbjets/(Nqjets+Ngjets)))

                if gfracs:
                    df.loc[mask_gjets, 'weight_nominal'] = df[mask_gjets]['weight_nominal']*gfracs[counter]/(Ngjets/(Ngjets+Nqjets))
                    df.loc[mask_qjets, 'weight_nominal'] = df[mask_qjets]['weight_nominal']*(1-gfracs[counter])/(Nqjets/(Ngjets+Nqjets))
                    counter += 1

    return df

def qgratio_plot(df, tau_number, plt, isNorm=True, sample='all'):

    # Define a dictionary to map the partonTruthLabelID values to categories
    # See https://gitlab.cern.ch/atlasphys-top/singletop/SingleTopAnalysis/-/blob/master/Root/SingleTopEventSaver.cxx#L1050
    id_to_category = {1: 'light quarks', 2: 'light quarks', 3: 'light quarks', 4: 'light quarks', 5: 'b-quarks', 21: 'gluon jets', -99: 'No truthJetLink', -1: 'Unknown'}
    categories = ['light quarks', 'b-quarks', 'gluon jets', 'No truthJetLink', 'Unknown']

    # Definition for tau variabeles
    which_tau = {1: '', 2: '_2'}

    counter = 0
    for nb in [1,2]:
        for prong in [1,3]:
            for pT in [1,2,3]:
                pt_min_map = {1: 20, 2: 30, 3: 40}
                pt_max_map = {1: 30, 2: 40, 3: 2000}

                mask = (df['m_nbjets']==nb) & (df['fs_had_tau_'+str(tau_number)+'_nTrack']==prong)
                mask =  mask & (df['fs_had_tau_'+str(tau_number)+'_pt']>=pt_min_map[pT]) & (df['fs_had_tau_'+str(tau_number)+'_pt']<pt_max_map[pT])
                mask = mask & (df.process!='data')
                mask = mask & (df['fs_had_tau'+which_tau[tau_number]+'_true_partonTruthLabelID']>=-10000)
                if sample!='all':
                    mask = mask & (df.process==sample)

                # partonTruthLabelID = df[mask]['fs_had_tau'+which_tau[tau_number]+'_true_partonTruthLabelID'].values
                # RNNScore = df[mask]['fs_had_tau_'+str(tau_number)+'_RNNScore'].values
                # weights = df[mask]['weight_nominal'].values
                # dpg = df[mask]['fs_had_tau'+which_tau[tau_number]+'_true_pdg'].values

                # print(df[mask]['process'].values)

                # plt.hist2d(RNNScore, partonTruthLabelID, weights=weights,
                #            #bins=(20, 21),
                #            bins=(np.arange(0, 1.05, 0.05), np.arange(22)+0.5),
                #            density=True,
                #            cmap=plt.cm.BuPu)
                # plt.colorbar()
                # plt.xlabel('fs_had_tau_'+str(tau_number)+'_RNNScore')
                # plt.ylabel('fs_had_tau'+which_tau[tau_number]+'_true_partonTruthLabelID')
                # plt.savefig('QGplots/qgcomp_'+str(nb)+'b_'+str(prong)+'p_'+str(pT)+'pT_'+str(tau_number)+'.png')
                # plt.clf()

                ###### Stacked histogram
                # Filter out the rows where the partonTruthLabelID is not one of the categories we care about
                dfnew = df[mask].copy()
                dfnew = dfnew[dfnew['fs_had_tau'+which_tau[tau_number]+'_true_partonTruthLabelID'].isin(id_to_category.keys())]

                # Create a new column in the dataframe to hold the category name for each row
                dfnew['category'] = dfnew['fs_had_tau'+which_tau[tau_number]+'_true_partonTruthLabelID'].map(id_to_category)

                # Normalize the weights
                total_weight = dfnew['weight_nominal'].sum()
                dfnew['normalized_weight'] = dfnew['weight_nominal'] / total_weight

                # Create a list of dataframes, one for each category
                category_dfs = [dfnew[dfnew['category'] == category] for category in categories]

                if isNorm:
                    bin_edges = np.arange(0, 1.05, 0.05)

                    for ibin in range(len(bin_edges)-1):
                        bin_total_value = 0
                        for category_df in category_dfs:
                            mask_bin = (category_df['fs_had_tau_'+str(tau_number)+'_RNNScore']>=bin_edges[ibin]) & (category_df['fs_had_tau_'+str(tau_number)+'_RNNScore']<bin_edges[ibin+1])
                            bin_total_value += category_df[mask_bin]['normalized_weight'].sum()

                        # Apply weight
                        for category_df in category_dfs:
                            mask_bin = (category_df['fs_had_tau_'+str(tau_number)+'_RNNScore']>=bin_edges[ibin]) & (category_df['fs_had_tau_'+str(tau_number)+'_RNNScore']<bin_edges[ibin+1])
                            if bin_total_value:
                                category_df.loc[mask_bin, 'normalized_weight'] = category_df[mask_bin]['normalized_weight']/bin_total_value


                # Plot the stacked histogram
                plt.hist([category_df['fs_had_tau_'+str(tau_number)+'_RNNScore'] for category_df in category_dfs],
                    bins=np.arange(0, 1.05, 0.05),
                    weights=[category_df['normalized_weight'] for category_df in category_dfs],
                    stacked=True,
                    label=categories)

                # Add labels and a legend
                plt.xlabel('fs_had_tau_'+str(tau_number)+'_RNNScore')
                plt.ylabel('Normalized count')
                plt.legend()

                xcut = 0.25 if prong==1 else 0.40
                plt.axvline(x=xcut, color='blue', linestyle='--')

                plt.savefig('QGplots/qgprof_'+str(nb)+'b_'+str(prong)+'p_'+str(pT)+'pT_'+str(tau_number)+'.png')
                plt.clf()