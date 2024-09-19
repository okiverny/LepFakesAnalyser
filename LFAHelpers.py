import numpy as np
import pandas as pd
import glob
import os,sys
import logging
import ROOT
import ctypes
from array import array
import math
import itertools
import matplotlib.pyplot as plt
import errno
import multiprocessing


# Class of different styles
class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    LINE_UP = '\x1b[1A'
    LINE_CLEAR = '\x1b[2K'

def ensure_dir(path):
    try:
      os.makedirs(path)
    except OSError as exc: # Python >2.5
      if exc.errno == errno.EEXIST:
        pass
      else: raise

################################################################################
############################## Load data  ######################################

def load_data_RDF(input_folder, dict_samples, list_of_branches_leptons, list_of_branches_mc, tree_name='tHqLoop_nominal_Loose'):

    logging.info("======================= R e a d i n g  D a t a ======================= ")
    logging.info("Loading data from "+style.YELLOW+input_folder+style.RESET)

    ROOT.ROOT.EnableImplicitMT()

    # for cross-check
    ntuplefiles = glob.glob(input_folder + '*.root')

    output_df = None
    dframes = []
    for iproc, process in enumerate(dict_samples['sample']):
        is_MC = True
        branches = list_of_branches_leptons+list_of_branches_mc
        if process == 'data':
            is_MC = False
            branches = list_of_branches_leptons

        logging.info(style.YELLOW + 'Reading ' + process + ':' + style.RESET)

        # Constructing file paths
        fname_keys = dict_samples['fname_keys'][iproc]
        files_glob = [input_folder+file for file in fname_keys]

        # Reading files
        msg = '     '
        for index,file_glob in enumerate(files_glob):

            # Check missing files
            if not glob.glob(file_glob):
                #print('WARNING! Did not find any file with the key:', fname_keys[index])
                logging.warning('WARNING! Did not find any file with the key: ' + fname_keys[index])
                continue
            
            if index>0: msg += ', '
            msg += fname_keys[index]
            #print(msg)

            rdf = ROOT.RDataFrame(tree_name, file_glob)
            #rdf_local = (
            #    rdf.Define("sample_Id", str(iproc))
            #       .Define("process", 'std::string("'+process+'")')
            #)

            # Convert to pandas
            data = rdf.AsNumpy(columns=branches)
            data_df = pd.DataFrame(data, columns=branches, dtype=np.float32)

            # Additional columns
            data_df['sample_Id'] = iproc
            data_df['process'] = process

            dframes.append(data_df)
            #print(style.LINE_UP, end=style.LINE_CLEAR)
            
        #sys.stdout.write("\033[F") # Cursor up one line
        logging.info(msg)

    # Concatinating
    output_df = pd.concat(dframes, ignore_index=True)

    return output_df


def load_data_rootnp(input_folder, dict_samples, list_of_branches_leptons, list_of_branches_mc, tree_name='tHqLoop_nominal_Loose'):
    print(style.YELLOW+"Loading data from "+input_folder+style.RESET)

    try:
        import root_numpy as rootnp
    except ImportError:
        print("Cannot import root_numpy")


    # for cross-check
    ntuplefiles = glob.glob(input_folder + '*.root')

    output_df = None
    dframes = []
    for iproc, process in enumerate(dict_samples['sample']):
        is_MC = True
        if process == 'data':
            is_MC = False
        print('Reading', process,'...')
        fname_keys = dict_samples['fname_keys'][iproc]
        for file_key in fname_keys:
            if not glob.glob(input_folder + file_key):
                print('WARNING! Did not find any file with the key:', file_key)
                continue

            # Loading file by file
            for file in glob.glob(input_folder + file_key):
                if not file:
                    print('Missing input files with', file_key, 'key')
                    continue

                # Read file
                f = ROOT.TFile(file)
                tree = f.Get(tree_name)
                data, data_df = None, None
                if is_MC:
                    data = rootnp.tree2array(tree, branches=list_of_branches_leptons+list_of_branches_mc, selection="", start=0, stop=None)
                    data_df = pd.DataFrame(data, columns=list_of_branches_leptons+list_of_branches_mc, dtype=np.float32)
                else:
                    data = rootnp.tree2array(tree, branches=list_of_branches_leptons, selection="", start=0, stop=None)
                    data_df = pd.DataFrame(data, columns=list_of_branches_leptons, dtype=np.float32)
                
                # Additional columns
                data_df['sample_Id'] = iproc
                data_df['process'] = process
                # Consistency check 
                if file in ntuplefiles:
                    ntuplefiles.remove(file)

                dframes.append(data_df)
        #print(len(dframes))

    output_df = pd.concat(dframes, ignore_index=True)
    
    # Check if something is not in dictionary
    print(style.YELLOW+"Control Check "+style.RESET)
    for file in ntuplefiles:
        print('WARNING! not in dictionary but in the input dir:', file)

    print(style.CYAN+"Done! Loaded "+str(len(dframes))+" files ( "+str(output_df.shape[0])+" events )."+style.RESET)

    return output_df

################################################################################
############################## ANNOTATORS ######################################

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
    logging.info(style.YELLOW+"Labeling light lepton and tau combinations."+style.RESET)
    logging.info("      Input variables: {vars}' ".format(vars = templators))

    data['TempCombinations'] = data[templators].agg(lambda row: ','.join(row), axis=1)
    
    data['TempCombinationsEncoded'] = data['TempCombinations'].factorize()[0]
    # Another method
    #data['TemplateCombinations'] = data.groupby(templators, sort=False).ngroup()

    logging.info("      Output variable: '{v}' = {t}".format(v = 'TempCombinations', t=data['TempCombinations'].unique().tolist()))
    logging.info("      Output variable: '{v}' = {t}".format(v = 'TempCombinationsEncoded', t=data['TempCombinationsEncoded'].unique().tolist()))

    return data

def RegionsAnnotator(data, masks, region_names):
    logging.info(style.YELLOW+"Annotating regions."+style.RESET)
    data['regions_encoded'] = 0
    data['regions'] = ''
    for i, mask in enumerate(masks):
        data.loc[mask, 'regions_encoded'] = i+1
        data.loc[mask, 'regions'] = region_names[i]
    return data

####################################################################################
############################## BKG CORRECTORS ######################################

def BKGCorrector(data, templator, dict_tmps_comb, fbkg_vals, fbkg_errs):
    logging.info(style.YELLOW+"Correcting background templates with Template Fit Method."+style.RESET)
    temp_list = [x for x in dict_tmps_comb['sample'] if 'data' not in x]
    print('Template','NormFactor applied', sep=' | ')
    for template, normfactor, normfactorerr in zip(temp_list, fbkg_vals, fbkg_errs):
        print(template +  " | %4.3f pm %4.3f" % (normfactor,normfactorerr))

    def compute_weight(row):  #, templator, temp_list
        for template, normfactor in zip(temp_list, fbkg_vals):
            if template == row[templator]:
                return row['weight_nominal']*normfactor
        return row['weight_nominal']*1.
    data['weight_nominal'] = data.apply(compute_weight, axis=1)

    def compute_weight_error(row):
        for template, normfactor, normfactorerr in zip(temp_list, fbkg_vals, fbkg_errs):
            if template == row[templator]:
                return row['weight_nominal']*(normfactor + normfactorerr)
        return row['weight_nominal']*1.
    data['weight_nominal_TF_AllUp'] = data.apply(compute_weight_error, axis=1)

    return data

def BKGCorrector_DiLepSyst(data, templator, dict_tmps_comb, fbkg_vals, fbkg_errs, sys):

    if (sys==''):
        logging.info(style.YELLOW+"Correcting dilepton background templates derived with Template Fit."+style.RESET)
    else:
        logging.info(style.YELLOW+"Computing weight for dilepton background templates ("+str(sys)+")."+style.RESET)

    # Init syst flags
    is_dilep_simjet = 0
    is_dilep_jetsim = 0

    if (sys=="dilep_simjet_up"): is_dilep_simjet = 1
    elif (sys=="dilep_simjet_down"): is_dilep_simjet = -1
    elif (sys=="dilep_jetsim_up"): is_dilep_jetsim = 1
    elif (sys=="dilep_jetsim_down"): is_dilep_jetsim = -1
    

    temp_list = [x for x in dict_tmps_comb['sample'] if 'data' not in x]
    if sys=='':
        print('Template','NormFactor applied', sep=' | ')
        for template, normfactor, normfactorerr in zip(temp_list, fbkg_vals, fbkg_errs):
            print(template +  " | %4.3f pm %4.3f" % (normfactor,normfactorerr))

    def compute_weight(row):
        nbjets = row['m_nbjets']

        for template, normfactor, normfactorerr in zip(temp_list, fbkg_vals, fbkg_errs):
            if template == row[templator]:
                return row['weight_nominal']*(normfactor + is_dilep_simjet*normfactorerr + is_dilep_jetsim*normfactorerr)
        return row['weight_nominal']*1.

    if sys=='':
        data['weight_nominal'] = data.apply(compute_weight, axis=1)
    else:
        data['weight_'+sys] = data.apply(compute_weight, axis=1)

    return data

####################################################################################
############################## DILEP Correction with sys ###########################
def BKGCorrector_DiLepSyst_1b2b(data, templator, dict_tmps_comb, sys, SS_Selection=False):

    ##### Hard-coded SFs:
    # OS 1b
    fbkg_vals_1b = [1.0, 0.981, 1.143, 10.612]
    fbkg_errs_1b = [0.0, 0.016, 0.098, 2.707]
    # OS 2b
    fbkg_vals_2b = [1.0, 1.676, 1.0, 1.0]
    fbkg_errs_2b = [0.0, 0.071, 1.0, 1.0]

    if SS_Selection:
        fbkg_vals_1b = [1.0, 0.612, 0.527, 4.693]
        fbkg_errs_1b = [0.0, 0.008, 0.041, 2.623]

        fbkg_vals_2b = [1.0, 0.612, 0.527, 4.693]
        fbkg_errs_2b = [0.0, 0.008, 0.041, 2.623]


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
        logging.info('Template','NormFactor applied for 1b', sep=' | ')
        for template, normfactor, normfactorerr in zip(temp_list, fbkg_vals_1b, fbkg_errs_1b):
            logging.info(template +  " | %4.3f pm %4.3f" % (normfactor,normfactorerr))
        logging.info('Template','NormFactor applied for 2b', sep=' | ')
        for template, normfactor, normfactorerr in zip(temp_list, fbkg_vals_2b, fbkg_errs_2b):
            logging.info(template +  " | %4.3f pm %4.3f" % (normfactor,normfactorerr))

    def compute_weight(row):
        nbjets = row['m_nbjets']
        
        if nbjets==1:
            for template, normfactor, normfactorerr in zip(temp_list, fbkg_vals_1b, fbkg_errs_1b):
                if template == row[templator]:
                    return row['weight_nominal']*(normfactor + is_dilep_simjet*normfactorerr + is_dilep_jetsim*normfactorerr)
        if nbjets==2:
            for template, normfactor, normfactorerr in zip(temp_list, fbkg_vals_2b, fbkg_errs_2b):
                if template == row[templator]:
                    return row['weight_nominal']*(normfactor + is_dilep_simjet*normfactorerr + is_dilep_jetsim*normfactorerr)
        return row['weight_nominal']*1.0

    if sys=='':
        data['weight_nominal'] = data.apply(compute_weight, axis=1)
    else:
        data['weight_'+sys] = data.apply(compute_weight, axis=1)

    return data

def BKGCorrector_DiLepSyst_vectorized(data, templator, dict_tmps_comb, sys, SS_Selection=False):

    ##### Hard-coded SFs:
    # OS 1b
    fbkg_vals_1b = [1.0, 0.981, 1.143, 10.612]
    fbkg_errs_1b = [0.0, 0.016, 0.098, 2.707]
    # OS 2b
    fbkg_vals_2b = [1.0, 1.676, 1.0, 1.0]
    fbkg_errs_2b = [0.0, 0.071, 1.0, 1.0]

    if SS_Selection:
        fbkg_vals_1b = [1.0, 0.612, 0.527, 4.693]
        fbkg_errs_1b = [0.0, 0.008, 0.041, 2.623]

        fbkg_vals_2b = [1.0, 0.612, 0.527, 4.693]
        fbkg_errs_2b = [0.0, 0.008, 0.041, 2.623]


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
def BKGCorrector_DiTau(data, templator, dict_tmps_comb, fbkg_vals, fbkg_errs, tau1_nTracks, tau2_nTracks, nbjets):
    logging.info(style.YELLOW+"Correcting background templates for tau pairs."+style.RESET)

    temp_list = [x for x in dict_tmps_comb['sample'] if 'data' not in x]
    print('tau1 = prong'+str(tau1_nTracks)+'; tau2 = prong'+str(tau2_nTracks)+'; nbjets='+str(nbjets))
    print('Template','NormFactor', sep=' | ')
    for template, normfactor, normfactorerr in zip(temp_list, fbkg_vals, fbkg_errs):
        print(template +  " | %4.3f pm %4.3f" % (normfactor,normfactorerr))

    def compute_weight(row):  #, templator, temp_list
        for template, normfactor in zip(temp_list, fbkg_vals):
            if template == row[templator] and row['fs_had_tau_1_nTrack']==tau1_nTracks and row['fs_had_tau_2_nTrack']==tau2_nTracks and row['m_nbjets']==nbjets:
                return row['weight_nominal']*normfactor
        return row['weight_nominal']*1.
    data['weight_nominal'] = data.apply(compute_weight, axis=1)

    return data

######################################################################
#####  Correct for single tau fakes with syst (universal)
def BKGCorrector_DiTauSyst(data, sys):

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

    # Compute the tau background SF in the case of q/g jet split
    def compute_weight_ditau(row):
        sample = row['process']
        
        tau_1_nTracks = row['fs_had_tau_1_nTrack']
        tau_2_nTracks = row['fs_had_tau_2_nTrack']
    
        tau_1_partonTruthLabelID = row['fs_had_tau_true_partonTruthLabelID']
        tau_2_partonTruthLabelID = row['fs_had_tau_2_true_partonTruthLabelID']
        #pdg = row['fs_had_tau'+which_tau+'_true_pdg']
        nbjets = row['m_nbjets']
        TempCombinations=row['TempCombinations']

        normfactor=1
        if sample!='data':

            # 1-prong, 1-prong
            if tau_1_nTracks==1 and tau_2_nTracks==1:
                if TempCombinations=='sim,jet':
                #if tau_1_partonTruthLabelID<=0 and tau_2_partonTruthLabelID>0: # sim,jet
                    if (nbjets==1): normfactor = 1.266 + is_ditau_simjet*0.154
                    if (nbjets==2): normfactor = 1.253 + is_ditau_simjet*0.228
			    
                if TempCombinations=='jet,sim':
                #if tau_1_partonTruthLabelID>0 and tau_2_partonTruthLabelID<=0: # jet,sim
                    if (nbjets==1): normfactor = 1.846 + is_ditau_jetsim*0.305
                    if (nbjets==2): normfactor = 1.107 + is_ditau_jetsim*0.520

                if TempCombinations=='jet,jet':
                #if tau_1_partonTruthLabelID>0 and tau_2_partonTruthLabelID>0:  # jet,jet
                    if (nbjets==1): normfactor = 0.667 - (is_ditau_simjet+is_ditau_jetsim)*0.122/1.414
                    if (nbjets==2): normfactor = 0.641 - (is_ditau_simjet+is_ditau_jetsim)*0.219/1.414

            # 1-prong, 3-prong
            if tau_1_nTracks==1 and tau_2_nTracks==3:
                if TempCombinations=='sim,jet':
                #if tau_1_partonTruthLabelID<=0 and tau_2_partonTruthLabelID>0: # sim,jet
                    if (nbjets==1): normfactor = 1.031 + is_ditau_simjet*0.471
                    if (nbjets==2): normfactor = 0.893 + is_ditau_simjet*0.240
			    
                if TempCombinations=='jet,sim':
                #if tau_1_partonTruthLabelID>0 and tau_2_partonTruthLabelID<=0: # jet,sim
                    if (nbjets==1): normfactor = 2.150 + is_ditau_jetsim*1.267
                    if (nbjets==2): normfactor = 0.135 + is_ditau_jetsim*0.491

                if TempCombinations=='jet,jet':
                #if tau_1_partonTruthLabelID>0 and tau_2_partonTruthLabelID>0:  # jet,jet
                    if (nbjets==1): normfactor = 0.976 - (is_ditau_simjet+is_ditau_jetsim)*0.364/1.414
                    if (nbjets==2): normfactor = 1.184 - (is_ditau_simjet+is_ditau_jetsim)*0.255/1.414

            # 3-prong, 1-prong
            if tau_1_nTracks==3 and tau_2_nTracks==1:
                if TempCombinations=='sim,jet':
                #if tau_1_partonTruthLabelID<=0 and tau_2_partonTruthLabelID>0: # sim,jet
                    if (nbjets==1): normfactor = 0.941 + is_ditau_simjet*0.194
                    if (nbjets==2): normfactor = 1.554 + is_ditau_simjet*0.406
			    
                if TempCombinations=='jet,sim':
                #if tau_1_partonTruthLabelID>0 and tau_2_partonTruthLabelID<=0: # jet,sim
                    if (nbjets==1): normfactor = 0.798 + is_ditau_jetsim*0.598
                    if (nbjets==2): normfactor = 1.958 + is_ditau_jetsim*0.598

                if TempCombinations=='jet,jet':
                #if tau_1_partonTruthLabelID>0 and tau_2_partonTruthLabelID>0:  # jet,jet
                    if (nbjets==1): normfactor = 1.360 - (is_ditau_simjet+is_ditau_jetsim)*0.264/1.414
                    if (nbjets==2): normfactor = 0.616 - (is_ditau_simjet+is_ditau_jetsim)*0.323/1.414

            # 3-prong, 3-prong
            if tau_1_nTracks==3 and tau_2_nTracks==3:
                if TempCombinations=='sim,jet':
                #if tau_1_partonTruthLabelID<=0 and tau_2_partonTruthLabelID>0: # sim,jet
                    if (nbjets==1): normfactor = 1.187 + is_ditau_simjet*0.297
                    if (nbjets==2): normfactor = 0.772 + is_ditau_simjet*0.417
			    
                if TempCombinations=='jet,sim':
                #if tau_1_partonTruthLabelID>0 and tau_2_partonTruthLabelID<=0: # jet,sim
                    if (nbjets==1): normfactor = 1.540 + is_ditau_jetsim*0.338
                    if (nbjets==2): normfactor = 1.615 + is_ditau_jetsim*1.271

                if TempCombinations=='jet,jet':
                #if tau_1_partonTruthLabelID>0 and tau_2_partonTruthLabelID>0:  # jet,jet
                    if (nbjets==1): normfactor = 0.681 - (is_ditau_simjet+is_ditau_jetsim)*0.136/1.414
                    if (nbjets==2): normfactor = 1.317 - (is_ditau_simjet+is_ditau_jetsim)*0.391/1.414

        return row['weight_nominal']*normfactor

    # Correction
    if sys=='':
        data['weight_nominal'] = data.apply(compute_weight_ditau, axis=1)
    else:
        data['weight_'+sys] = data.apply(compute_weight_ditau, axis=1)


    return data

######################################################################
#####  Correct for single tau fakes
def BKGCorrector_Tau(data, cortype, which_tau):
    logging.info(style.YELLOW+"Correcting single tau background templates derived with Template Fit."+style.RESET)

    # Construct a tag for association of the tau with the corresponding histo name
    which_tau_cor=''
    if which_tau=='':
        which_tau_cor = '_1'
    else:
        which_tau_cor = which_tau

    # Compute the tau background SF in the case of q/g jet split
    def compute_weight_QGMethod(row):
        sample = row['process']
        
        nTracks = row['fs_had_tau'+which_tau_cor+'_nTrack']
        pt = row['fs_had_tau'+which_tau_cor+'_pt']
        partonTruthLabelID = row['fs_had_tau'+which_tau+'_true_partonTruthLabelID']
        pdg = row['fs_had_tau'+which_tau+'_true_pdg']

        normfactor=1
        if sample!='data':
            if nTracks==1:
                if partonTruthLabelID==21: ### --> gluons
                    if (pt>=20 and pt<30): normfactor=0.635
                    elif (pt>=30 and pt<40): normfactor=0.896
                    elif (pt>=40): normfactor=1.132
                elif partonTruthLabelID!=21 and partonTruthLabelID>0: ### --> quarks
                    if (pt>=20 and pt<30): normfactor=1.376
                    elif (pt>=30 and pt<40): normfactor=1.088
                    elif (pt>=40): normfactor=0.916
                elif partonTruthLabelID<0 and pdg==0: ### --> unknown
                    normfactor = 0.899*(pt>=20 and pt<30) + 0.858*(pt>=30 and pt<40) + 1.110*(pt>=40)
         
            if nTracks==3:
                if partonTruthLabelID==21:  ### --> gluons
                    if (pt>=20 and pt<30): normfactor=0.717
                    elif (pt>=30 and pt<40): normfactor=0.607
                    elif (pt>=40): normfactor=1.142
                elif partonTruthLabelID!=21 and partonTruthLabelID>0:  ### --> quarks
                    if (pt>=20 and pt<30): normfactor=1.195
                    elif (pt>=30 and pt<40): normfactor=1.073
                    elif (pt>=40): normfactor=0.828
                elif partonTruthLabelID<0 and pdg==0: ### --> unknown
                    normfactor = 0.940*(pt>=20 and pt<30) + 1.437*(pt>=30 and pt<40) + 2.153*(pt>=40)

        return row['weight_nominal']*normfactor

    def compute_weight_1Bin(row):
        sample = row['process']
        nTracks = row['fs_had_tau'+which_tau_cor+'_nTrack']
        pt = row['fs_had_tau'+which_tau_cor+'_pt']
        partonTruthLabelID = row['fs_had_tau'+which_tau+'_true_partonTruthLabelID']
        pdg = row['fs_had_tau'+which_tau+'_true_pdg']

        normfactor=1
        if sample!='data' and pdg==0:
            if nTracks==1:
                if (pt>=20 and pt<30): normfactor=1.144
                elif (pt>=30 and pt<40): normfactor=0.972
                elif (pt>=40): normfactor=0.909
            elif nTracks==3:
                if (pt>=20 and pt<30): normfactor=1.067
                elif (pt>=30 and pt<40): normfactor=0.964
                elif (pt>=40): normfactor=0.818

        return row['weight_nominal']*normfactor

    # Correction
    if cortype=='QGMethod':
        data['weight_nominal'] = data.apply(compute_weight_QGMethod, axis=1)
    elif cortype=='1Bin':
        data['weight_nominal'] = data.apply(compute_weight_1Bin, axis=1)

    return data

######################################################################
#####  Correct for single tau fakes with syst
def BKGCorrector_TauSyst(data, cortype, which_tau, sys):

    if (sys==''):
        logging.info(style.YELLOW+"Correcting single tau background templates derived with Template Fit."+style.RESET)
    else:
        logging.info(style.YELLOW+"Computing weight for single tau background templates derived with Template Fit ("+str(sys)+")."+style.RESET)

    # Init syst flags
    is_tau_shape = 0
    is_tau_norm = 0
    is_ditau_simjet = 0
    is_ditau_jetsim = 0
    is_lep_sys = 0
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

    # Compute the tau background SF in the case of q/g jet split
    def compute_weight_QGMethod(row):
        sample = row['process']
        
        nTracks = row['fs_had_tau'+which_tau_cor+'_nTrack']
        pt = row['fs_had_tau'+which_tau_cor+'_pt']
        partonTruthLabelID = row['fs_had_tau'+which_tau+'_true_partonTruthLabelID']
        pdg = row['fs_had_tau'+which_tau+'_true_pdg']
        nbjets = row['m_nbjets']

        normfactor=1
        if sample!='data':
            if nTracks==1:
                if partonTruthLabelID==21: ### --> gluons
                    if (pt>=20 and pt<30): normfactor = 0.635 - is_tau_shape*0.300 + is_tau_norm*0.049
                    elif (pt>=30 and pt<40): normfactor = 0.896 - is_tau_shape*0.437 + is_tau_norm*0.055
                    elif (pt>=40): normfactor=1.132 - is_tau_shape*0.907 + is_tau_norm*0.042
                elif partonTruthLabelID!=21 and partonTruthLabelID>0: ### --> quarks
                    if (pt>=20 and pt<30): normfactor=1.376 + is_tau_shape*0.109 + is_tau_norm*0.049
                    elif (pt>=30 and pt<40): normfactor=1.088 + is_tau_shape*0.098 + is_tau_norm*0.055
                    elif (pt>=40): normfactor=0.916 + is_tau_shape*0.101 + is_tau_norm*0.042
                elif partonTruthLabelID<0 and pdg==0: ### --> unknown
                    normfactor = 0.899*(pt>=20 and pt<30) + 0.858*(pt>=30 and pt<40) + 1.110*(pt>=40)
         
            if nTracks==3:
                if partonTruthLabelID==21:  ### --> gluons
                    if (pt>=20 and pt<30): normfactor=0.717 - is_tau_shape*0.290 + is_tau_norm*0.055
                    elif (pt>=30 and pt<40): normfactor=0.607 - is_tau_shape*0.521 + is_tau_norm*0.127
                    elif (pt>=40): normfactor=1.142 - is_tau_shape*0.617 + is_tau_norm*0.065
                elif partonTruthLabelID!=21 and partonTruthLabelID>0:  ### --> quarks
                    if (pt>=20 and pt<30): normfactor=1.195 + is_tau_shape*0.065 + is_tau_norm*0.055
                    elif (pt>=30 and pt<40): normfactor=1.073 + is_tau_shape*0.090 + is_tau_norm*0.127
                    elif (pt>=40): normfactor=0.828 + is_tau_shape*0.045 + is_tau_norm*0.065
                elif partonTruthLabelID<0 and pdg==0: ### --> unknown
                    normfactor = 0.940*(pt>=20 and pt<30) + 1.437*(pt>=30 and pt<40) + 2.153*(pt>=40)

        return row['weight_nominal']*normfactor

    def compute_weight_1Bin(row):
        sample = row['process']
        nTracks = row['fs_had_tau'+which_tau_cor+'_nTrack']
        pt = row['fs_had_tau'+which_tau_cor+'_pt']
        partonTruthLabelID = row['fs_had_tau'+which_tau+'_true_partonTruthLabelID']
        pdg = row['fs_had_tau'+which_tau+'_true_pdg']

        normfactor=1
        if sample!='data' and pdg==0:
            if nTracks==1:
                if (pt>=20 and pt<30): normfactor=1.144
                elif (pt>=30 and pt<40): normfactor=0.972
                elif (pt>=40): normfactor=0.909
            elif nTracks==3:
                if (pt>=20 and pt<30): normfactor=1.067
                elif (pt>=30 and pt<40): normfactor=0.964
                elif (pt>=40): normfactor=0.818

        return row['weight_nominal']*normfactor

    # Correction
    if cortype=='QGMethod':
        if sys=='':
            data['weight_nominal'] = data.apply(compute_weight_QGMethod, axis=1)
        else:
            data['weight_'+sys] = data.apply(compute_weight_QGMethod, axis=1)

    elif cortype=='1Bin':
        data['weight_nominal'] = data.apply(compute_weight_1Bin, axis=1)

    return data

######################################################################
#####  Method to correct only unknown template (for some studies)
def BKGCorrector_TauUnknown_OS_1b(data, which_tau):

    logging.info(style.YELLOW+"Correcting single tau UMKNOWN background template."+style.RESET)

    # Construct a tag for association of the tau with the corresponding histo name
    which_tau_cor=''
    if which_tau=='':
        which_tau_cor = '_1'
    else:
        which_tau_cor = which_tau

    # Compute the tau background SF in the case of q/g jet split
    def compute_weight_QGMethod(row):
        sample = row['process']
        
        nTracks = row['fs_had_tau'+which_tau_cor+'_nTrack']
        pt = row['fs_had_tau'+which_tau_cor+'_pt']
        partonTruthLabelID = row['fs_had_tau'+which_tau+'_true_partonTruthLabelID']
        pdg = row['fs_had_tau'+which_tau+'_true_pdg']

        normfactor=1
        if sample!='data':
            if nTracks==1:
                if partonTruthLabelID<0 and pdg==0: ### --> unknown
                    normfactor = 0.958*(pt>=20 and pt<30) + 0.848*(pt>=30 and pt<40) + 0.953*(pt>=40)
            if nTracks==3:
                if partonTruthLabelID<0 and pdg==0: ### --> unknown
                    normfactor = 0.925*(pt>=20 and pt<30) + 1.397*(pt>=30 and pt<40) + 2.696*(pt>=40)

        return row['weight_nominal']*normfactor

    # Correction
    data['weight_nominal'] = data.apply(compute_weight_QGMethod, axis=1)

    return data

######################################################################
#####  Correct for single tau fakes with syst (universal)
def BKGCorrector_TauSyst_OS_1b(data, cortype, which_tau, sys):

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

    # Compute the tau background SF in the case of q/g jet split
    def compute_weight_QGMethod(row):
        sample = row['process']
        
        nTracks = row['fs_had_tau'+which_tau_cor+'_nTrack']
        pt = row['fs_had_tau'+which_tau_cor+'_pt']
        partonTruthLabelID = row['fs_had_tau'+which_tau+'_true_partonTruthLabelID']
        pdg = row['fs_had_tau'+which_tau+'_true_pdg']
        nbjets = row['m_nbjets']

        normfactor=1
        if sample!='data':
            if nTracks==1 and nbjets==1:
                if partonTruthLabelID==21: ### --> gluons
                    if (pt>=20 and pt<30): normfactor = 0.611 - is_tau_shape*0.297 + is_tau_norm*0.050
                    elif (pt>=30 and pt<40): normfactor = 0.993 - is_tau_shape*0.473 + is_tau_norm*0.057
                    elif (pt>=40): normfactor=1.122 - is_tau_shape*0.977 + is_tau_norm*0.042
                elif partonTruthLabelID!=21 and partonTruthLabelID>0: ### --> quarks
                    if (pt>=20 and pt<30): normfactor=1.360 + is_tau_shape*0.106 + is_tau_norm*0.050
                    elif (pt>=30 and pt<40): normfactor=1.058 + is_tau_shape*0.103 + is_tau_norm*0.057
                    elif (pt>=40): normfactor=0.909 + is_tau_shape*0.106 + is_tau_norm*0.042
                elif partonTruthLabelID<0 and pdg==0: ### --> unknown
                    normfactor = 0.958*(pt>=20 and pt<30) + 0.848*(pt>=30 and pt<40) + 0.953*(pt>=40)
         
            if nTracks==3 and nbjets==1:
                if partonTruthLabelID==21:  ### --> gluons
                    if (pt>=20 and pt<30): normfactor=0.719 - is_tau_shape*0.439 + is_tau_norm*0.050
                    elif (pt>=30 and pt<40): normfactor=0.605 - is_tau_shape*0.455 + is_tau_norm*0.062
                    elif (pt>=40): normfactor=0.841 - is_tau_shape*0.784 + is_tau_norm*0.062
                elif partonTruthLabelID!=21 and partonTruthLabelID>0:  ### --> quarks
                    if (pt>=20 and pt<30): normfactor=1.159 + is_tau_shape*0.097 + is_tau_norm*0.050
                    elif (pt>=30 and pt<40): normfactor=0.932 + is_tau_shape*0.070 + is_tau_norm*0.062
                    elif (pt>=40): normfactor=0.897 + is_tau_shape*0.056 + is_tau_norm*0.062
                elif partonTruthLabelID<0 and pdg==0: ### --> unknown
                    normfactor = 0.925*(pt>=20 and pt<30) + 1.397*(pt>=30 and pt<40) + 2.696*(pt>=40)

            if nTracks==1 and nbjets==2:
                if partonTruthLabelID==21: ### --> gluons
                    if (pt>=20 and pt<30): normfactor = 1.747 - is_tau_shape*0.234 + is_tau_norm*0.059
                    elif (pt>=30 and pt<40): normfactor = 1.560 - is_tau_shape*0.596 + is_tau_norm*0.096
                    elif (pt>=40): normfactor=0.918 - is_tau_shape*0.680 + is_tau_norm*0.059
                elif partonTruthLabelID!=21 and partonTruthLabelID>0: ### --> quarks
                    if (pt>=20 and pt<30): normfactor=0.810 + is_tau_shape*0.102 + is_tau_norm*0.059
                    elif (pt>=30 and pt<40): normfactor=0.856 + is_tau_shape*0.157 + is_tau_norm*0.096
                    elif (pt>=40): normfactor=0.848 + is_tau_shape*0.081 + is_tau_norm*0.059
                elif partonTruthLabelID<0 and pdg==0: ### --> unknown
                    normfactor = 1.0*(pt>=20 and pt<30) + 1.0*(pt>=30 and pt<40) + 1.0*(pt>=40)

            if nTracks==3 and nbjets==2:
                if partonTruthLabelID==21:  ### --> gluons
                    if (pt>=20 and pt<30): normfactor=1.623 - is_tau_shape*0.331 + is_tau_norm*0.061
                    elif (pt>=30 and pt<40): normfactor=1.696 - is_tau_shape*0.556 + is_tau_norm*0.066
                    elif (pt>=40): normfactor=1.168 - is_tau_shape*1.168 + is_tau_norm*0.070
                elif partonTruthLabelID!=21 and partonTruthLabelID>0:  ### --> quarks
                    if (pt>=20 and pt<30): normfactor=0.996 + is_tau_shape*0.063 + is_tau_norm*0.061
                    elif (pt>=30 and pt<40): normfactor=0.743 + is_tau_shape*0.065 + is_tau_norm*0.066
                    elif (pt>=40): normfactor=0.897 + is_tau_shape*0.073 + is_tau_norm*0.070
                elif partonTruthLabelID<0 and pdg==0: ### --> unknown
                    normfactor = 1.0*(pt>=20 and pt<30) + 1.0*(pt>=30 and pt<40) + 1.0*(pt>=40)

        return row['weight_nominal']*normfactor

    def compute_weight_1Bin(row):
        sample = row['process']
        nTracks = row['fs_had_tau'+which_tau_cor+'_nTrack']
        pt = row['fs_had_tau'+which_tau_cor+'_pt']
        pdg = row['fs_had_tau'+which_tau+'_true_pdg']

        normfactor=1
        if sample!='data' and pdg==0:
            if nTracks==1:                                  # 1b+2b
                if (pt>=20 and pt<30): normfactor=2.903 + is_tau_norm*0.804    # 2.903 $\pm$ 0.804   (old: 1.888)
                elif (pt>=30 and pt<40): normfactor=2.695 + is_tau_norm*1.093  # 2.695 $\pm$ 1.093   (old: 1.647)
                elif (pt>=40): normfactor=1.427 + is_tau_norm*0.697            # 1.427 $\pm$ 0.697   (old: 1.127)
            elif nTracks==3:
                if (pt>=20 and pt<30): normfactor=1.234 + is_tau_norm*0.558    # 1.234 $\pm$ 0.558   (old: 0.872)
                elif (pt>=30 and pt<40): normfactor=1.659 + is_tau_norm*0.960  # 1.659 $\pm$ 0.960   (old: 1.358)
                elif (pt>=40): normfactor=0.066 + is_tau_norm*1.000            # 0.066 $\pm$ 1.000   (old: 0.836)

        return row['weight_nominal']*normfactor

    # Correction
    if cortype=='QGMethod':
        if sys=='':
            data['weight_nominal'] = data.apply(compute_weight_QGMethod, axis=1)
        else:
            data['weight_'+sys] = data.apply(compute_weight_QGMethod, axis=1)

    elif cortype=='1Bin':
        if sys=='':
            data['weight_nominal'] = data.apply(compute_weight_1Bin, axis=1)
        else:
            data['weight_'+sys] = data.apply(compute_weight_1Bin, axis=1)

    return data


######################################################################
#####  Correct for single tau fakes with syst in a vectorized way (universal)
def BKGCorrector_TauSyst_OS_vectorized(data, cortype, which_tau, sys):

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
            data['weight_'+sys] = data['weight_nominal']
        else:
            tag = 'nominal'

        sample = data['process']
        nTracks = data['fs_had_tau'+which_tau_cor+'_nTrack']
        pt = data['fs_had_tau'+which_tau_cor+'_pt']
        partonTruthLabelID = data['fs_had_tau'+which_tau+'_true_partonTruthLabelID']
        pdg = data['fs_had_tau'+which_tau+'_true_pdg']
        nbjets = data['m_nbjets']

        ############################################################################
        ####                    1b / 1-Tack
        ############################################################################
        ## 1b / 1-Tack / gluon-jet / pt1
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=20) & (pt<30)
        normfactor = 0.611 - is_tau_shape*0.297 + is_tau_norm*0.050
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b / 1-Tack / gluon-jet / pt2
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=30) & (pt<40)
        normfactor = 0.993 - is_tau_shape*0.473 + is_tau_norm*0.057
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b / 1-Tack / gluon-jet / pt3
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=40)
        normfactor = 1.122 - is_tau_shape*0.977 + is_tau_norm*0.042
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        ############################################################################
        ## 1b / 1-Tack / quark-jet / pt1
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=20) & (pt<30)
        normfactor = 1.360 + is_tau_shape*0.106 + is_tau_norm*0.050
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b / 1-Tack / quark-jet / pt2
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=30) & (pt<40)
        normfactor = 1.058 + is_tau_shape*0.103 + is_tau_norm*0.057
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b / 1-Tack / quark-jet / pt3
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=40)
        normfactor = 0.909 + is_tau_shape*0.106 + is_tau_norm*0.042
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        ############################################################################
        ## 1b / 1-Tack / unknown / pt1
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=20) & (pt<30)
        normfactor = 0.958
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b / 1-Tack / unknown / pt2
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=30) & (pt<40)
        normfactor = 0.848
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b / 1-Tack / unknown / pt3
        mask = (sample!='data') & (nbjets==1) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=40)
        normfactor = 0.953
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        ############################################################################
        ####                    1b / 3-Tack
        ############################################################################
        ## 1b / 3-Tack / gluon-jet / pt1
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=20) & (pt<30)
        normfactor = 0.719 - is_tau_shape*0.439 + is_tau_norm*0.050
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b / 3-Tack / gluon-jet / pt2
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=30) & (pt<40)
        normfactor = 0.605 - is_tau_shape*0.455 + is_tau_norm*0.062
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b / 3-Tack / gluon-jet / pt3
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=40)
        normfactor = 0.841 - is_tau_shape*0.784 + is_tau_norm*0.062
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        ############################################################################
        ## 1b / 3-Tack / quark-jet / pt1
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=20) & (pt<30)
        normfactor=1.159 + is_tau_shape*0.097 + is_tau_norm*0.050
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b / 3-Tack / quark-jet / pt2
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=30) & (pt<40)
        normfactor = 0.932 + is_tau_shape*0.070 + is_tau_norm*0.062
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b / 3-Tack / quark-jet / pt3
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=40)
        normfactor = 0.897 + is_tau_shape*0.056 + is_tau_norm*0.062
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        ############################################################################
        ## 1b / 3-Tack / unknown / pt1
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=20) & (pt<30)
        normfactor = 0.925
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b / 3-Tack / unknown / pt2
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=30) & (pt<40)
        normfactor = 1.397
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b / 3-Tack / unknown / pt3
        mask = (sample!='data') & (nbjets==1) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=40)
        normfactor = 2.696
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        ############################################################################
        ####                    2b / 1-Tack
        ############################################################################
        ## 2b / 1-Tack / gluon-jet / pt1
        mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=20) & (pt<30)
        normfactor = 1.747 - is_tau_shape*0.234 + is_tau_norm*0.059
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 2b / 1-Tack / gluon-jet / pt2
        mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=30) & (pt<40)
        normfactor = 1.560 - is_tau_shape*0.596 + is_tau_norm*0.096
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 2b / 1-Tack / gluon-jet / pt3
        mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID==21) & (pt>=40)
        normfactor = 0.918 - is_tau_shape*0.680 + is_tau_norm*0.059
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        ############################################################################
        ## 2b / 1-Tack / quark-jet / pt1
        mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=20) & (pt<30)
        normfactor = 0.810 + is_tau_shape*0.102 + is_tau_norm*0.059
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 2b / 1-Tack / quark-jet / pt2
        mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=30) & (pt<40)
        normfactor = 0.856 + is_tau_shape*0.157 + is_tau_norm*0.096
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 2b / 1-Tack / quark-jet / pt3
        mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=40)
        normfactor = 0.848 + is_tau_shape*0.081 + is_tau_norm*0.059
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        ############################################################################
        # ## 2b / 1-Tack / unknown / pt1
        # mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=20) & (pt<30)
        # normfactor = 1.0
        # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        # ## 2b / 1-Tack / unknown / pt2
        # mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=30) & (pt<40)
        # normfactor = 1.0
        # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        # ## 2b / 1-Tack / unknown / pt3
        # mask = (sample!='data') & (nbjets==2) & (nTracks==1) & (partonTruthLabelID<0) & (pdg==0) & (pt>=40)
        # normfactor = 1.0
        # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        ############################################################################
        ####                    2b / 3-Tack
        ############################################################################
        ## 2b / 1-Tack / gluon-jet / pt1
        mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=20) & (pt<30)
        normfactor = 1.623 - is_tau_shape*0.331 + is_tau_norm*0.061
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 2b / 1-Tack / gluon-jet / pt2
        mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=30) & (pt<40)
        normfactor = 1.696 - is_tau_shape*0.556 + is_tau_norm*0.066
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 2b / 1-Tack / gluon-jet / pt3
        mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID==21) & (pt>=40)
        normfactor = 1.168 - is_tau_shape*1.168 + is_tau_norm*0.070
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        ############################################################################
        ## 2b / 1-Tack / quark-jet / pt1
        mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=20) & (pt<30)
        normfactor = 0.996 + is_tau_shape*0.063 + is_tau_norm*0.061
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 2b / 1-Tack / quark-jet / pt2
        mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=30) & (pt<40)
        normfactor = 0.743 + is_tau_shape*0.065 + is_tau_norm*0.066
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 2b / 1-Tack / quark-jet / pt3
        mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID!=21) & (partonTruthLabelID>0) & (pt>=40)
        normfactor = 0.897 + is_tau_shape*0.073 + is_tau_norm*0.070
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        # ############################################################################
        # ## 2b / 1-Tack / unknown / pt1
        # mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=20) & (pt<30)
        # normfactor = 1.0
        # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        # ## 2b / 1-Tack / unknown / pt2
        # mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=30) & (pt<40)
        # normfactor = 1.0
        # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        # ## 2b / 1-Tack / unknown / pt3
        # mask = (sample!='data') & (nbjets==2) & (nTracks==3) & (partonTruthLabelID<0) & (pdg==0) & (pt>=40)
        # normfactor = 1.0
        # data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    ### Numbers for SS
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
        normfactor = 2.903 + is_tau_norm*0.804
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b+2b / 1-Tack / pt2
        mask = (sample!='data') & (nTracks==1) & (pdg==0) & (pt>=30) & (pt<40)
        normfactor = 2.695 + is_tau_norm*1.093
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b+2b / 1-Tack / pt3
        mask = (sample!='data') & (nTracks==1) & (pdg==0) & (pt>=40)
        normfactor = 1.427 + is_tau_norm*0.697 
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

        ############################################################################
        ####                    3-Tack
        ############################################################################
        ## 1b+2b / 1-Tack / pt1
        mask = (sample!='data') & (nTracks==3) & (pdg==0) & (pt>=20) & (pt<30)
        normfactor = 1.234 + is_tau_norm*0.558
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b+2b / 1-Tack / pt2
        mask = (sample!='data') & (nTracks==3) & (pdg==0) & (pt>=30) & (pt<40)
        normfactor = 1.659 + is_tau_norm*0.960
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor
        ## 1b+2b / 1-Tack / pt3
        mask = (sample!='data') & (nTracks==3) & (pdg==0) & (pt>=40)
        normfactor = 1.0 + is_tau_norm*1.0  # max(0.066 + is_tau_norm*1.000, 0)
        data.loc[mask, 'weight_'+tag] = data['weight_nominal'] * normfactor

    return data

######################################################################
#####  Correct for single tau fakes with syst
def BKGCorrector_TauSyst_OS_2b(data, cortype, which_tau, sys):

    if (sys==''):
        logging.info(style.YELLOW+"Correcting single tau background templates derived with Template Fit."+style.RESET)
    else:
        logging.info(style.YELLOW+"Computing weight for single tau background templates derived with Template Fit (with "+str(sys)+" SYST)."+style.RESET)

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

    # Compute the tau background SF in the case of q/g jet split
    def compute_weight_QGMethod(row):
        sample = row['process']
        
        nTracks = row['fs_had_tau'+which_tau_cor+'_nTrack']
        pt = row['fs_had_tau'+which_tau_cor+'_pt']
        partonTruthLabelID = row['fs_had_tau'+which_tau+'_true_partonTruthLabelID']
        pdg = row['fs_had_tau'+which_tau+'_true_pdg']
        nbjets = row['m_nbjets']

        normfactor=1
        if sample!='data':
            if nTracks==1:
                if partonTruthLabelID==21: ### --> gluons
                    if (pt>=20 and pt<30): normfactor = 1.747 - is_tau_shape*0.234 + is_tau_norm*0.059
                    elif (pt>=30 and pt<40): normfactor = 1.560 - is_tau_shape*0.596 + is_tau_norm*0.096
                    elif (pt>=40): normfactor=0.918 - is_tau_shape*0.680 + is_tau_norm*0.059
                elif partonTruthLabelID!=21 and partonTruthLabelID>0: ### --> quarks
                    if (pt>=20 and pt<30): normfactor=0.810 + is_tau_shape*0.102 + is_tau_norm*0.059
                    elif (pt>=30 and pt<40): normfactor=0.856 + is_tau_shape*0.157 + is_tau_norm*0.096
                    elif (pt>=40): normfactor=0.848 + is_tau_shape*0.081 + is_tau_norm*0.059
                elif partonTruthLabelID<0 and pdg==0: ### --> unknown
                    normfactor = 1.0*(pt>=20 and pt<30) + 1.0*(pt>=30 and pt<40) + 1.0*(pt>=40)
         
            if nTracks==3:
                if partonTruthLabelID==21:  ### --> gluons
                    if (pt>=20 and pt<30): normfactor=1.623 - is_tau_shape*0.331 + is_tau_norm*0.061
                    elif (pt>=30 and pt<40): normfactor=1.696 - is_tau_shape*0.556 + is_tau_norm*0.066
                    elif (pt>=40): normfactor=1.168 - is_tau_shape*1.168 + is_tau_norm*0.070
                elif partonTruthLabelID!=21 and partonTruthLabelID>0:  ### --> quarks
                    if (pt>=20 and pt<30): normfactor=0.996 + is_tau_shape*0.063 + is_tau_norm*0.061
                    elif (pt>=30 and pt<40): normfactor=0.743 + is_tau_shape*0.065 + is_tau_norm*0.066
                    elif (pt>=40): normfactor=0.897 + is_tau_shape*0.073 + is_tau_norm*0.070
                elif partonTruthLabelID<0 and pdg==0: ### --> unknown
                    normfactor = 1.0*(pt>=20 and pt<30) + 1.0*(pt>=30 and pt<40) + 1.0*(pt>=40)

        return row['weight_nominal']*normfactor

    def compute_weight_1Bin(row):
        sample = row['process']
        nTracks = row['fs_had_tau'+which_tau_cor+'_nTrack']
        pt = row['fs_had_tau'+which_tau_cor+'_pt']
        partonTruthLabelID = row['fs_had_tau'+which_tau+'_true_partonTruthLabelID']
        pdg = row['fs_had_tau'+which_tau+'_true_pdg']

        normfactor=1
        if sample!='data' and pdg==0:
            if nTracks==1:
                if (pt>=20 and pt<30): normfactor=1.144
                elif (pt>=30 and pt<40): normfactor=0.972
                elif (pt>=40): normfactor=0.909
            elif nTracks==3:
                if (pt>=20 and pt<30): normfactor=1.067
                elif (pt>=30 and pt<40): normfactor=0.964
                elif (pt>=40): normfactor=0.818

        return row['weight_nominal']*normfactor

    # Correction
    if cortype=='QGMethod':
        if sys=='':
            data['weight_nominal'] = data.apply(compute_weight_QGMethod, axis=1)
        else:
            data['weight_'+sys] = data.apply(compute_weight_QGMethod, axis=1)

    elif cortype=='1Bin':
        data['weight_nominal'] = data.apply(compute_weight_1Bin, axis=1)

    return data

######################################################################
#####  Correct for single lepton fakes (lepditau channel)
def LepBKGCorrector(data, templator):
    logging.info(style.YELLOW+"Correcting jet faking lepton."+style.RESET)

    def compute_weight(row):
        sample = row['process']
        TruthIFF_Class_lep1 = row['TruthIFF_Class_lep1']
        templator = row['lep_1_tmp']

        normfactor=1
        if sample!='data':
            if templator=='jet':
                if row['m_nbjets']==1:
                    normfactor = 1.396 # pm 0.310
                elif row['m_nbjets']==2:
                    normfactor = 3.956 # pm 2.034

        return row['weight_nominal']*normfactor

    # Correction
    data['weight_nominal'] = data.apply(compute_weight, axis=1)

    return data

def BkgYieldCorrector(hists):
    logging.info(style.YELLOW+"Computing Scale-Factor for jets using "+hists[0].GetName()+" histogram."+style.RESET)

    sig = ROOT.TH1D( hists[1].Clone("mc") )
    data, jet = None, None
    is_sig_started = False
    for ihist, hist in enumerate(hists):
        
        if 'data' in hist.GetName():
            data = ROOT.TH1D( hist.Clone("data") )
        elif 'jet' in hist.GetName():
            jet = ROOT.TH1D( hist.Clone("jet") )
        else:
            if not is_sig_started:
                sig = ROOT.TH1D( hist.Clone("mc") )
                is_sig_started = True
            else:
                sig.Add(hist)

    # Compute integrals
    Ndata_err, Nsig_err, Njet_err = ctypes.c_double(0.), ctypes.c_double(0.), ctypes.c_double(0.)
    Ndata = data.IntegralAndError(0, data.GetNbinsX()+1, Ndata_err, "")
    Nsig = sig.IntegralAndError(0, sig.GetNbinsX()+1, Nsig_err, "")
    Njet = jet.IntegralAndError(0, jet.GetNbinsX()+1, Njet_err, "")
    print("Data: ",Ndata,'#pm',Ndata_err.value)
    print("SIG: ",Nsig,'#pm',Nsig_err.value)
    print("Jets: ",Njet,'#pm',Njet_err.value)

    # Subtract sig from data
    Nbkg = Ndata - Nsig
    Nbkg_err = math.sqrt(Ndata_err.value**2 + Nsig_err.value**2)
    print("Subtr: ",Nbkg,'#pm',Nbkg_err)

    # Compute normalization factor Nbkg/Njet
    NF = Nbkg/Njet
    NF_err = NF*math.sqrt( (Nbkg_err/Nbkg)**2 + (Njet_err.value/Njet)**2 )

    print("Estimated SF: ",NF,"+-",NF_err)
    return NF, NF_err



def TauFakeYieldCorrector(hists, unknown_is_jet=True):
    logging.info(style.YELLOW+"Computing Scale-Factor for jet-faking-tau using "+hists[0].GetName()+" histogram."+style.RESET)

    # dummy initialization
    sig = ROOT.TH1D( hists[1].Clone("mc") )

    data, jet = None, None
    is_sig_started = False
    Nunknown = 0.0
    for ihist, hist in enumerate(hists):

        if '_data_' in hist.GetName():
            data = ROOT.TH1D( hist.Clone("data") )
        elif '_jet_' in hist.GetName():
            jet = ROOT.TH1D( hist.Clone("jet") )
        elif '_unknown_' in hist.GetName() and unknown_is_jet:
            jet.Add(hist)
        else:
            if '_tau_' in hist.GetName():
                print('Tau: %.1f' % (hist.Integral()))
            if '_unknown_' in hist.GetName():
                print('Unknown: %.1f' % (hist.Integral()))
                Nunknown = hist.Integral()
            if not is_sig_started:
                sig = ROOT.TH1D( hist.Clone("mc") )
                is_sig_started = True
            else:
                sig.Add(hist)

    # Compute integrals
    Ndata_err, Nsig_err, Njet_err = ctypes.c_double(0.), ctypes.c_double(0.), ctypes.c_double(0.)
    Ndata = data.IntegralAndError(0, data.GetNbinsX()+1, Ndata_err, "")
    Nsig = sig.IntegralAndError(0, sig.GetNbinsX()+1, Nsig_err, "")
    Njet = jet.IntegralAndError(0, jet.GetNbinsX()+1, Njet_err, "")
    print("Data: %.1f $\pm$ %.1f" % (Ndata ,Ndata_err.value) )
    print("SIG: %.1f $\pm$ %.1f" % (Nsig ,Nsig_err.value) )
    if unknown_is_jet==False: print('SIG-Unknown: %.1f' % (Nsig-Nunknown))
    print("Jets: %.1f $\pm$ %.1f" % (Njet ,Njet_err.value) )

    # Subtract sig from data
    Nbkg = Ndata - Nsig
    Nbkg_err = math.sqrt(Ndata_err.value**2 + Nsig_err.value**2)
    print("Estimated Jets: %.1f $\pm$ %.1f" % (Nbkg ,Nbkg_err) )

    # Compute normalization factor Nbkg/Njet
    NF = Nbkg/Njet
    NF_err = NF*math.sqrt( (Nbkg_err/Nbkg)**2 + (Njet_err.value/Njet)**2 )
    print("Estimated SF: %.3f $\pm$ %.3f" % (NF ,NF_err) )

    ##### For table
    print('-----------------------')
    print(Ndata,"&",Nsig-Nunknown,"&",Nunknown, "&",Njet,"&",str(Nbkg)+'\pm'+str(Nbkg_err),"&",str(NF)+'\pm'+str(NF_err))
    print('-----------------------')

    return NF, NF_err

####################################################################
###                         HISTMaker                            ###
####################################################################

def HistMaker(data, templator, dict_samples, varname, bins, xtitle=None, weight_name='weight_nominal', region_names=None, Lumi = 139, UncertaintyBand=None, UncertaintyBandRatio=None, PlotPurity=None):
    """ Function to make histograms corresponding to templates
    data          - input data frame
    templator     - name of variable in data to identify templates
    dict_samples  - disctionary containing the names of templates and fillcolors
    varname       - name of the variable to fill the histogram
    bins          - standard histogram binning [nbins, xmin, xmax]
    weight_name   - which column of dataframe to use as the weight for hists
    region_names  - list of names for each hist bin (speciall sumarry histograms)
    Lumi          - 139 (140.5) fb^-1 by default

    Returns the list of histograms for each template from templator
    """

    plottag = ''
    if templator=='tau_1_tmp':
        plottag = '_Tau1Tmp'
    elif templator=='lep_1_tmp':
        plottag = '_Lep1Tmp'
    elif templator=='tau_2_tmp':
        plottag = '_Tau2Tmp'
    elif templator=='lep_2_tmp':
        plottag = '_Lep2Tmp'

    ## find best binning
    # mask = ( data[templator] == 'data' )*( data[varname]>0 )*(data[varname]<bins[-1])
    # _, bin_edges = np.histogram(data[varname][mask].values, bins='auto')
    # print(bin_edges)
    # plt.clf()
    # _ = plt.hist(data[varname][mask].values, bins='auto') 
    # plt.savefig(varname+plottag+data.name+'.png')
    # bins = bin_edges

    hists = []
    for iproc, process in enumerate(dict_samples['sample']):
        is_MC = True
        if 'data' in process:
            is_MC = False

        histname = 'hist_' + varname + '_'+process+plottag
        hist = None
        if len(bins)==3:
            hist = ROOT.TH1D(histname, '', bins[0], bins[1], bins[2])
        else:
            xbins = np.array(bins)
            hist = ROOT.TH1D(histname, '', len(bins)-1, xbins)

        mask = data[templator] == process

        if 'eta' in varname and bins[1]>=0:
            #rootnp.fill_hist(hist, np.absolute(data[varname][mask].values), weights = data[weight_name][mask].values)
            values = np.absolute(data[varname][mask].values)
            weights = data[weight_name][mask].values
            for value,weight in zip(values,weights):
                hist.Fill(value, weight)
        else:
            #rootnp.fill_hist(hist, data[varname][mask].values, weights = data[weight_name][mask].values)
            values = data[varname][mask].values
            weights = data[weight_name][mask].values
            for value,weight in zip(values,weights):
                hist.Fill(value, weight)
        #hist.Sumw2()
        if is_MC:
            hist.Scale(Lumi)
        if region_names:
            for i in range(hist.GetNbinsX()):
                hist.GetXaxis().SetBinLabel(i+1, region_names[i])
            
        # Don't add this hist to internal registry (avoid warning message)
        hist.SetDirectory(0)

        # Append to the list
        hists.append(hist)

    # Plotting hists with Plotter (only for nominal wights)
    if weight_name=='weight_nominal':
        Plotter(hists, dict_samples, varname+plottag+data.name, xtitle=xtitle, region_tag=data.name, UncertaintyBand=UncertaintyBand,UncertaintyBandRatio=UncertaintyBandRatio)

        if PlotPurity:
            PlotSensitivity(hists, dict_samples, varname+plottag+data.name, sig_tag=PlotPurity, xtitle=xtitle, region_tag=data.name)

    return hists

####################################################################
###                         Plotter                              ###
####################################################################

def Plotter(hists, dict_samples, varname, xtitle=None, region_tag=None, UncertaintyBand=None, UncertaintyBandRatio=None):
    canvas   = ROOT.TCanvas("canvas", "canvas",10,32,668,643)

    # Legend
    leg = ROOT.TLegend(0.60, 0.65, 0.92, 0.92)
    leg.SetNColumns(2)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize( 0.03 )
    leg.SetFillColor(0)
    leg.SetLineColor(0)

    # Text block for ATLAS label
    la = ROOT.TLatex()
    la.SetNDC()
    la.SetTextFont(52)
    la.SetTextSize(0.05)
    la.SetTextColor(ROOT.kBlack)


    stack = ROOT.THStack("stack","stacked histograms")

    uplongpad = ROOT.TPad("uplongpad", "uplongpad", 0, 0.31, 1, 1)
    uplongpad.SetBottomMargin(0)
    lowerPad = ROOT.TPad("lowerPad", "lowerPad", 0, 0.03, 1, 0.31)
    lowerPad.SetBottomMargin(0.35)
    lowerPad.SetTopMargin(0.05)
    originPad = ROOT.TPad("originPad", "originPad", 0.12, 0.285, 0.155, 0.335)

    doRatio = True
    if doRatio:
        uplongpad.Draw()
        lowerPad.Draw()
        originPad.Draw()
        uplongpad.cd()

    mc = ROOT.TH1D( hists[1].Clone("mc") )
    data = ROOT.TH1D( hists[0].Clone("data") )
    for ihist, hist in enumerate(hists):
        if ihist>0:
            hist.SetLineColor(dict_samples['fillcolor'][ihist])
            hist.SetFillColor(dict_samples['fillcolor'][ihist])
            stack.Add(hist)
        else: # data
            hist.SetMarkerColor(1)
            hist.SetMarkerSize(0.85)
            hist.SetMarkerStyle(20)
            hist.SetLineColor(1) 
            hist.SetLabelSize(0.107)
            hist.GetYaxis().SetTitle("Events")
        # Avoid double counting of hist[1] when adding to 'mc'
        if ihist<2:
            continue
        mc.Add(hist)

    # Set the Y-axis range
    ymax = 1111111
    if hists[0]:
        ymax = hists[0].GetBinContent( hists[0].GetMaximumBin() )
        if ymax < mc.GetBinContent( mc.GetMaximumBin() ):
            ymax = mc.GetBinContent( mc.GetMaximumBin() )
        hists[0].GetYaxis().SetRangeUser(0, 1.6*ymax)
        hists[0].Draw('e')


    # Drawing
    stack.Draw("histsame")
    stack.GetHistogram().SetLineWidth(1)

    #if hists[0]:
    hists[0].Draw('AXISSAME')
    hists[0].Draw('ESAME')

    # Uncertanty
    if UncertaintyBand:
        UncertaintyBand.SetFillColor(ROOT.kAzure+2)
        UncertaintyBand.SetFillStyle(3001)
        UncertaintyBand.Draw('E2SAME')


    # Legend
    leg.Clear()
    for ihist, hist in enumerate(hists):
        if dict_samples['sample'][ihist] == 'data':
            leg.AddEntry(hist, 'Data','p') # lp
        else:
            leg.AddEntry(hist, dict_samples['sample'][ihist],'f')
    if UncertaintyBand:
        leg.AddEntry(UncertaintyBand,'Bkg.Unc.','f')
    leg.Draw()

    # Add ATLAS label
    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextFont(72)
    text.SetTextSize(0.045)
    text.DrawLatex(0.21, 0.86, "ATLAS")
    text.SetTextFont(42)
    #text.DrawLatex(0.21 + 0.16, 0.86, "Internal")
    text.SetTextSize(0.04)
    text.DrawLatex(0.21, 0.80, "#sqrt{{s}} = 13 TeV, {:.1f} fb^{{-1}}".format(139.0))

    if xtitle:
        #region_tag = varname.split("_")[-1]
        region_tag = region_tag[1:]
        # Add region label
        text_tag = ROOT.TLatex()
        text_tag.SetNDC()
        text_tag.SetTextFont(42)
        text_tag.SetTextSize(0.045)
        text_tag.DrawLatex(0.21, 0.72, "Region: "+region_tag)

    # Ratio
    if doRatio and hists[0]:
        lowerPad.cd()
        lowerPad.SetGridy()
        hratio = ROOT.TH1D(hists[0].Clone("hratio"))
        hratio.SetMarkerSize(0)
        hratio.SetLineWidth(2)
        hratio.Divide(hists[0],mc,1.,1.,"P")
        
        if xtitle:
            hratio.SetXTitle(xtitle)
        else:
            hratio.SetXTitle(varname)
        hratio.GetYaxis().SetRangeUser(0.6,1.4)
        hratio.SetMarkerSize(0.85)
        hratio.SetMarkerStyle(20)
        ratiosize=0.127
        hratio.GetYaxis().SetLabelSize(ratiosize)
        hratio.GetYaxis().SetTitleSize(ratiosize)
        hratio.GetXaxis().SetLabelSize(ratiosize)
        hratio.GetXaxis().SetTitleSize(ratiosize)
        hratio.GetYaxis().SetTitle('Data/Pred.')
        hratio.GetYaxis().SetTitleOffset(0.43)
        hratio.GetYaxis().SetNdivisions(506)
        hratio.Draw("p:e")

        # Uncertanty in ratio
        if UncertaintyBandRatio:
            UncertaintyBandRatio.SetFillColor(ROOT.kAzure+2)
            UncertaintyBandRatio.SetFillStyle(3001)
            UncertaintyBandRatio.Draw('E2SAME')


    canvas.SaveAs('Plots/'+varname+'.pdf')

####################################################################
###                       Sensitivity                            ###
####################################################################

def PlotSensitivity(hists, dict_samples, varname, sig_tag='tau', xtitle=None, region_tag=None):

    canvas   = ROOT.TCanvas("canvas", "canvas",10,10,800,600)
    # Legend
    leg = ROOT.TLegend(0.50, 0.67, 0.82, 0.85)
    leg.SetNColumns(2)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.05)
    leg.SetFillColor(0)
    leg.SetLineColor(0)

    # Text block for ATLAS label
    la = ROOT.TLatex()
    la.SetNDC()
    la.SetTextFont(52)
    la.SetTextSize(0.06)
    la.SetTextColor(ROOT.kBlack)

    canvas.Divide(1,2)
    canvas.cd(1)
    ROOT.gPad.SetBottomMargin(0.15)

    # Exctract the histograms
    data, mc, sig = ROOT.TH1D( hists[0].Clone("data") ), None, None
    for ihist, hist in enumerate(hists):
        if ihist==0: continue # skip data
        if not mc:
            mc = ROOT.TH1D( hists[ihist].Clone("mc") )
        else:
            mc.Add(hist)
        if dict_samples['sample'][ihist] == sig_tag:
            sig = ROOT.TH1D( hists[ihist].Clone("sig") )

    # Get Cumulative
    sig = sig.GetCumulative(False)
    mc = mc.GetCumulative(False)

    # Purity
    purity_mc = ROOT.TH1D( sig.Clone("purity_mc") )
    purity_mc.Divide(mc)
    purity_mc.GetYaxis().SetTitle(sig_tag+' purity')
    purity_mc.GetYaxis().SetTitleOffset(0.75)
    purity_mc.SetXTitle(xtitle) if xtitle else purity_mc.SetXTitle(varname)
    purity_mc.SetLineColor(ROOT.kBlack)
    purity_mc.Draw("E")
    leg.AddEntry(purity_mc,sig_tag,"fl")
    leg.Draw()

    # Add ATLAS label
    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextFont(72)
    text.SetTextSize(0.045)
    text.DrawLatex(0.21, 0.86, "ATLAS")
    text.SetTextFont(42)
    text.DrawLatex(0.21 + 0.07, 0.86, "Simulation")
    text.SetTextSize(0.04)
    text.DrawLatex(0.21, 0.80, "#sqrt{{s}} = 13 TeV, {:.1f} fb^{{-1}}".format(139.0))

    if xtitle:
        region_tag = region_tag[1:]
        text_tag = ROOT.TLatex()
        text_tag.SetNDC()
        text_tag.SetTextFont(42)
        text_tag.SetTextSize(0.048)
        text_tag.DrawLatex(0.21, 0.72, "Region: "+region_tag)

    # Significance
    canvas.cd(2)
    significance_mc = ROOT.TH1D( sig.Clone("significance_mc") )
    mc_sqrt = ROOT.TH1D( mc.Clone("mc_sqrt") )
    for i in range(significance_mc.GetNbinsX()):
        if not mc_sqrt.GetBinContent(i+1):
            mc_sqrt.SetBinContent(i+1, 0)
        else:
            mc_sqrt.SetBinContent(i+1, math.sqrt(mc_sqrt.GetBinContent(i+1)))
    significance_mc.Divide(mc_sqrt)
    significance_mc.SetLineColor(ROOT.kBlack)
    significance_mc.Draw("E")
    significance_mc.SetXTitle(xtitle) if xtitle else significance_mc.SetXTitle(varname)
    significance_mc.GetYaxis().SetTitle(sig_tag+' significance')
    significance_mc.GetYaxis().SetTitleOffset(0.73)

    canvas.SaveAs('Plots/'+varname+'_sensit.pdf')


####################################################################
###                    For uncertainty band                      ###
####################################################################
def SumMCHists(hists, systtag):
    mc = ROOT.TH1D( hists[1].Clone("mc_"+systtag) )

    # Iterate over hists and add them to the model
    for ihist, hist in enumerate(hists):
        # Avoid double counting of hist[1] when adding to 'mc'
        # Skip hist[0] because it's reserved for data
        if ihist<2:
            continue
        mc.Add(hist)

    return mc

### Subtract nominal hist
def GetAbsVariation(varhist, nomhist):
    varhist.Add(nomhist, -1)
    return varhist

### Produce two TGraph with uncertanties
def GetUncertGraph(varhistsUp, varhistsDn, mc_nom, isRatio=False):

    if len(varhistsUp)!=len(varhistsDn):
        logging.error(style.RED+"Number of UP and DN syst varitions does not agree."+style.RESET)
        return None

    systvariations = []
    for ihist, histUp in enumerate(varhistsUp):
        histDn = varhistsDn[ihist]

        systperbin = []
        for i in range(histUp.GetNbinsX()):
            
            # Symmetrize
            binvalue = 0.5*abs( histUp.GetBinContent(i+1) - histDn.GetBinContent(i+1))
            systperbin.append(binvalue)

        systvariations.append(systperbin)

    # compute the sum in quadrature per bin
    a = np.array(systvariations)
    a2 = np.square(a)
    a2sum = np.sum(a2, axis=0)
    a2sumsqrt = np.sqrt(a2sum).tolist()

    # extract information from nominal hist
    mc_nom_yvalues = []
    mc_nom_xvalues = []
    mc_nom_xerrvalues = []
    for i in range(mc_nom.GetNbinsX()):
        mc_nom_yvalues.append( mc_nom.GetBinContent(i+1) )
        mc_nom_xvalues.append( mc_nom.GetBinCenter(i+1) )
        mc_nom_xerrvalues.append( mc_nom.GetBinWidth(i+1)*0.5 )


    # Produce the TGraph for uncertainty band in upper plot
    n = len(a2sumsqrt)
    y = array( 'f', mc_nom_yvalues)
    x = array( 'f', mc_nom_xvalues)
    yerr  = array( 'f', a2sumsqrt)
    xerr = array( 'f', mc_nom_xerrvalues)
    gr = ROOT.TGraphErrors( n, x, y, xerr, yerr )

    # Produce the TGraph for uncertainty band in ratio plot
    y_ratio = array( 'f', np.ones(n))
    denom = np.array(mc_nom_yvalues)
    yerr_ratio = array( 'f', np.divide(a2sumsqrt, denom))
    gr_ratio = ROOT.TGraphErrors( n, x, y_ratio, xerr, yerr_ratio )

    return gr,gr_ratio

def UncertaintyMaker(data, templator, dict_samples, varname, bins, systnames):

    # Nominal histogram
    hists = HistMaker(data, templator, dict_samples, varname, bins)
    mc_nom = SumMCHists(hists, 'nominal')

    # Variated histograms paired into up/down
    totmchists = []
    varhistsUp,varhistsDn=[],[]
    for systtag in systnames:
        if logging.getLogger().getEffectiveLevel()==logging.DEBUG:
            logging.debug('LFAHelpers.UncertaintyMaker   -->Producing variated MC with weight_'+systtag+' for '+varname+' histogram')
        hists = HistMaker(data, templator, dict_samples, varname, bins, weight_name='weight_'+systtag)
        mc = SumMCHists(hists, systtag)
        mc_scaled = GetAbsVariation(mc, mc_nom)
        if '_up' in systtag:
            varhistsUp.append(mc_scaled)
        elif '_down' in systtag:
            varhistsDn.append(mc_scaled)
        totmchists.append( mc_scaled )

    UncertaintyBand,UncertaintyBandRatio = GetUncertGraph(varhistsUp, varhistsDn, mc_nom)
    return UncertaintyBand,UncertaintyBandRatio


####################################################################
###                    Manipulations with Data                   ###
####################################################################
def sort_taus(dframe):
    logging.info(" ")
    logging.info("======================= Splitting light leptons and taus ======================= ")
    logging.info(style.YELLOW+"Sorting lepX variables and separating taus."+style.RESET)

    # if lep1 is tau: move lep2 to lep1 and lep3 to lep2
    mask = dframe['type_lep1']==3
    dframe.loc[mask, 'pt_lep1'] = dframe[mask]['pt_lep2']
    dframe.loc[mask, 'eta_lep1'] = dframe[mask]['eta_lep2']
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
    dframe.loc[mask, 'charge_lep2'] = dframe[mask]['charge_lep3']
    dframe.loc[mask, 'isTight_lep2'] = dframe[mask]['isTight_lep3']
    dframe.loc[mask, 'type_lep2'] = dframe[mask]['type_lep3']
    dframe.loc[mask, 'TruthIFF_Class_lep2'] = dframe[mask]['TruthIFF_Class_lep3']
    dframe.loc[mask, 'ECIDS_lep2'] = dframe[mask]['ECIDS_lep3']
    dframe.loc[mask, 'ele_ambiguity_lep2'] = dframe[mask]['ele_ambiguity_lep3']
    dframe.loc[mask, 'ele_AddAmbiguity_lep2'] = dframe[mask]['ele_AddAmbiguity_lep3']
    
    # drop lep3 columns
    dframe = dframe.drop(['pt_lep3','eta_lep3','charge_lep3','type_lep3','isTight_lep3', 'TruthIFF_Class_lep3',
                          'ECIDS_lep3', 'ele_ambiguity_lep3', 'ele_AddAmbiguity_lep3'], axis=1)

    return dframe

def sort_taus_ditau(dframe):
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
###                          PIE-CHARTS                          ###
####################################################################

def PieCharter(data, sampler, dict_samples, plotname, show_fractions=False):
    # To print out total yields:
    Lumi = 139

    df = data.groupby([sampler])['weight_nominal'].sum().reset_index()
    for iproc, process in enumerate(dict_samples['sample']):
        df = df.replace(iproc, process)

    df = df.set_index(sampler)
    if 'data' in df.index.values:
        df = df.drop(index='data')
    ## disable sorting?
    #df = df.sort_values(by=['weight_nominal'], ascending=False)
    total = df['weight_nominal'].sum()

    df_yields = df.copy()
    df_yields['weight_nominal'] = df_yields.weight_nominal.apply(lambda row: row*Lumi if row>0 else 0.000001 )
    if show_fractions:
        df_yields.loc['Total'] = df_yields.sum(numeric_only=True)
        logging.info('Showing yields for '+data.name[1:]+' dataframe.\n\t =============== Total yields ===============\n\t'+df_yields.to_string().replace('\n', '\n\t') )
    
    ## Normalize
    df['weight_nominal'] = df.weight_nominal.apply(lambda row: row/total if row>0 else 0.000001 )
    if show_fractions:
        logging.info(' =============== Fractions ===============')
        logging.info('Showing fractions for '+data.name[1:]+' dataframe.\n\t =============== Fractions ===============\n\t'+df.to_string().replace('\n', '\n\t') )

    plot = df.plot.pie( y='weight_nominal',
                        #title=sampler,
                        fontsize=15,
                        figsize=(5, 5))
    y_axis = plot.axes.get_yaxis()
    y_axis.set_visible(False)
    fig = plot.get_figure()
    fig.savefig(plotname)
    plt.close(fig)

    
def compute_last_fraction(frac_vals, frac_errs):
    logging.debug('Computing the last fraction as 1-sum(L), with frac_vals={}'.format(frac_vals))
    fbkg_val_last = 1. - sum(frac_vals)
    logging.debug('Computing the last fraction error as sqrt(sum(frac_errs))')
    fbkg_err_last = -1*math.sqrt(sum(x*x for x in frac_errs))
    return fbkg_val_last, fbkg_err_last


####################################################################
###                    Cross-check functions                     ###
####################################################################

def SystSolver(hists, dict_tmps_comb, region_names):
    """ This function get yields from histogram and solves the system of linear equations.
    region_names     - names of regions in each bin of the histos (SR, TnT, nTT, nTnT)
    hists            - list of histograms 'data', 'sim,sim', 'sim,jet', 'jet,jet'
    """

    b, a = [], []
    Ndata, Nsim=[],[]
    for ibin,binname in enumerate(region_names):
        if ibin==0: continue # ignore SR


        # b = Ndata - Nsig
        b.append(hists[0].GetBinContent(ibin+1) - hists[1].GetBinContent(ibin+1))
        Ndata.append(hists[0].GetBinContent(ibin+1))
        Nsim.append(hists[1].GetBinContent(ibin+1))

        ai = []
        for ihist, hist in enumerate(hists[2:]):
            ai.append(hist.GetBinContent(ibin+1))
            #name = dict_tmps_comb['fillcolor'][ihist]

        a.append(ai)
    
    print('Ndata = ', Ndata)
    print('Nsimsim = ', Nsim)
    print('N[TnT] = ', a[0])
    print('N[nTT] = ', a[1])
    print('N[nTnT] = ', a[2])
    x = np.linalg.solve(a, b)
    print('Exact solution: ',x)

####################################################################
###                 Transform Config arguments                   ###
####################################################################

def config_parse(config):
    if config.get('BJETS'): config['BJETS'] = [int(x) for x in config['BJETS'].split(',')] if type(config['BJETS']) is str else [config['BJETS']]

    if config.get('PLOTPROCESSES_VARS'): config['PLOTPROCESSES_VARS'] = config['PLOTPROCESSES_VARS'].replace(' ','').split(',')
    if config.get('PLOTPROCESSES_REGIONS'): config['PLOTPROCESSES_REGIONS'] = config['PLOTPROCESSES_REGIONS'].replace(' ','').split(',')
    if config.get('PLOTTAU1TMP_VARS'):  config['PLOTTAU1TMP_VARS'] = config['PLOTTAU1TMP_VARS'].replace(' ','').split(',')
    if config.get('PLOTTAU1TMP_REGIONS'):  config['PLOTTAU1TMP_REGIONS'] = config['PLOTTAU1TMP_REGIONS'].replace(' ','').split(',')
    if config.get('PLOTTAU2TMP_VARS'):  config['PLOTTAU2TMP_VARS'] = config['PLOTTAU2TMP_VARS'].replace(' ','').split(',')
    if config.get('PLOTTAU2TMP_REGIONS'):  config['PLOTTAU2TMP_REGIONS'] = config['PLOTTAU2TMP_REGIONS'].replace(' ','').split(',')
    if config.get('PLOTLEP1TMP_VARS'):  config['PLOTLEP1TMP_VARS'] = config['PLOTLEP1TMP_VARS'].replace(' ','').split(',')
    if config.get('PLOTLEP1TMP_REGIONS'):  config['PLOTLEP1TMP_REGIONS'] = config['PLOTLEP1TMP_REGIONS'].replace(' ','').split(',')
    if config.get('PLOTLEP2TMP_VARS'):  config['PLOTLEP2TMP_VARS'] = config['PLOTLEP2TMP_VARS'].replace(' ','').split(',')
    if config.get('PLOTLEP2TMP_REGIONS'):  config['PLOTLEP2TMP_REGIONS'] = config['PLOTLEP2TMP_REGIONS'].replace(' ','').split(',')
    
    if config.get('EXTRACT_TAUBKG_METHOD'):   config['EXTRACT_TAUBKG_METHOD'] = config['EXTRACT_TAUBKG_METHOD'].replace(' ','').split(',')
    if config.get('CORRECT_BKG_TAU_CONFIGS'): config['CORRECT_BKG_TAU_CONFIGS'] = config['CORRECT_BKG_TAU_CONFIGS'].replace(' ','').split(',')

    # Uncertainty band?
    if not config.get('CORRECT_BKG_DILEP'): config['CORRECT_BKG_DILEP'] = False
    if not config.get('CORRECT_BKG_LEP'): config['CORRECT_BKG_LEP'] = False
    if not config.get('CORRECT_BKG_DITAU'): config['CORRECT_BKG_DITAU'] = False
    if not config.get('CORRECT_BKG_TAU'): config['CORRECT_BKG_TAU'] = False
    if not config.get('USE_SF_FROM_OS'): config['USE_SF_FROM_OS'] = False
    config['ADD_UNCERTAINTY'] = config['CORRECT_BKG_TAU'] or config['CORRECT_BKG_DILEP'] or config['CORRECT_BKG_DITAU'] or config['CORRECT_BKG_LEP']

    if not config.get('PLOTTAU1TMP'): config['PLOTTAU1TMP'] = False
    if not config.get('PLOTTAU2TMP'): config['PLOTTAU2TMP'] = False
    if not config.get('PLOTPROCESSES'): config['PLOTPROCESSES'] = False
    if not config.get('PLOTLEP1TMP'): config['PLOTLEP1TMP'] = False
    if not config.get('PLOTLEP2TMP'): config['PLOTLEP2TMP'] = False

    # MVA paramters
    if not config.get('TRAIN_THQ'): config['TRAIN_THQ'] = False
    if not config.get('MVA_METHOD'): config['MVA_METHOD'] = 'LGBM'
    if not config.get('TUNE_PARS'): config['TUNE_PARS'] = None
    if not config.get('EVAL_ARGMAX'): config['EVAL_ARGMAX'] = False
    if not config.get('EVAL_TERNARY'): config['EVAL_TERNARY'] = False
    if not config.get('EVAL_SHAP'): config['EVAL_SHAP'] = False
    if not config.get('EVAL_LGBM_IMPORTANCE'): config['EVAL_LGBM_IMPORTANCE'] = False

    # QGTagger
    if not config.get('TRAIN_QGTAGGER'): config['TRAIN_QGTAGGER'] = False

    return config

####################################################################
###                        Get fakes SFs                         ###
####################################################################
def get_fakesSFs(config):
    from LFAConfig import get_fakesSFs_default
    fakesSFs = get_fakesSFs_default()

    if config['AnalysisChannel']=='lephad' and config['CORRECT_BKG_TAU']:
        if config['CORRECT_BKG_TAU_CONFIGS'] and len(config['CORRECT_BKG_TAU_CONFIGS'])==2:
            if config['CORRECT_BKG_TAU_CONFIGS'][0]:  # Counting method
                logging.info('Reading {} file with fake SFs from Counting method'.format(config['CORRECT_BKG_TAU_CONFIGS'][0]))
                file = ROOT.TFile.Open(config['CORRECT_BKG_TAU_CONFIGS'][0], "READ")

                for nbjet in [1,2]:
                    for prong in [1,3]:
                        hist = ROOT.TH1D(file.Get('SFs_'+str(nbjet)+'b_'+str(prong)+'prong'))
                        for ptbin in [1,2,3]:
                            SF_name = 'Counting_'+str(nbjet)+'b_'+str(prong)+'p_pt'+str(ptbin)+'_nominal'
                            SF_err_name = 'Counting_'+str(nbjet)+'b_'+str(prong)+'p_pt'+str(ptbin)+'_norm'

                            fakesSFs[SF_name] = hist.GetBinContent(ptbin)
                            fakesSFs[SF_err_name] = hist.GetBinError(ptbin)
                            logging.debug(SF_name, fakesSFs[SF_name])
                            logging.debug(SF_err_name, fakesSFs[SF_err_name])

                file.Close()

    return fakesSFs
