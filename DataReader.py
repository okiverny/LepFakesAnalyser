import numpy as np
import pandas as pd
import glob
import os,sys
import logging
import ROOT

from LFAHelpers import style

################################################################################
############################## Load data  ######################################

def load_data_RDF(input_folder, dict_samples, list_of_branches_leptons, list_of_branches_mc, tree_name='tHqLoop_nominal_Loose'):

    logging.info("======================= R e a d i n g  D a t a ======================= ")
    logging.info("Loading data from "+style.YELLOW+input_folder+style.RESET)

    ROOT.ROOT.EnableImplicitMT()

    # for cross-check
    remaining_files = glob.glob(input_folder + '*.root')

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
                logging.warning('WARNING! Did not find any file with the key: ' + fname_keys[index])
                continue
            
            if index>0: msg += ', '
            msg += fname_keys[index]

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

            # Are there files in the input folder but not in the config?
            for file in glob.glob(file_glob):
                if file in remaining_files:
                    remaining_files.remove(file)
            
        logging.info(msg)

    # Concatinating
    output_df = pd.concat(dframes, ignore_index=True)

    # Are there files in the input folder but not in the config?
    for file in remaining_files:
        logging.warning('This input file is in the input folder but not in the dictionary: ' + file.split('/')[-1])

    return output_df