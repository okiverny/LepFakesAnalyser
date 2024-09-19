def Configurate(channel):
    # Input folder
    tree_name = 'tHqLoop_nominal_Loose'

    dict_samples = {
        'sample':     ['data','tH','Diboson','tZ','ttW','ttZ','tWZ/tWH','ttH','other','tW','Wjets','Zjets',
                       #'Wjetsnew','Zjetsnew',
                       'ttbar'],
        'fname_keys': [ ["*data*.root"],          # data
                        ["*346799*"],            # tH=2
                        ["*36425*", "*36428*","*36429*","*36335*","*36336*","*36348*"], # Diboson=83
                        ["*512059*" if channel=='lephad' else "*412063*"], # tZq=6
                        ["*410155*"], # ttW=205
                        ["*410156*","*410157*","*410218*","*410219*","*410220*","*410276*","*410277*","*410278*"], #ttZ=881
                        ["*412118*","*346678*"], #tWZ/tWH=429
                        ["*346343*","*346344*","*346345*"],  #ttH=4
                        ["*41065*","*41204*","*41040*","*36424*","*36424*", "*34228*", "*304014*"],  # minor=91
                        ["*41064*"],  # tW=210
                        #["*36417*", "*36418*","*36415*","*36416*","*364190*","*364191*","*364192*","*364193*",  # Wjets=13
                        #   "*364194*","*364195*","*364196*","*364197*"],  # "*36420*"
                        #["*36410*", "*36411*","*36412*","*36413*",  "*364140*","*364141*", "*364198*", "*364199*","*36420*", "*36421*"],  # Zjets=93
                        ["*700338*","*700339*","*70034*"],   # New W+jets=13
                        ["*70032*","*700330*","*700331*","*700332*","*700333*","*700334*"],   # New Z+jets=93
                        #["*410472.FS*.root"]    # ttbar=209 (older version)
                        ["*410470.FS*.root", "*410471.FS*.root"]    # ttbar=209 (new version)
                      ],
        'fillcolor':  [ 1,2,83,6,205,881,429,4,91,210,13,93,
                       #14,94, # new V+jets
                       209]
    }


    dict_tmps_lep = {
        'sample':    ['data', 'elec', 'muon', 'c-flip', 'jet', 'rest'],
        'fillcolor': [1, 3, 4, 5, 6, 7]
    }

    dict_tmps_tau = {
        'sample':    ['data', 'tau', 'elec', 'muon', 'jet', 'unknown'],
        'fillcolor': [1, 3, 4, 5, 6, 42]
    }

    dict_tmps_tau_fine = {
        'sample':    ['data', 'tau', 'elec', 'muon', 'q-jet', 'g-jet', 'unknown'],
        #'fillcolor': [1, 3, 4, 5, 6, 7, 42],
        'fillcolor': [1, 416+3, 4, 800+5, 632+1, 432-6, 42],
    }

    # Variables
    list_of_branches_leptons=['pt_lep1','eta_lep1','charge_lep1','type_lep1','isTight_lep1',
                          'pt_lep2','eta_lep2','charge_lep2','type_lep2','isTight_lep2',
                          'pt_lep3','eta_lep3','charge_lep3','type_lep3','isTight_lep3',
                          'ECIDS_lep1','ECIDS_lep2','ECIDS_lep3',
                          'ele_ambiguity_lep1','ele_ambiguity_lep2','ele_ambiguity_lep3',
                          'ele_AddAmbiguity_lep1','ele_AddAmbiguity_lep2','ele_AddAmbiguity_lep3',
                          'weight_nominal', # 'weight_nominalWtau' 'weight_nominalWtau_fix' weight_nominal
                          'm_njets','m_nbjets',
                          #'m_met',
                          'fs_had_tau_1_pt','fs_had_tau_1_eta','fs_had_tau_1_nTrack','fs_had_tau_1_tight','fs_had_tau_1_RNNScore','fs_had_tau_1_JetTrackWidth','fs_had_tau_1_phi',
                          'fs_had_tau_2_pt','fs_had_tau_2_eta','fs_had_tau_2_nTrack','fs_had_tau_2_tight','fs_had_tau_2_RNNScore','fs_had_tau_2_JetTrackWidth',
                          'fs_had_tau_1_JetCaloWidth','fs_had_tau_2_JetCaloWidth',
                          #'pt_jet1',
                          'pt_jet2']
                          #'NNout_tauFakes']

    list_of_branches_mc = ['TruthIFF_Class_lep1', 'TruthIFF_Class_lep2','TruthIFF_Class_lep3',
                        'fs_had_tau_true_pdg',  'fs_had_tau_true_partonTruthLabelID',
                        'fs_had_tau_2_true_pdg','fs_had_tau_2_true_partonTruthLabelID']

    list_of_branches_ditau = ['had_tau_1_charge','had_tau_2_charge']

    list_of_branches_mva = ['eta_jf','phi_jf','pt_jf', 'M_b_jf',
                            'eta_b','phi_b','pt_b','MMC_out_1',
                            'HvisEta','HvisPt','m_sumet','m_met',
                            'TvisEta','TvisMass','TvisPt','HT_all',
                            'lep_Top_eta','lep_Top_pt','lep_Top_phi','deltaRTau',
                            'pt_jet1', 'phi_jet1','eta_jet1','m_phi_met',
                            'deltaPhiTau', 'had_tau_pt','had_tau_eta',
                            'lep_Higgs_pt','lep_Higgs_eta','lep_Higgs_phi']

    
    if channel=='lephad':
        #input_folder = '/Users/okiverny/workspace/tH/3l1tau_loose_april_2/' # older version
        input_folder = '/Users/okiverny/workspace/tH/3l1tau_loose_august/'
        list_of_branches_leptons += list_of_branches_mva
    elif channel=='hadhad':
        input_folder = '/Users/okiverny/workspace/tH/3l2tau_loose_april/'
        list_of_branches_leptons += list_of_branches_ditau

    return tree_name, dict_samples, dict_tmps_lep, dict_tmps_tau, dict_tmps_tau_fine, list_of_branches_leptons, list_of_branches_mc, input_folder

def Configurate_VarHistBins():
    dict_hists = {
        'm_njets':  [8,0,8],
        'm_nbjets':  [3,0,3],

        ### Tau
        'fs_had_tau_1_pt': [10,0,100], # 20
        'fs_had_tau_2_pt': [8,20,100], # 20
        'fs_had_tau_1_eta': [6,0,2.4],
        'fs_had_tau_2_eta': [6,0,2.4],
        'fs_had_tau_1_tight': [2,0,2],
        'fs_had_tau_2_tight': [2,0,2],
        'fs_had_tau_1_RNNScore': [20,0,1],
        'fs_had_tau_2_RNNScore': [20,0,1],
        'fs_had_tau_true_pdg': [32,-15,16],
        'fs_had_tau_2_true_pdg': [32,-15,16],
        'fs_had_tau_true_partonTruthLabelID': [24,-1.5,22.5],
        'fs_had_tau_2_true_partonTruthLabelID': [24,-1.5,22.5],
        'fs_had_tau_1_nTrack': [4, 0, 4],
        'fs_had_tau_2_nTrack': [4, 0, 4],
        #'fs_had_tau_1_JetTrackWidth': [20, 0.0, 0.4],
        'fs_had_tau_2_JetTrackWidth': [20, 0.0, 0.4],
        'fs_had_tau_1_JetCaloWidth': [20, 0.0, 0.4],
        'fs_had_tau_2_JetCaloWidth': [20, 0.0, 0.4],
        'fs_had_tau_1_JetTrackWidth': [31, -0.01, 0.30],
        #'fs_had_tau_1_JetTrackWidth': [130, -1.1, 0.30],

        'fs_had_tau_1_JetTrackWidth_p1pt1': [x*0.01 for x in range(30)],
        'fs_had_tau_1_JetTrackWidth_p1pt2': [x*0.01 for x in range(25)],
        'fs_had_tau_1_JetTrackWidth_p1pt3': [x*0.01 for x in range(18)],
        'fs_had_tau_1_JetTrackWidth_p3pt1': [x*0.01 for x in range(30)],
        'fs_had_tau_1_JetTrackWidth_p3pt2': [x*0.01 for x in range(18)],
        'fs_had_tau_1_JetTrackWidth_p3pt3': [x*0.01 for x in range(18)],


        ### Optimized binning (dileptau channel)
        # 'fs_had_tau_1_JetTrackWidth_p1pt1_pf': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29],
        # 'fs_had_tau_1_JetTrackWidth_p1pt2_pf': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24],
        # 'fs_had_tau_1_JetTrackWidth_p1pt3_pf': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.11, 0.12, 0.13, 0.15, 0.16],
        # 'fs_had_tau_1_JetTrackWidth_p3pt1_pf': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.15, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28],
        # 'fs_had_tau_1_JetTrackWidth_p3pt2_pf': [0.0, 0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.1, 0.11, 0.13, 0.14, 0.15, 0.16],
        # 'fs_had_tau_1_JetTrackWidth_p3pt3_pf': [0.0, 0.01, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.16, 0.17],

        ### Optimized binning (dileptau OS channel)
        'fs_had_tau_1_JetTrackWidth_p1pt1_pf': [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28],
        'fs_had_tau_1_JetTrackWidth_p1pt2_pf': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24],
        'fs_had_tau_1_JetTrackWidth_p1pt3_pf': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.11, 0.16],
        'fs_had_tau_1_JetTrackWidth_p3pt1_pf': [0.0, 0.01, 0.02, 0.03, 0.04, 0.07, 0.08, 0.1, 0.11, 0.12, 0.15, 0.16, 0.18, 0.19, 0.21, 0.22, 0.23, 0.25, 0.26, 0.27, 0.28, 0.29],
        'fs_had_tau_1_JetTrackWidth_p3pt2_pf': [0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.09, 0.11, 0.13, 0.14, 0.15, 0.16, 0.17],
        'fs_had_tau_1_JetTrackWidth_p3pt3_pf': [0.0, 0.01, 0.03, 0.05, 0.06, 0.07, 0.09, 0.1, 0.11, 0.14, 0.15, 0.17],

        ### Optimized binning (dileptau OS 2b channel)
        # 'fs_had_tau_1_JetTrackWidth_p1pt1_pf': [0.0, 0.02, 0.03, 0.04, 0.08, 0.09, 0.1, 0.11, 0.13, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.25, 0.26, 0.27, 0.28],
        # 'fs_had_tau_1_JetTrackWidth_p1pt2_pf': [0.0, 0.01, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.22, 0.23, 0.24],
        # 'fs_had_tau_1_JetTrackWidth_p1pt3_pf': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
        # 'fs_had_tau_1_JetTrackWidth_p3pt1_pf': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29],
        # 'fs_had_tau_1_JetTrackWidth_p3pt2_pf': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
        # 'fs_had_tau_1_JetTrackWidth_p3pt3_pf': [0.0, 0.01, 0.04, 0.07, 0.08, 0.09, 0.1, 0.11, 0.16],

        ### Optimized binning (dileptau OS 1b channel)
        'fs_had_tau_1_JetTrackWidth_1b_p1pt1': [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28],
        'fs_had_tau_1_JetTrackWidth_1b_p1pt2': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24],
        'fs_had_tau_1_JetTrackWidth_1b_p1pt3': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.11, 0.16],
        'fs_had_tau_1_JetTrackWidth_1b_p3pt1': [0.0, 0.01, 0.02, 0.03, 0.04, 0.07, 0.08, 0.1, 0.11, 0.12, 0.15, 0.16, 0.18, 0.19, 0.21, 0.22, 0.23, 0.25, 0.26, 0.27, 0.28, 0.29],
        'fs_had_tau_1_JetTrackWidth_1b_p3pt2': [0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.09, 0.11, 0.13, 0.14, 0.15, 0.16, 0.17],
        'fs_had_tau_1_JetTrackWidth_1b_p3pt3': [0.0, 0.01, 0.03, 0.05, 0.06, 0.07, 0.09, 0.1, 0.11, 0.14, 0.15, 0.17],
        ### Optimized binning (dileptau OS 2b channel)
        'fs_had_tau_1_JetTrackWidth_2b_p1pt1': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29],
        'fs_had_tau_1_JetTrackWidth_2b_p1pt2': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.22, 0.23, 0.24],
        'fs_had_tau_1_JetTrackWidth_2b_p1pt3': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
        'fs_had_tau_1_JetTrackWidth_2b_p3pt1': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29],
        'fs_had_tau_1_JetTrackWidth_2b_p3pt2': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
        'fs_had_tau_1_JetTrackWidth_2b_p3pt3': [0.0, 0.01, 0.04, 0.09, 0.1, 0.12, 0.13, 0.15, 0.17],

        ### Optimized binning (dileptau OS 1b channel)
        'fs_had_tau_1_JetTrackWidth_1b_p1pt1_opt': [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28],
        'fs_had_tau_1_JetTrackWidth_1b_p1pt2_opt': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24],
        'fs_had_tau_1_JetTrackWidth_1b_p1pt3_opt': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.11, 0.16],
        'fs_had_tau_1_JetTrackWidth_1b_p3pt1_opt': [0.0, 0.01, 0.02, 0.03, 0.04, 0.07, 0.08, 0.1, 0.11, 0.12, 0.15, 0.16, 0.18, 0.19, 0.21, 0.22, 0.23, 0.25, 0.26, 0.27, 0.28, 0.29],
        'fs_had_tau_1_JetTrackWidth_1b_p3pt2_opt': [0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.09, 0.11, 0.13, 0.14, 0.15, 0.16, 0.17],
        'fs_had_tau_1_JetTrackWidth_1b_p3pt3_opt': [0.0, 0.01, 0.03, 0.05, 0.06, 0.07, 0.09, 0.1, 0.11, 0.14, 0.15, 0.17],
        ### Optimized binning (dileptau OS 2b channel)
        'fs_had_tau_1_JetTrackWidth_2b_p1pt1_opt': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29],
        'fs_had_tau_1_JetTrackWidth_2b_p1pt2_opt': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.22, 0.23, 0.24],
        'fs_had_tau_1_JetTrackWidth_2b_p1pt3_opt': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
        'fs_had_tau_1_JetTrackWidth_2b_p3pt1_opt': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29],
        'fs_had_tau_1_JetTrackWidth_2b_p3pt2_opt': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
        'fs_had_tau_1_JetTrackWidth_2b_p3pt3_opt': [0.0, 0.01, 0.04, 0.09, 0.1, 0.12, 0.13, 0.15, 0.17],

        ### Light leptons
        'pt_lep1': [6,0,120],
        'pt_lep2': [5,0,100],
        'eta_lep1': [6,0,2.4], # 6 bins
        'eta_lep2': [6,0,2.4],
        'TruthIFF_Class_lep1': [11,-0.5,10.5],
        'TruthIFF_Class_lep2': [11,-0.5,10.5],
        'TempCombinationsEncoded': [5,0,5],

        ### Jets
        'pt_jet1': [20,20,120],
        'pt_jet2': [16,20,100],
        'phi_jet1': [10,-3.14,3.14],
        'eta_jet1': [10,-2.5,2.5],

        ### MET
        'm_met': [30,0,150],

        'fs_had_tau_1_RNNScore_1b_p1pt1_opt': [30,0,150],
        'fs_had_tau_1_RNNScore_1b_p1pt2_opt': [30,0,150],
        'fs_had_tau_1_RNNScore_1b_p1pt3_opt': [30,0,150],
        'fs_had_tau_1_RNNScore_1b_p3pt1_opt': [30,0,150],
        'fs_had_tau_1_RNNScore_1b_p3pt2_opt': [30,0,150],
        'fs_had_tau_1_RNNScore_1b_p3pt3_opt': [30,0,150],
        'fs_had_tau_1_RNNScore_2b_p1pt1_opt': [30,0,150],
        'fs_had_tau_1_RNNScore_2b_p1pt2_opt': [30,0,150],
        'fs_had_tau_1_RNNScore_2b_p1pt3_opt': [30,0,150],
        'fs_had_tau_1_RNNScore_2b_p3pt1_opt': [30,0,150],
        'fs_had_tau_1_RNNScore_2b_p3pt2_opt': [30,0,150],
        'fs_had_tau_1_RNNScore_2b_p3pt3_opt': [30,0,150],
        'fs_had_tau_1_RNNScore_1b_p1pt1': [30,0,150],
        'fs_had_tau_1_RNNScore_1b_p1pt2': [30,0,150],
        'fs_had_tau_1_RNNScore_1b_p1pt3': [30,0,150],
        'fs_had_tau_1_RNNScore_1b_p3pt1': [30,0,150],
        'fs_had_tau_1_RNNScore_1b_p3pt2': [30,0,150],
        'fs_had_tau_1_RNNScore_1b_p3pt3': [30,0,150],
        'fs_had_tau_1_RNNScore_2b_p1pt1': [30,0,150],
        'fs_had_tau_1_RNNScore_2b_p1pt2': [30,0,150],
        'fs_had_tau_1_RNNScore_2b_p1pt3': [30,0,150],
        'fs_had_tau_1_RNNScore_2b_p3pt1': [30,0,150],
        'fs_had_tau_1_RNNScore_2b_p3pt2': [30,0,150],
        'fs_had_tau_1_RNNScore_2b_p3pt3': [30,0,150],

        ### MVA
        'eta_jf': [10,-5,5],
        'phi_jf': [10,-3.14,3.14],
        'pt_jf':  [10,0,400],
        'M_b_jf': [10,0,500],
        'eta_b':  [10,-2.5,2.5],
        'phi_b':  [10,-3.14,3.14],
        'pt_b':   [10,0,400],
        'MMC_out_1':[10,20,200],
        'HvisEta':[10,-4.5,4.5],
        'HvisPt': [10,0,250],
        'm_sumet': [20,0,1000],
        'TvisEta':[10,-4.5,4.5],
        'TvisMass': [20,0,700],
        'TvisPt': [20,0,500],
        'HT_all': [10,0,1000],
        'lep_Top_eta': [10,-4.5,4.5],
        'lep_Top_pt':  [10,0,250],
        'lep_Top_phi': [10,-3.14,3.14],
        'deltaRTau': [7,0,7],
        'm_phi_met': [10,-3.14,3.14],
        'deltaPhiTau':[10,-3.14,3.14],
        'had_tau_pt': [10,0,150],
        'had_tau_eta': [10,-2.5,2.5],
        'lep_Higgs_pt': [10,0,200],
        'lep_Higgs_eta': [10,-2.5,2.5],
        'lep_Higgs_phi':[10,-3.14,3.14],

        ### ML Scores
        'lgbm_score': [20,0.0,1],

        'lgbm_score_p1pt1': [x*0.05 for x in range(21)],
        'lgbm_score_p1pt2': [x*0.05 for x in range(21)],
        'lgbm_score_p1pt3': [x*0.05 for x in range(21)],
        'lgbm_score_p3pt1': [x*0.05 for x in range(21)],
        'lgbm_score_p3pt2': [x*0.05 for x in range(21)],
        'lgbm_score_p3pt3': [x*0.05 for x in range(21)],

        'lgbm_score_1b_p1pt1_opt': [x*0.05 for x in range(21)],
        'lgbm_score_1b_p1pt2_opt': [x*0.05 for x in range(21)],
        'lgbm_score_1b_p1pt3_opt': [x*0.05 for x in range(21)],
        'lgbm_score_1b_p3pt1_opt': [x*0.05 for x in range(21)],
        'lgbm_score_1b_p3pt2_opt': [x*0.05 for x in range(21)],
        'lgbm_score_1b_p3pt3_opt': [x*0.05 for x in range(21)],

        'lgbm_score_2b_p1pt1_opt': [x*0.05 for x in range(21)],
        'lgbm_score_2b_p1pt2_opt': [x*0.05 for x in range(21)],
        'lgbm_score_2b_p1pt3_opt': [x*0.05 for x in range(21)],
        'lgbm_score_2b_p3pt1_opt': [x*0.05 for x in range(21)],
        'lgbm_score_2b_p3pt2_opt': [x*0.05 for x in range(21)],
        'lgbm_score_2b_p3pt3_opt': [x*0.05 for x in range(21)],

        'class_pred':   [3,0,3],
        'tH_prob':      [10,0,1],
        'tZ_prob':      [10,0,1],
        'others_prob':  [10,0,1],
        # 'tH_LHscore':   [12, -3.0, 3.4],
        # 'tH_score':     [15, -0.5, 0.9],
        # 'tZ_LHscore':   [12, -3.0, 3.4],
        # 'tZ_score':     [15, -0.5, 0.9],
        'tH_LHscore':   [12, -3.0, 4.1],
        'tH_score':     [15, -0.5, 1.0],
        'tZ_LHscore':   [12, -3.0, 4.1],
        'tZ_score':     [15, -0.5, 1.0],

        'NNout_tauFakes': [20,0,1],
        'NNout_tauFakes_p1pt1': [20,0.0,1],
        'NNout_tauFakes_p1pt2': [20,0.0,1],
        'NNout_tauFakes_p1pt3': [20,0.0,1],
        'NNout_tauFakes_p3pt1': [20,0.0,1],
        'NNout_tauFakes_p3pt2': [20,0.0,1],
        'NNout_tauFakes_p3pt3': [20,0.0,1],
    }
    return dict_hists

def Configurate_Xtitles():
    dict_hists = {
        'm_njets':  'Jet multiplicity',
        'm_nbjets': 'b-jet multiplicity',

        ### Tau
        'fs_had_tau_1_pt': 'p_{T} (#tau_{had}) [GeV]',
        'fs_had_tau_2_pt': 'Subleading #tau_{had} p_{T} [GeV]',
        'fs_had_tau_1_eta': '|#eta (#tau_{had})|',
        'fs_had_tau_2_eta': 'Subleading #tau_{had} |#eta|',
        'fs_had_tau_1_tight': '#tau_{had} pass Medium RNN',
        'fs_had_tau_2_tight': 'Subleading #tau_{had} pass Medium RNN',
        'fs_had_tau_1_RNNScore': 'Leading #tau_{had} RNN score',
        'fs_had_tau_2_RNNScore': 'Subleading #tau_{had} RNN score',
        'fs_had_tau_true_pdg': 'Truth lepton PDG Id',
        'fs_had_tau_2_true_pdg': 'Subleading #tau_{had} PDG Id',
        'fs_had_tau_true_partonTruthLabelID': 'partonTruthLabelID',
        'fs_had_tau_2_true_partonTruthLabelID': 'Subleading #tau_{had} partonTruthLabelID',
        'fs_had_tau_1_nTrack': '# of tracks in #tau_{had}',
        'fs_had_tau_2_nTrack': '# of tracks in #tau_{had}',
        'fs_had_tau_1_JetTrackWidth': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_2_JetTrackWidth': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetCaloWidth': 'JetCaloWidth (#tau_{had})',
        'fs_had_tau_2_JetCaloWidth': 'JetCaloWidth (#tau_{had})',


        'fs_had_tau_1_JetTrackWidth_p1pt1': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_p1pt2': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_p1pt3': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_p3pt1': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_p3pt2': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_p3pt3': 'JetTrackWidth (#tau_{had})',

        'fs_had_tau_1_JetTrackWidth_1b_p1pt1': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_1b_p1pt2': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_1b_p1pt3': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_1b_p3pt1': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_1b_p3pt2': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_1b_p3pt3': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_2b_p1pt1': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_2b_p1pt2': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_2b_p1pt3': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_2b_p3pt1': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_2b_p3pt2': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_2b_p3pt3': 'JetTrackWidth (#tau_{had})',

        ### Optimized binning (dileptau channel) -> new samples
        'fs_had_tau_1_JetTrackWidth_p1_pt1_pf': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_p1_pt2_pf': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_p1_pt3_pf': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_p3_pt1_pf': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_p3_pt2_pf': 'JetTrackWidth (#tau_{had})',
        'fs_had_tau_1_JetTrackWidth_p3_pt3_pf': 'JetTrackWidth (#tau_{had})',

        'lgbm_score_1b_p1pt1': 'LGBM score',
        'lgbm_score_1b_p1pt2': 'LGBM score',
        'lgbm_score_1b_p1pt3': 'LGBM score',
        'lgbm_score_1b_p3pt1': 'LGBM score',
        'lgbm_score_1b_p3pt2': 'LGBM score',
        'lgbm_score_1b_p3pt3': 'LGBM score',

        'lgbm_score_2b_p1pt1': 'LGBM score',
        'lgbm_score_2b_p1pt2': 'LGBM score',
        'lgbm_score_2b_p1pt3': 'LGBM score',
        'lgbm_score_2b_p3pt1': 'LGBM score',
        'lgbm_score_2b_p3pt2': 'LGBM score',
        'lgbm_score_2b_p3pt3': 'LGBM score',

        'fs_had_tau_1_RNNScore_1b_p1pt1': 'RNNScore',
        'fs_had_tau_1_RNNScore_1b_p1pt2': 'RNNScore',
        'fs_had_tau_1_RNNScore_1b_p1pt3': 'RNNScore',
        'fs_had_tau_1_RNNScore_1b_p3pt1': 'RNNScore',
        'fs_had_tau_1_RNNScore_1b_p3pt2': 'RNNScore',
        'fs_had_tau_1_RNNScore_1b_p3pt3': 'RNNScore',
        'fs_had_tau_1_RNNScore_2b_p1pt1': 'RNNScore',
        'fs_had_tau_1_RNNScore_2b_p1pt2': 'RNNScore',
        'fs_had_tau_1_RNNScore_2b_p1pt3': 'RNNScore',
        'fs_had_tau_1_RNNScore_2b_p3pt1': 'RNNScore',
        'fs_had_tau_1_RNNScore_2b_p3pt2': 'RNNScore',
        'fs_had_tau_1_RNNScore_2b_p3pt3': 'RNNScore',


        ### Light leptons
        'pt_lep1': 'Leading lepton p_{T}',
        'pt_lep2': 'Subleading lepton p_{T}',
        'eta_lep1': 'Leading lepton |#eta|',
        'eta_lep2': 'Subleading lepton |#eta|',
        'TruthIFF_Class_lep1': 'Leading lepton Truth IFF class',
        'TruthIFF_Class_lep2': 'Subleading lepton Truth IFF class',
        'TempCombinationsEncoded': '',

        ### Jets
        'pt_jet1': 'Leading jet p_{T}',
        'pt_jet2': 'Sub-leading jet p_{T}',
        'phi_jet1': 'Leading jet #phi',
        'eta_jet1': 'Leading jet #eta',

        ### MET
        'm_met': 'MET [GeV]',

        ### MVA
        'eta_jf': 'eta_jf',
        'phi_jf': 'phi_jf',
        'pt_jf':  'pt_jf',
        'M_b_jf': 'M_b_jf',
        'eta_b':  'eta_b',
        'phi_b':  'phi_b',
        'pt_b':   'pt_b',
        'MMC_out_1':'MMC_out_1',
        'HvisEta':'HvisEta',
        'HvisPt': 'HvisPt',
        'm_sumet': 'm_sumet',
        'TvisEta':'TvisEta',
        'TvisMass': 'TvisMass',
        'TvisPt': 'TvisPt',
        'HT_all': 'HT_all',
        'lep_Top_eta': 'lep_Top_eta',
        'lep_Top_pt':  'lep_Top_pt',
        'lep_Top_phi': 'lep_Top_phi',
        'deltaRTau': 'deltaRTau',
        'm_phi_met': 'm_phi_met',
        'deltaPhiTau':'deltaPhiTau',
        'had_tau_pt': 'had_tau_pt',
        'had_tau_eta': 'had_tau_eta',
        'lep_Higgs_pt': 'lep_Higgs_pt',
        'lep_Higgs_eta': 'lep_Higgs_eta',
        'lep_Higgs_phi': 'lep_Higgs_phi',

        ### Scores
        'lgbm_score': 'LGBM score',
        'NNout_tauFakes': 'NN score',

        'class_pred':   'class labels',
        'tH_prob':      'tH score',
        'tZ_prob':      'tZ score',
        'others_prob':  'others score',
        'tH_LHscore':   'log[P(tH)/(f*P(tZ)+(1-f)*P(others))]',
        'tH_score':     'P(tH) - [P(tZ)+P(others)]/2',
        'tZ_LHscore':   'log[P(tZ)/(f*P(tH)+(1-f)*P(others))]',
        'tZ_score':     'P(tZ) - [P(tH)+P(others)]/2',
    }
    return dict_hists

def get_fakesSFs_default():
    fakesSFs = {}

    ############################################################################
    ####                    1b / 1-Tack
    ############################################################################
    ## 1b / 1-Tack / gluon-jet / pt1
    fakesSFs['1b_1p_gjet_pt1_nominal'] = 0.611
    fakesSFs['1b_1p_gjet_pt1_shape']   = 0.297
    fakesSFs['1b_1p_gjet_pt1_norm']    = 0.050
    ## 1b / 1-Tack / gluon-jet / pt2
    fakesSFs['1b_1p_gjet_pt2_nominal'] = 0.993
    fakesSFs['1b_1p_gjet_pt2_shape']   = 0.473
    fakesSFs['1b_1p_gjet_pt2_norm']    = 0.057
    ## 1b / 1-Tack / gluon-jet / pt3
    fakesSFs['1b_1p_gjet_pt3_nominal'] = 1.122
    fakesSFs['1b_1p_gjet_pt3_shape']   = 0.977
    fakesSFs['1b_1p_gjet_pt3_norm']    = 0.042
    ############################################################################
    ## 1b / 1-Tack / quark-jet / pt1
    fakesSFs['1b_1p_qjet_pt1_nominal'] = 1.360
    fakesSFs['1b_1p_qjet_pt1_shape']   = 0.106
    fakesSFs['1b_1p_qjet_pt1_norm']    = 0.050
    ## 1b / 1-Tack / quark-jet / pt2
    fakesSFs['1b_1p_qjet_pt2_nominal'] = 1.058
    fakesSFs['1b_1p_qjet_pt2_shape']   = 0.103
    fakesSFs['1b_1p_qjet_pt2_norm']    = 0.057
    ## 1b / 1-Tack / quark-jet / pt3
    fakesSFs['1b_1p_qjet_pt3_nominal'] = 0.909
    fakesSFs['1b_1p_qjet_pt3_shape']   = 0.106
    fakesSFs['1b_1p_qjet_pt3_norm']    = 0.042
    ############################################################################
    ## 1b / 1-Tack / unknown / pt1
    fakesSFs['1b_1p_unknown_pt1_nominal'] = 0.958
    fakesSFs['1b_1p_unknown_pt2_nominal'] = 0.848
    fakesSFs['1b_1p_unknown_pt3_nominal'] = 0.953
    
    ############################################################################
    ####                    1b / 3-Tack
    ############################################################################
    ## 1b / 3-Tack / gluon-jet / pt1
    fakesSFs['1b_3p_gjet_pt1_nominal'] = 0.719
    fakesSFs['1b_3p_gjet_pt1_shape']   = 0.439
    fakesSFs['1b_3p_gjet_pt1_norm']    = 0.050
    ## 1b / 3-Tack / gluon-jet / pt2
    fakesSFs['1b_3p_gjet_pt2_nominal'] = 0.605
    fakesSFs['1b_3p_gjet_pt2_shape']   = 0.455
    fakesSFs['1b_3p_gjet_pt2_norm']    = 0.062
    ## 1b / 3-Tack / gluon-jet / pt3
    fakesSFs['1b_3p_gjet_pt3_nominal'] = 0.841
    fakesSFs['1b_3p_gjet_pt3_shape']   = 0.784
    fakesSFs['1b_3p_gjet_pt3_norm']    = 0.062
    ############################################################################
    ## 1b / 3-Tack / quark-jet / pt1
    fakesSFs['1b_3p_qjet_pt1_nominal'] = 1.159
    fakesSFs['1b_3p_qjet_pt1_shape']   = 0.097
    fakesSFs['1b_3p_qjet_pt1_norm']    = 0.050
    ## 1b / 3-Tack / quark-jet / pt2
    fakesSFs['1b_3p_qjet_pt2_nominal'] = 0.932
    fakesSFs['1b_3p_qjet_pt2_shape']   = 0.070
    fakesSFs['1b_3p_qjet_pt2_norm']    = 0.062
    ## 1b / 3-Tack / quark-jet / pt3
    fakesSFs['1b_3p_qjet_pt3_nominal'] = 0.897
    fakesSFs['1b_3p_qjet_pt3_shape']   = 0.056
    fakesSFs['1b_3p_qjet_pt3_norm']    = 0.062
    ############################################################################
    ## 1b / 3-Tack / unknown / pt1
    fakesSFs['1b_3p_unknown_pt1_nominal'] = 0.925
    fakesSFs['1b_3p_unknown_pt2_nominal'] = 1.397
    fakesSFs['1b_3p_unknown_pt3_nominal'] = 2.696

    ############################################################################
    ####                    2b / 1-Tack
    ############################################################################
    ## 2b / 1-Tack / gluon-jet / pt1
    fakesSFs['2b_1p_gjet_pt1_nominal'] = 1.747
    fakesSFs['2b_1p_gjet_pt1_shape']   = 0.234
    fakesSFs['2b_1p_gjet_pt1_norm']    = 0.059
    ## 2b / 1-Tack / gluon-jet / pt2
    fakesSFs['2b_1p_gjet_pt2_nominal'] = 1.560
    fakesSFs['2b_1p_gjet_pt2_shape']   = 0.596
    fakesSFs['2b_1p_gjet_pt2_norm']    = 0.096
    ## 2b / 1-Tack / gluon-jet / pt3
    fakesSFs['2b_1p_gjet_pt3_nominal'] = 0.918
    fakesSFs['2b_1p_gjet_pt3_shape']   = 0.680
    fakesSFs['2b_1p_gjet_pt3_norm']    = 0.059
    ############################################################################
    ## 2b / 1-Tack / quark-jet / pt1
    fakesSFs['2b_1p_qjet_pt1_nominal'] = 0.810
    fakesSFs['2b_1p_qjet_pt1_shape']   = 0.102
    fakesSFs['2b_1p_qjet_pt1_norm']    = 0.059
    ## 2b / 1-Tack / quark-jet / pt2
    fakesSFs['2b_1p_qjet_pt2_nominal'] = 0.856
    fakesSFs['2b_1p_qjet_pt2_shape']   = 0.157
    fakesSFs['2b_1p_qjet_pt2_norm']    = 0.096
    ## 2b / 1-Tack / quark-jet / pt3
    fakesSFs['2b_1p_qjet_pt3_nominal'] = 0.848
    fakesSFs['2b_1p_qjet_pt3_shape']   = 0.081
    fakesSFs['2b_1p_qjet_pt3_norm']    = 0.059
    ############################################################################
    ## 2b / 1-Tack / unknown / pt1
    fakesSFs['2b_1p_unknown_pt1_nominal'] = 1.0
    fakesSFs['2b_1p_unknown_pt2_nominal'] = 1.0
    fakesSFs['2b_1p_unknown_pt3_nominal'] = 1.0

    ############################################################################
    ####                    2b / 3-Tack
    ############################################################################
    ## 2b / 3-Tack / gluon-jet / pt1
    fakesSFs['2b_3p_gjet_pt1_nominal'] = 1.623
    fakesSFs['2b_3p_gjet_pt1_shape']   = 0.331
    fakesSFs['2b_3p_gjet_pt1_norm']    = 0.061
    ## 2b / 3-Tack / gluon-jet / pt2
    fakesSFs['2b_3p_gjet_pt2_nominal'] = 1.696
    fakesSFs['2b_3p_gjet_pt2_shape']   = 0.556
    fakesSFs['2b_3p_gjet_pt2_norm']    = 0.066
    ## 2b / 3-Tack / gluon-jet / pt3
    fakesSFs['2b_3p_gjet_pt3_nominal'] = 1.168
    fakesSFs['2b_3p_gjet_pt3_shape']   = 1.168
    fakesSFs['2b_3p_gjet_pt3_norm']    = 0.070
    ############################################################################
    ## 2b / 3-Tack / quark-jet / pt1
    fakesSFs['2b_3p_qjet_pt1_nominal'] = 0.996
    fakesSFs['2b_3p_qjet_pt1_shape']   = 0.063
    fakesSFs['2b_3p_qjet_pt1_norm']    = 0.061
    ## 2b / 3-Tack / quark-jet / pt2
    fakesSFs['2b_3p_qjet_pt2_nominal'] = 0.743
    fakesSFs['2b_3p_qjet_pt2_shape']   = 0.065
    fakesSFs['2b_3p_qjet_pt2_norm']    = 0.066
    ## 2b / 3-Tack / quark-jet / pt3
    fakesSFs['2b_3p_qjet_pt3_nominal'] = 0.897
    fakesSFs['2b_3p_qjet_pt3_shape']   = 0.073
    fakesSFs['2b_3p_qjet_pt3_norm']    = 0.070
    ############################################################################
    ## 2b / 3-Tack / unknown / pt1
    fakesSFs['2b_3p_unknown_pt1_nominal'] = 1.0
    fakesSFs['2b_3p_unknown_pt2_nominal'] = 1.0
    fakesSFs['2b_3p_unknown_pt3_nominal'] = 1.0

    ############################################################################
    ####                    1Bin/Counting
    ############################################################################
    fakesSFs['Counting_1b_1p_pt1_nominal'] = 1.154
    fakesSFs['Counting_1b_1p_pt1_norm']    = 0.050
    fakesSFs['Counting_1b_1p_pt2_nominal'] = 1.025
    fakesSFs['Counting_1b_1p_pt2_norm']    = 0.057
    fakesSFs['Counting_1b_1p_pt3_nominal'] = 0.926
    fakesSFs['Counting_1b_1p_pt3_norm']    = 0.042

    fakesSFs['Counting_1b_3p_pt1_nominal'] = 1.077
    fakesSFs['Counting_1b_3p_pt1_norm']    = 0.050
    fakesSFs['Counting_1b_3p_pt2_nominal'] = 0.904
    fakesSFs['Counting_1b_3p_pt2_norm']    = 0.062
    fakesSFs['Counting_1b_3p_pt3_nominal'] = 0.902
    fakesSFs['Counting_1b_3p_pt3_norm']    = 0.062

    fakesSFs['Counting_2b_1p_pt1_nominal'] = 1.095
    fakesSFs['Counting_2b_1p_pt1_norm']    = 0.059
    fakesSFs['Counting_2b_1p_pt2_nominal'] = 1.002
    fakesSFs['Counting_2b_1p_pt2_norm']    = 0.096
    fakesSFs['Counting_2b_1p_pt3_nominal'] = 0.856
    fakesSFs['Counting_2b_1p_pt3_norm']    = 0.059

    fakesSFs['Counting_2b_3p_pt1_nominal'] = 1.096
    fakesSFs['Counting_2b_3p_pt1_norm']    = 0.061
    fakesSFs['Counting_2b_3p_pt2_nominal'] = 0.843
    fakesSFs['Counting_2b_3p_pt2_norm']    = 0.066
    fakesSFs['Counting_2b_3p_pt3_nominal'] = 0.913
    fakesSFs['Counting_2b_3p_pt3_norm']    = 0.07

    return fakesSFs