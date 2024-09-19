import ROOT
import logging
import math
import ctypes
import logging
from array import array

from LFAHelpers import style
from LFAHelpers import compute_last_fraction

# Loading RooFit macros
ROOT.gROOT.LoadMacro("LFAFitter.C+")
from ROOT import LFA_TemplatesFitter

def TauFakeYieldCorrector(hists, unknown_is_jet=True, mymsg=None):

    # dummy initialization
    sig = ROOT.TH1D( hists[1].Clone("mc") )

    # dictionary for results
    result = {}

    if mymsg: logging.info(mymsg)

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
                logging.debug('Tau hist integral = {}'.format(hist.Integral()))
            if '_unknown_' in hist.GetName():
                logging.debug('Unknown hist integral = {}'.format(hist.Integral()))
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

    # Fill the results dictionary
    result['N_data'] = round(Ndata)
    result['N_sig']  = round(Nsig, 1)
    result['N_unknown']  = round(Nunknown, 1)
    if unknown_is_jet==False: result['N_sig'] = round(Nsig - Nunknown, 1)
    result['N_jets']  = round(Njet, 1)

    # Subtract sig from data
    Nbkg = Ndata - Nsig
    Nbkg_err = math.sqrt(Ndata_err.value**2 + Nsig_err.value**2)
    result['N_jets_est']  = round(Nbkg, 1)
    result['N_jets_est_err'] = round(Nbkg_err, 1)

    # Compute normalization factor Nbkg/Njet
    NF = Nbkg/Njet
    NF_err = NF*math.sqrt( (Nbkg_err/Nbkg)**2 + (Njet_err.value/Njet)**2 )
    result['SF_jets_est'] = round(NF, 3)
    result['SF_jets_err_est'] = round(NF_err, 3)

    logging.debug("   N_data = {}".format(result['N_data']))
    logging.debug("   N_sig = {}".format(result['N_sig']))
    logging.debug("   N_unknown = {}".format(result['N_unknown']))
    logging.debug("   N_jets = {}".format(result['N_jets']))
    logging.debug("   N_jets estimated = {x} \u00B1 {dx}".format(x=result['N_jets_est'], dx=result["N_jets_est_err"]))
    logging.info("     Estimated jet SFs = {x} \u00B1 {dx}".format(x=result['SF_jets_est'], dx=result["SF_jets_err_est"]))

    return result

# Template fit to JetTrackWidth distribution
def Fit_Tau(hists, unknown_is_jet=False, BKGonlyFit=True, DEBUG=False, mymsg=None):

    # dictionary for results
    result = {}

    if mymsg: logging.info(mymsg)

    # dummy initialization
    sig = ROOT.TH1D( hists[1].Clone("mc") )

    data, qjet, gjet = None, None, None
    is_sig_started = False
    N_underflow_unknown = 0.0
    N_underflow_data = 0.0
    N_rest = 0.0
    for ihist, hist in enumerate(hists):

        # Build template histograms
        if '_data_' in hist.GetName():
            data = ROOT.TH1D( hist.Clone("data") )
        elif '_q-jet_' in hist.GetName():
            qjet = ROOT.TH1D( hist.Clone("qjet") )
        elif '_g-jet_' in hist.GetName():
            gjet = ROOT.TH1D( hist.Clone("gjet") )
        else:
            if not is_sig_started:
                sig = ROOT.TH1D( hist.Clone("mc") )
                is_sig_started = True
            else:
                sig.Add(hist)

        # Find inefficiency scale factor for unknown template
        if '_data_' in hist.GetName():
            N_underflow_data = hist.GetBinContent(0)
        elif '_unknown_' in hist.GetName():
            N_underflow_unknown = hist.GetBinContent(0)
        elif '_q-jet_' in hist.GetName() or '_g-jet_' in hist.GetName():
            N_rest += hist.GetBinContent(0)
        
        
    # SF for unknown
    SF_unknown = 1.0
    if N_underflow_unknown>0:
        # ToDo: subtract N_rest from data
        SF_unknown = N_underflow_data/N_underflow_unknown

    ### Template Fit
    gjet.SetTitle("gjet")
    qjet.SetTitle("qjet")
    data.SetTitle("data")
    sig.SetTitle("sig")

    vect_bkgs = ROOT.std.vector('TH1D')()
    vect_bkgs.push_back(qjet)
    vect_bkgs.push_back(gjet)

    fsig_val, fsig_err = ctypes.c_double(0.), ctypes.c_double(0.)
    fbkg_val_pairs = LFA_TemplatesFitter(data, sig, vect_bkgs, fsig_val, fsig_err, ctypes.c_bool(BKGonlyFit), ctypes.c_bool(DEBUG)) # vector containing vect_bkgs.size()-1 pairs of background fractions <val, err> from the template fit

    fbkg_vals, fbkg_errs = [fsig_val.value], [fsig_err.value]

    # initial fractions
    IntData = data.Integral()
    fbkg_init_vals = [sig.Integral()/IntData]
    fbkg_init_vals.append(qjet.Integral()/IntData)
    fbkg_init_vals.append(gjet.Integral()/IntData)

    # fractions from the fit: fbkg*(1-fsig)
    fbkg_vals.append( fbkg_val_pairs[0].first*(1-fsig_val.value) )
    fbkg_errs.append( fbkg_val_pairs[0].second*(1-fsig_val.value) )

    fbkg_val_last, fbkg_err_last = compute_last_fraction(fbkg_vals, fbkg_errs)
    fbkg_vals.append(fbkg_val_last)
    fbkg_errs.append(fbkg_err_last)

    if DEBUG:
        logging.info('frac val: {}'.format(fbkg_vals))
        logging.info('frac err: {}'.format(fbkg_errs))

    # Store results
    result['f_sig'] = round(fbkg_vals[0], 2)
    result['f_q'] = round(fbkg_init_vals[1], 2)
    result['f_q_est'] = round(fbkg_vals[1], 2)
    result['f_q_err_est'] = round(fbkg_errs[1], 2)
    result['f_g_est'] = round(fbkg_vals[2], 2)
    result['f_g_err_est'] = round(fbkg_errs[2], 2)
    result['SF_unknown_est'] = round(SF_unknown, 3)

    # Convert fractions into norm factors
    for i, fval in enumerate(fbkg_vals):
        fbkg_vals[i] = fval/fbkg_init_vals[i]
        fbkg_errs[i] = fbkg_errs[i]/fbkg_init_vals[i]


    result['SF_q_est'] = round(fbkg_vals[1], 3)
    result['SF_q_err_est'] = round(fbkg_errs[1], 3)
    result['SF_g_est'] = round(fbkg_vals[2], 3)
    result['SF_g_err_est'] = round(abs(fbkg_errs[2]), 3)

    if mymsg:
        logging.debug("    f_sig = {} ".format(result['f_sig']))
        logging.debug("    Initial f_q = {} ".format(result['f_q']))
        logging.debug("    Estimated f_q = {x} \u00B1 {dx}".format(x=result['f_q_est'], dx=result["f_q_err_est"]))
        logging.debug("    Estimated f_g = {x} \u00B1 {dx}".format(x=result['f_g_est'], dx=result["f_g_err_est"]))
        logging.info("     Estimated q-jet SF = {x} \u00B1 {dx}".format(x=result['SF_q_est'], dx=result["SF_q_err_est"]))
        logging.info("     Estimated g-jet SF = {x} \u00B1 {dx}".format(x=result['SF_g_est'], dx=result["SF_g_err_est"]))
        logging.info("     Estimated uknown SF = {}".format(result['SF_unknown_est']))

    return result

def Fit_DiTau(data_df, hists, dict_tmps_comb, BKGonlyFit=True, DEBUG=False, mymsg=None):

    # dictionary for results
    result = {}

    if mymsg: logging.info(mymsg)

    vect_bkgs = ROOT.std.vector('TH1D')()
    hdata, hsig = None, None
    for idx, hist in enumerate(hists):
        temp_name = dict_tmps_comb['sample'][idx].replace(',', '_')
        hist.SetTitle(temp_name)
        if 'jet' in temp_name:
            vect_bkgs.push_back(hist)
        if 'jet' not in temp_name and 'data' not in temp_name:
            hsig = hist
        if 'data' in temp_name:
            hdata = hist

    fsig_val, fsig_err = ctypes.c_double(0.), ctypes.c_double(0.)
    # vector containing vect_bkgs.size()-1 pairs of background fractions <val, err> from the template fit
    fbkg_val_pairs = LFA_TemplatesFitter(hdata, hsig, vect_bkgs, fsig_val, fsig_err, ctypes.c_bool(BKGonlyFit), ctypes.c_bool(DEBUG))

    fbkg_vals, fbkg_errs = [fsig_val.value], [fsig_err.value]
    IntData = hdata.Integral()
    fbkg_init_vals = [hsig.Integral()/IntData]

    # Store data yields
    regions = ['TTbar', 'TbarT', 'TbarTbar']

    # Store yields as aux data
    result['aux'] = {}
    for index, region_tag in enumerate(regions):
        result['aux']['N_data_'+region_tag] = round(hdata.GetBinContent(index+1))
        result['aux']['N_simsim_'+region_tag] = round(hsig.GetBinContent(index+1), 1)
        result['aux']['N_simjet_'+region_tag] = round(vect_bkgs.at(0).GetBinContent(index+1), 1)
        result['aux']['N_jetsim_'+region_tag] = round(vect_bkgs.at(1).GetBinContent(index+1), 1)
        result['aux']['N_jetjet_'+region_tag] = round(vect_bkgs.at(2).GetBinContent(index+1), 1)

    for idx, hist in enumerate(hists):
        if 'jet' in dict_tmps_comb['sample'][idx] and idx<fbkg_val_pairs.size()+2:
            fbkg_vals.append(fbkg_val_pairs[idx-2].first*(1-fsig_val.value))
            fbkg_errs.append(fbkg_val_pairs[idx-2].second*(1-fsig_val.value))
            IntBkg = hists[idx].Integral()
            fbkg_init_vals.append(IntBkg/IntData)

    fbkg_val_last, fbkg_err_last = compute_last_fraction(fbkg_vals, fbkg_errs)
    fbkg_vals.append(fbkg_val_last)
    fbkg_errs.append(fbkg_err_last)
    fbkg_init_vals.append(hists[-1].Integral()/IntData)

    if DEBUG:
        logging.info('frac val: {}'.format(fbkg_vals))
        logging.info('frac err: {}'.format(fbkg_errs))

    # Store results
    result['f_sig'] = round(fbkg_vals[0], 2)
    result['f_jetsim'] = round(fbkg_init_vals[1], 2)
    result['f_jetsim_est'] = round(fbkg_vals[1], 2)
    result['f_jetsim_err_est'] = round(fbkg_errs[1], 2)
    result['f_simjet'] = round(fbkg_init_vals[2], 2)
    result['f_simjet_est'] = round(fbkg_vals[2], 2)
    result['f_simjet_err_est'] = round(fbkg_errs[2], 2)

    result['f_jetjet_est'] = round(fbkg_vals[3], 2)
    result['f_jetjet_err_est'] = round(fbkg_errs[3], 2)


    # Convert fractions into norm factors
    for i, fval in enumerate(fbkg_vals):
        fbkg_vals[i] = fval/fbkg_init_vals[i]
        fbkg_errs[i] = fbkg_errs[i]/fbkg_init_vals[i]


    result['SF_simjet_est'] = round(fbkg_vals[1], 3)
    result['SF_simjet_err_est'] = round(fbkg_errs[1], 3)
    result['SF_jetsim_est'] = round(fbkg_vals[2], 3)
    result['SF_jetsim_err_est'] = round(fbkg_errs[2], 3)
    result['SF_jetjet_est'] = round(fbkg_vals[3], 3)
    result['SF_jetjet_err_est'] = round(abs(fbkg_errs[3]), 3)

    logging.info("     Estimated simjet SF = {x} \u00B1 {dx}".format(x=result['SF_simjet_est'], dx=result["SF_simjet_err_est"]))
    logging.info("     Estimated jetsim SF = {x} \u00B1 {dx}".format(x=result['SF_jetsim_est'], dx=result["SF_jetsim_err_est"]))
    logging.info("     Estimated jetjet SF = {x} \u00B1 {dx}".format(x=result['SF_jetjet_est'], dx=result["SF_jetjet_err_est"]))

    #return BKGCorrector(data_df, 'TempCombinations', dict_tmps_comb, fbkg_vals, fbkg_errs)

    return result

####################################################################
###                      LATEX functions                         ###
####################################################################
def print_to_latex_start(file, method):
    if method=='Counting':
        file.write("\\begin{table}[htbp]\n")
        file.write("  \\tablesetup\n")
        file.write("  \\sisetup{table-align-uncertainty}\n")
        file.write("  \\caption{Measured jet faking tau background in the region with the tau leptons failing Medium ID requirement in \dileptau OS channel.}\n")
        file.write("  \\label{tab:taufakes_yields_method}\n")
        file.write("  \\begin{tabular}{l | S[table-format=4.0]S[table-format=3.1]S[table-format=4.1] |\n    S[table-format=4.1]\n    S[table-format=4.1(1)]\n    S[table-format=1.3(1)]}\n")
        file.write("    \\toprule\n")
        file.write("    \\pT(\\tauhad)  & $N_{\\text{data}}$ & $N_{\\text{MC}}^{\\tau+e+\\mu}$\n    & {$N_{\\text{MC}}^{\\text{unknown}}$} & {$N_{\\text{MC}}^{q+g}$}\n    & {$N^{q+g}$ est.} & {\\SF($q+g$)}\\\\\n")
        file.write("    {[\\si{\\GeV}]} & & & & & \\\\\n")
    elif method=='TemplateFit':
        file.write("\\begin{table}[htbp]\n")
        file.write("  \\tablesetup\n")
        file.write("  \\sisetup{table-align-uncertainty, table-format=1.3}\n")
        file.write("  \\caption{Measured jet faking tau background fractions in the \\dileptau channel\n")
        file.write("    with the template fit method. The fits are performed to the \\texttt{JetTrackWidth}\n")
        file.write("    variable in case of 1 b-jet and to BDT-based discriminator for events with 2 b-jets.\n")
        file.write("    control region defined in the text. The quark- and gluon-initiated jet SF uncertainties\n")
        file.write("    are anti-correlated due to the $f_q+f_g=1$ constraint.}\n")
        file.write("  \\label{tab:taufakes_templatefit}\n")
        file.write("  \\begin{tabular}{l | S\n")
        file.write("    S[table-format=1.3(1)]\n")
        file.write("    S[table-format=1.3(1)]\n")
        file.write("    S[table-format=1.3]@{$\\,\\mp\\,$}S[table-format=1.3]}\n")
        file.write("    \\toprule\n")
        file.write("     \\pT(\\tauhad) & {$f_S$ (fixed)} & {$f_q$ (derived)}\n")
        file.write("     & {\\Pq-jet \\SF} & \\multicolumn{2}{c}{\\Pg-jet \SF} \\\\\n")
        file.write("     {[\\si{\\GeV}]} & & & & \\multicolumn{2}{c}{}\\\\\n")

    elif method=='di-lep':
        file.write("\\begin{table}[htbp]\n")
        file.write("  \\tablesetup\n")
        file.write("  \\sisetup{table-format=4.1}\n")
        file.write("  \\caption{Event yields and truth composition in the dilepton background control regions. The scale factors (SFs) are measured\n")
        file.write("    separately for events with 1 and 2 b-jets. The templates with the large SF uncertanty are not corrected in the signal region,\n")
        file.write("    but the uncertainty is assigned to be 100\\%. }\n")
        file.write("  \\label{tab:leppairfakes_templatefit}\n")
        file.write("  \\begin{tabular}{l | S[table-format=4.0] SSSS}\n")
        file.write("    \\toprule\n")
        file.write("    CR name & {$N_{\\text{data}}$} & {$N_{\\text{sim,sim}}$}\n")
        file.write("    & $N_{\\text{sim,jet}}$ & $N_{\\text{jet,sim}}$ & $N_{\\text{jet,jet}}$\\\\\n")


def print_to_latex_values(file, result, method):
    if method=='Counting':
        file.write("    {pt}     & {N_data}  & {N_sig}  & {N_unknown} & {N_jets}  & {N_jets_est} \\pm {N_jets_est_err} & {SF_jets_est} \\pm {SF_jets_err_est} \\\\\n".format(
            pt     = result['tau_pT_range'],
            N_data = result['N_data'],
            N_sig  = result['N_sig'],
            N_unknown = result['N_unknown'],
            N_jets = result['N_jets'],
            N_jets_est = result['N_jets_est'],
            N_jets_est_err = result['N_jets_est_err'],
            SF_jets_est = result['SF_jets_est'],
            SF_jets_err_est = result['SF_jets_err_est'])
        )
    elif method=='TemplateFit':
        file.write("    {pt}   &  {f_sig}  &  {f_q_est} \\pm {f_q_err_est}  &  {SF_q_est} \\pm {SF_q_err_est}  & {SF_g_est} & {SF_g_err_est} \\\\\n".format(
            pt     = result['tau_pT_range'],
            f_sig  = result['f_sig'],
            f_q_est= result['f_q_est'],
            f_q_err_est= result['f_q_err_est'],
            SF_q_est= result['SF_q_est'],
            SF_q_err_est= result['SF_q_err_est'],
            SF_g_est= result['SF_g_est'],
            SF_g_err_est= result['SF_g_err_est'])
        )
    elif method=='di-lep':
            # latex table
            file.write("    \\midrule\n")
            file.write("    $T\bar{{T}}$ ($n_{{b}}={nb}$)        & {N_data}  & {N_simsim} & {N_simjet}  & {N_jetsim}   &  {N_jetjet} \\\\\n".format(
                nb=result['nb'],
                N_data=result['aux']['N_data_TTbar'],
                N_simsim=result['aux']['N_simsim_TTbar'],
                N_simjet=result['aux']['N_simjet_TTbar'],
                N_jetsim=result['aux']['N_jetsim_TTbar'],
                N_jetjet=result['aux']['N_jetjet_TTbar'],
            ))
            file.write("    $\\bar{{T}}T$ ($n_{{b}}={nb}$)        & {N_data}  & {N_simsim} & {N_simjet}  & {N_jetsim}   &  {N_jetjet} \\\\\n".format(
                nb=result['nb'],
                N_data=result['aux']['N_data_TbarT'],
                N_simsim=result['aux']['N_simsim_TbarT'],
                N_simjet=result['aux']['N_simjet_TbarT'],
                N_jetsim=result['aux']['N_jetsim_TbarT'],
                N_jetjet=result['aux']['N_jetjet_TbarT'],
            ))
            file.write("    $\\bar{{T}}\\bar{{T}}$ ($n_{{b}}={nb}$)        & {N_data}  & {N_simsim} & {N_simjet}  & {N_jetsim}   &  {N_jetjet} \\\\\n".format(
                nb=result['nb'],
                N_data=result['aux']['N_data_TbarTbar'],
                N_simsim=result['aux']['N_simsim_TbarTbar'],
                N_simjet=result['aux']['N_simjet_TbarTbar'],
                N_jetsim=result['aux']['N_jetsim_TbarTbar'],
                N_jetjet=result['aux']['N_jetjet_TbarTbar'],
            ))
            file.write("    \\midrule\n")
            file.write("     Measured SF &   &   &  {{${SF_simjet_est} \\pm {SF_simjet_err_est}$}} & {{${SF_jetsim_est} \\pm {SF_jetsim_err_est}$}} & {{${SF_jetjet_est} \\pm {SF_jetjet_err_est}$}} \\\\\n".format(
                SF_simjet_est=result['SF_simjet_est'],
                SF_simjet_err_est=result['SF_simjet_err_est'],
                SF_jetsim_est=result['SF_jetsim_est'],
                SF_jetsim_err_est=result['SF_jetsim_err_est'],
                SF_jetjet_est=result['SF_jetjet_est'],
                SF_jetjet_err_est=result['SF_jetjet_err_est']
            ))


####################################################################
###                      Save to ROOT file                       ###
####################################################################
def result_to_root(results, outfilename):

    outFile = ROOT.TFile.Open(outfilename,"RECREATE")
    if results['name']=='Counting':
        for key in results.keys():
            if 'SFs_' in key:
                xbins = sorted(list(set(results[key]['pT_low'] + results[key]['pT_high'])))
                outhist = ROOT.TH1D(key, key, len(results[key]['pT_low']), array('d', xbins) )
                for ibin, value in enumerate(results[key]['SFs']):
                    outhist.SetBinContent(ibin+1, value)
                    outhist.SetBinError(ibin+1, results[key]['SFs_err'][ibin])
                outhist.Write()

    elif results['name']=='TemplateFit':
        tags = ['q', 'g', 'unknown']
        for key in results.keys():
            if 'SFs_' in key:
                xbins = sorted(list(set(results[key]['pT_low'] + results[key]['pT_high'])))
                for tag in tags:
                    outhist = ROOT.TH1D(key+'_'+tag, key+'_'+tag, len(results[key]['pT_low']), array('d', xbins) )
                    for ibin, value in enumerate(results[key]['SFs_'+tag]):
                        outhist.SetBinContent(ibin+1, value)
                        outhist.SetBinError(ibin+1, results[key]['SFs_'+tag+'_err'][ibin])
                    outhist.Write()

    elif results['name']=='di-tau':
        for key in results.keys():
            if 'SFs_' in key and '_err' not in key:
                outhist = ROOT.TH1D(key, key, len(results[key]), array('d', results['bins']) )
                for ibin, value in enumerate(results[key]):
                    outhist.SetBinContent(ibin+1, value)
                    outhist.SetBinError(ibin+1, results[key+'_err'][ibin])
                outhist.Write()

    elif results['name']=='di-lep':
        for key in results.keys():
            if 'SFs_' in key and '_err' not in key:
                outhist = ROOT.TH1D(key, key, len(results[key]), array('d', [0,1,2,3]) )
                for ibin, value in enumerate(results[key]):
                    outhist.SetBinContent(ibin+1, value)
                    outhist.SetBinError(ibin+1, results[key+'_err'][ibin])
                outhist.Write()

    outFile.Close()


        

    
    
    

    
  
  