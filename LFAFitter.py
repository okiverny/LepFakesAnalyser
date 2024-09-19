import ctypes

import ROOT
from LFAHelpers import compute_last_fraction, BKGCorrector

# Loading RooFit macros
ROOT.gROOT.LoadMacro("LFAFitter.C+")
from ROOT import LFA_TemplatesFitter


def Fit_DiTau(data_df, hists, dict_tmps_comb, BKGonlyFit=True):

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
    fbkg_val_pairs = LFA_TemplatesFitter(hdata, hsig, vect_bkgs, fsig_val, fsig_err, ctypes.c_bool(BKGonlyFit)) # vector containing vect_bkgs.size()-1 pairs of background fractions <val, err> from the template fit

    fbkg_vals, fbkg_errs = [fsig_val.value], [fsig_err.value]
    IntData = hdata.Integral()
    fbkg_init_vals = [hsig.Integral()/IntData]

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

    # Convert fractions into norm factors
    for i, fval in enumerate(fbkg_vals):
        fbkg_vals[i] = fval/fbkg_init_vals[i]
        fbkg_errs[i] = fbkg_errs[i]/fbkg_init_vals[i]

    print('Ditau fake SFs from fit:')
    print('SF    = ',fbkg_vals)
    print('SFerr = ',fbkg_errs)

    return BKGCorrector(data_df, 'TempCombinations', dict_tmps_comb, fbkg_vals, fbkg_errs)

# Template fit to JetTrackWidth distribution
def Fit_Tau(hists, unknown_is_jet=False, BKGonlyFit=True, DEBUG=False):

    # dummy initialization
    sig = ROOT.TH1D( hists[1].Clone("mc") )

    data, qjet, gjet = None, None, None
    is_sig_started = False
    N_underflow_unknown = 0.0
    N_underflow_data = 0.0
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
        
    # SF for unknown
    SF_unknown = 1.0
    if N_underflow_unknown>0:
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
    #print('Init frac val:',fbkg_init_vals)

    # fractions from the fit: fbkg*(1-fsig)
    fbkg_vals.append( fbkg_val_pairs[0].first*(1-fsig_val.value) )
    fbkg_errs.append( fbkg_val_pairs[0].second*(1-fsig_val.value) )

    fbkg_val_last, fbkg_err_last = compute_last_fraction(fbkg_vals, fbkg_errs)
    fbkg_vals.append(fbkg_val_last)
    fbkg_errs.append(fbkg_err_last)

    if DEBUG:
        print('frac val:',fbkg_vals)
        print('frac err:',fbkg_errs)

    # Convert fractions into norm factors
    for i, fval in enumerate(fbkg_vals):
        fbkg_vals[i] = fval/fbkg_init_vals[i]
        fbkg_errs[i] = fbkg_errs[i]/fbkg_init_vals[i]

    #print('SF val:',fbkg_vals)
    #print('SF err:',fbkg_errs)

    return [fbkg_vals[1], fbkg_vals[2]], [fbkg_errs[1],fbkg_errs[2]], SF_unknown
    