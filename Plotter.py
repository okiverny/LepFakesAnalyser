import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from array import array
import math
import ROOT
from LFAHelpers import style

####################################################################
###                         HISTMaker                            ###
####################################################################

def HistMaker(data, templator, dict_samples, varname, bins, xtitle=None, weight_name='weight_nominal', region_names=None, Lumi = 140.1, UncertaintyBand=None, UncertaintyBandRatio=None, PlotPurity=None):
    """ Function to make histograms corresponding to templates
    data          - input data frame
    templator     - name of variable in data to define templates
    dict_samples  - disctionary containing the names of templates and fillcolors
    varname       - name of the variable to fill the histogram
    bins          - standard histogram binning [nbins, xmin, xmax]
    weight_name   - which column of dataframe to use as the weight for hists
    region_names  - list of names for each hist bin (speciall sumarry histograms)
    Lumi          - 139 (140.5) fb^-1 by default

    Returns the list of histograms for each template from templator
    """

    plottag = ''
    if templator=='tau_1_tmp' or templator=='tau_1_tmp_qgcomb':
        plottag = '_Tau1Tmp'
    elif templator=='lep_1_tmp':
        plottag = '_Lep1Tmp'
    elif templator=='tau_2_tmp':
        plottag = '_Tau2Tmp'
    elif templator=='lep_2_tmp':
        plottag = '_Lep2Tmp'
    elif templator=='TempCombinations':
        plottag = '_TempCombs'

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
    canvas   = ROOT.TCanvas("canvas", "canvas", 10,32,668,643)

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

    # As a solution to a crush
    def create_pad(name, xlow, ylow, xup, yup):
        mypad = ROOT.TPad()
        mypad.name=name
        mypad.title=''
        mypad.xlow=xlow
        mypad.ylow=ylow
        mypad.xup=xup
        mypad.yup=yup
        return mypad

    uplongpad = ROOT.TPad('uplongpad', '', 0, 0.31, 1, 1)
    #uplongpad = create_pad('uplongpad', 0, 0.31, 1, 1)
    # uplongpad = ROOT.TPad()
    # uplongpad.name='uplongpad'
    # uplongpad.title=''
    # uplongpad.xlow=0
    # uplongpad.ylow=0.31
    # uplongpad.xup=1
    # uplongpad.yup=1
    uplongpad.SetBottomMargin(0)
    lowerPad = ROOT.TPad('lowerPad', '', 0, 0.03, 1, 0.31)
    #lowerPad = create_pad('lowerPad', 0, 0.03, 1, 0.31)
    # lowerPad = ROOT.TPad()
    # lowerPad.name='lowerPad'
    # lowerPad.title=''
    # lowerPad.xlow=0
    # lowerPad.ylow=0.3
    # lowerPad.xup=1
    # lowerPad.yup=0.31
    lowerPad.SetBottomMargin(0.35)
    lowerPad.SetTopMargin(0.05)
    #originPad = ROOT.TPad("originPad", "originPad", 0.12, 0.285, 0.155, 0.335)

    # workaround?
    ROOT.SetOwnership(uplongpad, False)

    doRatio = True
    if doRatio:
        uplongpad.Draw()
        lowerPad.Draw()
        #originPad.Draw()
        uplongpad.cd()
        #uplongpad.SetLogy()

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

            # Blinding
            # if varname=='tH_score_SR' or varname=='tH_LHscore_SR':
            #     print('-----> Blinding')
            #     hist.SetBinContent(1, 0)
            #     hist.SetBinContent(2, 0)
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
        hists[0].GetYaxis().SetRangeUser(1e-4, 1.6*ymax)
        hists[0].Draw('e')


    # Drawing
    stack.Draw("histsame")
    stack.GetHistogram().SetLineWidth(1)

    #if hists[0]:
    hists[0].Draw('AXISSAME')
    hists[0].Draw('ESAME')

    # Uncertanty
    if UncertaintyBand:
        UncertaintyBand.SetFillColor(ROOT.kGray+1)  # ROOT.kAzure+2
        UncertaintyBand.SetFillStyle(3001)
        UncertaintyBand.Draw('E2SAME')


    # Legend
    leg.Clear()
    for ihist, hist in enumerate(hists):
        if dict_samples['sample'][ihist] == 'data':
            leg.AddEntry(hist, 'Data','p') # lp
        else:
            mylabel = label_corrector(dict_samples['sample'][ihist])
            if xtitle and 'di-lep' in xtitle: mylabel = mylabel.replace('#tau_{had}', 'lep')
            leg.AddEntry(hist, mylabel,'f')

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
    text.DrawLatex(0.21 + 0.16, 0.86, "Internal")
    text.SetTextSize(0.04)
    text.DrawLatex(0.21, 0.80, "#sqrt{{s}} = 13 TeV, {:.1f} fb^{{-1}}".format(139.0))

    if xtitle:
        # Add region label
        text_tag = ROOT.TLatex()
        text_tag.SetNDC()
        text_tag.SetTextFont(42)
        text_tag.SetTextSize(0.045)
        if region_tag == label_corrector(region_tag):
            region_tag = region_tag[1:]
        else:
            region_tag = label_corrector(region_tag)
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
            UncertaintyBandRatio.SetFillColor(ROOT.kGray+1)  # ROOT.kAzure+2
            UncertaintyBandRatio.SetFillStyle(3001)
            UncertaintyBandRatio.Draw('E2SAME')

    canvas.SaveAs('Plots/'+varname+'.pdf')

    if varname=='tH_score_SR' or varname=='tH_LHscore_SR':
        canvas.SaveAs('Plots/'+varname+'.png')

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
    text.DrawLatex(0.21 + 0.10, 0.86, "Internal")
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
###                          PIE-CHARTS                          ###
####################################################################

def PieCharter(data, sampler, dict_samples, plotname, show_fractions=False, use_plotly=True):
    # To print out total yields:
    Lumi = 140.1 # old = 139

    df = data.groupby([sampler])['weight_nominal'].sum().reset_index()
    raw_events = []
    for iproc, process in enumerate(dict_samples['sample']):
        df = df.replace(iproc, process)

        # Skip empty entries
        if process not in df[sampler].values:
            continue
        
        # Get raw events depending on the sampler
        if sampler=='sample_Id':
            raw_events.append(len(data[data['sample_Id']==iproc]))
        else:
            raw_events.append(len(data[data[sampler]==process]))

    df['raw_events'] = raw_events
    df = df.set_index(sampler)
    if 'data' in df.index.values:
        df = df.drop(index='data')
    if 'data,data' in df.index.values:
        df = df.drop(index='data,data')
    ## disable sorting?
    #df = df.sort_values(by=['weight_nominal'], ascending=False)
    total = df['weight_nominal'].sum()

    df_yields = df.copy()
    df_yields['weight_nominal'] = df_yields.weight_nominal.apply(lambda row: row*Lumi if row>0 else 0.000001 )
    if show_fractions:
        df_yields.loc['Total'] = df_yields.sum(numeric_only=True)
        logging.info('Here are the yields for the '+data.name[1:]+' dataframe:\n\t =============== Total yields ===============\n\t'+df_yields.to_string().replace('\n', '\n\t') )
    
    ## Normalize
    df['weight_nominal'] = df.weight_nominal.apply(lambda row: row/total if row>0 else 0.000001 )
    if show_fractions:
        logging.info('Here are the fractions for the '+data.name[1:]+' dataframe:\n\t =============== Fractions ===============\n\t'+df.to_string().replace('\n', '\n\t') )

    if not use_plotly:
        plot = df.plot.pie( y='weight_nominal',
                        #title=sampler,
                        fontsize=15,
                        figsize=(5, 5))
        y_axis = plot.axes.get_yaxis()
        y_axis.set_visible(False)
        fig = plot.get_figure()
        fig.savefig(plotname)
        plt.close(fig)

    else:
        import plotly.express as px

        FONT_COLOR = "#010D36"
        BACKGROUND_COLOR = "#F6F5F5"

        #print('---------> plotly')
        df['labels']=df.index
        df['labels'] = df['labels'].replace('jet,sim','jet,tau')
        df['labels'] = df['labels'].replace('sim,jet','tau,jet')
        df['labels'] = df['labels'].replace('sim,sim','tau,tau')
        #print(df.head())

        fig = px.pie(
            df,
            values="weight_nominal",
            names='labels',
            height=540,
            width=840,
            hole=0.65,
            #title="Fakes",
            color_discrete_sequence=px.colors.sequential.RdBu,
        )
        fig.update_layout(
            font_color=FONT_COLOR,
            title_font_size=18,
            #plot_bgcolor=BACKGROUND_COLOR,
            #paper_bgcolor=BACKGROUND_COLOR,
            showlegend=False,
        )
        fig.add_annotation(
            x=0.5,
            y=0.5,
            align="center",
            xref="paper",
            yref="paper",
            showarrow=False,
            font_size=22,
            text="Fakes<br>Composition",
        )
        fig.update_traces(
            hovertemplate=None,
            textposition="outside",
            texttemplate="%{label} (%{percent})",
            textfont_size=16,
            rotation=-20,
            marker_line_width=25,
            marker_line_color=BACKGROUND_COLOR,
        )
        fig.write_image(plotname, format='pdf')


## For comparison of two histograms
def plot_histogram(df, histname, bins, mask1, mask2, weight_name='weight_nominal', xtitle=None, region_tag=None, figname=None, norm=False):

    # Function to construct a ratio title
    def construct_ratio_string(str1, str2):
        # Find the common part of the strings
        common_part = os.path.commonprefix([str1, str2])

        # Remove common part and any leading/trailing spaces
        str1_suffix = str1[len(common_part):].strip()
        str2_suffix = str2[len(common_part):].strip()

        # Construct the ratio string
        ratio_string = f"{common_part.strip()} {str1_suffix}/{str2_suffix}"

        return ratio_string

    # Create two histograms
    if len(bins)==3:
        hist1 = ROOT.TH1D(histname+'1', histname, bins[0], bins[1], bins[2])
        hist2 = ROOT.TH1D(histname+'2', histname, bins[0], bins[1], bins[2])
    else:
        xbins = np.array(bins)
        hist1 = ROOT.TH1D(histname+'1', histname, len(bins)-1, xbins)
        hist2 = ROOT.TH1D(histname+'2', histname, len(bins)-1, xbins)


    # Fill the histograms
    for value,weight in zip(df[mask1][histname].values, df[mask1][weight_name].values):
        hist1.Fill(value, weight)

    for value,weight in zip(df[mask2][histname].values, df[mask2][weight_name].values):
        hist2.Fill(value, weight)

    # Luminosity
    Lumi = 140.1 # old = 139
    hist1.Scale(Lumi)
    hist2.Scale(Lumi)
    if norm:
        hist1.Scale(1.0/hist1.Integral())
        hist2.Scale(1.0/hist2.Integral())

    # values1 = df[mask1][histname].xs(0, level='subentry')
    # values2 = df[mask2][histname].xs(0, level='subentry')

    # weights1 = df[mask1][weight_name].xs(0, level='subentry')
    # weights2 = df[mask2][weight_name].xs(0, level='subentry')

    # hist1.FillN(len(values1), values1, weights1)
    # hist2.FillN(len(values2), values2, weights2)

    # Create a canvas with two pads
    canvas = ROOT.TCanvas("canvas", "canvas", 10,32,668,643)
    pad1 = ROOT.TPad("pad1", "pad1", 0, 0.30, 1, 1)
    pad2 = ROOT.TPad("pad2", "pad2", 0, 0.01, 1, 0.30)
    pad1.SetBottomMargin(0)
    pad2.SetBottomMargin(0.35)
    pad2.SetTopMargin(0.05)
    pad1.Draw()
    pad2.Draw()

    # Draw the histograms on the first pad
    pad1.cd()
    hist1.SetLineColor(ROOT.kRed)
    hist2.SetLineColor(ROOT.kBlue)
    hist1.SetMarkerSize(0)
    hist1.SetLineWidth(4)
    hist2.SetMarkerSize(0)
    hist2.SetLineWidth(2)

    hist1.SetLabelSize(0.107)
    hist1.GetYaxis().SetTitle("Events")

    # Set the Y-axis range
    ymax = 1111111
    ymax = max( hist1.GetBinContent(hist1.GetMaximumBin()), hist2.GetBinContent(hist2.GetMaximumBin()) )
    hist1.GetYaxis().SetRangeUser(1e-4, 1.4*ymax)

    hist1.Draw()
    hist2.Draw("SAME")

    # Add a legend
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetTextFont(42)
    legend.SetTextSize( 0.06 )
    legend.SetFillColor(0)
    legend.SetLineColor(0)
    if mask1.name:
        legend.AddEntry(hist1, mask1.name, "L")
        legend.AddEntry(hist2, mask2.name, "L")
    else:
        legend.AddEntry(hist1, "W+jets", "L")
        legend.AddEntry(hist2, "W+jets (new)", "L")
    legend.Draw()

    # Draw the ratio plot on the second pad
    pad2.cd()
    pad2.SetGridy()
    #==========
    hratio = ROOT.TH1D(hist1.Clone("hratio"))
    hratio.SetMarkerSize(0)
    hratio.SetLineWidth(2)
    hratio.Divide(hist1, hist2, 1., 1., "P")

    if xtitle:
        hratio.SetXTitle(xtitle)
    else:
        hratio.SetXTitle(histname)
    hratio.GetYaxis().SetRangeUser(0.2, 1.8)
    hratio.SetMarkerSize(0.0)
    hratio.SetMarkerStyle(20)
    hratio.SetLineColor(ROOT.kBlack)
    ratiosize=0.127
    hratio.GetYaxis().SetLabelSize(ratiosize)
    hratio.GetYaxis().SetTitleSize(ratiosize)
    hratio.GetXaxis().SetLabelSize(ratiosize)
    hratio.GetXaxis().SetTitleSize(ratiosize)
    hratio.GetXaxis().SetTitleOffset(0.93)
    if mask1.name:
        ratiotitle = construct_ratio_string(mask1.name, mask2.name)
        hratio.GetYaxis().SetTitle(ratiotitle)
    else:
        hratio.GetYaxis().SetTitle('Old/New')
    hratio.GetYaxis().SetTitleOffset(0.43)
    hratio.GetYaxis().SetNdivisions(506)
    hratio.Draw("p:e")

    # Save the plot to a file
    if figname:
        canvas.SaveAs('Plots/'+figname)
    else:
        canvas.SaveAs('Plots/'+histname+'.pdf')

def plot_weight(df, mask1, mask2):
    # Separate positive and negative values
    positive_weights = df[mask1]['weight_nominal']
    negative_weights = df[mask2]['weight_nominal']

    # Plot the histograms
    f = plt.figure(figsize=(10,10))

    # plt.hist(positive_weights, bins=100, color='blue', alpha=0.5, label='Z+jets', range=(-0.01, 0.02), density=True)
    # plt.hist(negative_weights, bins=100, color='red', alpha=0.5, label='Z+jets (new)', range=(-0.01, 0.02), density=True)
    plt.hist(positive_weights, bins=100, color='blue', alpha=0.5, label='W+jets', range=(-0.003, 0.005), density=True)
    plt.hist(negative_weights, bins=100, color='red', alpha=0.5, label='W+jets (new)', range=(-0.003, 0.005), density=True)

    # Add labels and title
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('V+jets weights')
    plt.legend()

    plt.savefig('weights.png')
    plt.clf()

### Help function to transform/replace labels in the plots
def label_corrector(region_tag):
    region_tag_cor = ''
    my_tags = {
        '_Tau1CR': '#tau_{had} fail-ID',
        '_Tau2CR': '#tau_{had}^{sublead} fail-ID',
        '_Lep1CR': 'lepton fail ID and ISO',
        '_1b_': '1b',
        '_2b_': '2 b-jets',
        '_12b_': '1-2 b-jets',
        'lepT': 'tight lepton',
        '_SR_tight': 'SR + BDT cut',

        '_1p_': '1p',
        #'_pt1': '20<pT<30',

        # templates
        #'sim,sim': 'lep+lep',
        #'sim,jet': 'lep+jet',
        #'jet,sim': 'jet+lep',
        'jet,jet': 'jet+jet',
        'data,data': 'data',
        'sim,sim': '#tau_{had}+#tau_{had}',
        'sim,jet': '#tau_{had}+jet',
        'jet,sim': 'jet+#tau_{had}',
    }

    for key in my_tags.keys():
        if key in region_tag:
            if region_tag_cor:
                region_tag_cor+=', '+my_tags[key]
            else:
                region_tag_cor = my_tags[key]

    if not region_tag_cor:
        region_tag_cor = region_tag

    return region_tag_cor