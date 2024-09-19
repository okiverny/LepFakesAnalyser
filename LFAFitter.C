#include "TH1.h"
#include "TString.h"
#include "TError.h"

#include <vector>
#include <string> 

#include "RooArgList.h"
#include "RooRealVar.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"
#include "RooAddPdf.h"
#include "RooAbsReal.h"
#include "RooFitResult.h"
#include "TMatrixDSym.h"


std::vector<std::pair<double,double>>  LFA_TemplatesFitter(TH1D* hdata, TH1D* hsig, std::vector<TH1D*> hbkgs, double& fsig_val, double& fsig_err, 
                                        bool BkgOnlyFit = true, bool DEBUG = true){

    // enum MsgLevel { DEBUG=0, INFO=1, PROGRESS=2, WARNING=3, ERROR=4, FATAL=5 }
    RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

    // Setup everything for RooFit
    RooRealVar t("t","t",hdata->GetXaxis()->GetBinLowEdge(1),hdata->GetXaxis()->GetBinUpEdge(hdata->GetNbinsX()));

    RooDataHist rdhData("data",hdata->GetTitle(),t,hdata);
    RooDataHist rdhSig("rdhsig",hsig->GetTitle(),t,hsig);
    RooHistPdf sig("sig",hsig->GetTitle(),t, rdhSig);

    double Nsig = hsig->Integral();
    double Ndata = hdata->Integral();
    RooRealVar fsig("fsig","sig fraction",Nsig/Ndata,0.,1.);  //RooRealVar fsig("fsig","sig fraction",Nsig/Ndata,0.,1.);
    if (BkgOnlyFit){
        fsig.setRange(Nsig/Ndata, Nsig/Ndata);
    }

    RooArgList *bkgPdfList = new RooArgList("bkg_pdfs");
    RooArgList *bkgFracList = new RooArgList("bkg_fracs");

    
    for(unsigned int ibkg=0; ibkg<hbkgs.size(); ibkg++){

        TString rdhName = hbkgs.at(ibkg)->GetTitle();

        RooDataHist* rdhist5 = new RooDataHist(rdhName, "bkg7", t, hbkgs.at(ibkg));
        RooHistPdf* bkg5 = new RooHistPdf(rdhName+"_pdf","bkg6",t, *rdhist5);
        bkgPdfList->add(*bkg5);

        // background fractions
        if (ibkg<hbkgs.size()-1) {
            //RooRealVar *frac = new RooRealVar(Form("frac_%s",rdhName.Data()),Form("frac_%s",rdhName.Data()),0.3,0.05,0.90);
            RooRealVar *frac = new RooRealVar(Form("frac_%s",rdhName.Data()),Form("frac_%s",rdhName.Data()),0.8,0.01,0.99);
            bkgFracList->add(*frac);
        }
    }


    RooAddPdf bkg_model("bkg_model", "Background Model", *bkgPdfList, *bkgFracList); // RooArgList(bkg1,bkg2,bkg3)  RooArgList(frac,frac2)

    // model = fsig*S(X) + (1-fsig)*[ frac_qjet*Q(x) + frac_gjet*G(x) ]
    RooAddPdf model("model", "Sig+Bkg Model", RooArgList(sig, bkg_model), fsig);

    // Fitting model to data
    double valmin = hdata->GetXaxis()->GetBinLowEdge(1);
    double valmax = hdata->GetXaxis()->GetBinLowEdge(hdata->GetNbinsX()) + hdata->GetXaxis()->GetBinWidth(hdata->GetNbinsX());
    t.setRange("TheRange",valmin,valmax);
    RooFitResult *RFresult = model.fitTo(rdhData, RooFit::Range("TheRange"), RooFit::SumCoefRange("TheRange"), RooFit::PrintLevel(-1), RooFit::Save()); //Save()
    if (DEBUG){
        //model.Print("t");
        RFresult->Print("v");
        cout << "final value of floating parameters" << endl;
        RFresult->floatParsFinal().Print("s");
        // Access correlation matrix elements
        //cout << "correlation between sig1frac and a0 is  " << RFresult->correlation(sig1frac, a0) << endl;
        //cout << "correlation between bkgfrac and mean is " << RFresult->correlation("bkgfrac", "mean") << endl;
        //cout << "correlation between bkgfrac and mean is " << RFresult->correlation("frac_sim_jet", "frac_jet_sim") << endl;
        
        // Extract covariance and correlation matrix as TMatrixDSym
        const TMatrixDSym &cor = RFresult->correlationMatrix();
        const TMatrixDSym &cov = RFresult->covarianceMatrix();
        // Print correlation, covariance matrix
        cout << "correlation matrix" << endl;
        cor.Print();
        cout << "covariance matrix" << endl;
        cov.Print();
    }

    std::vector<double> fbkg_vals;
    std::vector<double> fbkg_errs;
    std::vector<std::pair<double, double>> fbkg_val_pairs;

    fsig_val=fsig.getVal();	
    if (!BkgOnlyFit) fsig_err=fsig.getError();

    for(unsigned int ibkg=0; ibkg<hbkgs.size()-1; ibkg++){
        double val = dynamic_cast<RooRealVar*>(bkgFracList->at(ibkg))->getVal();
        double err = dynamic_cast<RooRealVar*>(bkgFracList->at(ibkg))->getError();
        
        std::pair<double, double> p{val, err};
        fbkg_val_pairs.push_back(p);

        fbkg_vals.push_back(val);
        fbkg_errs.push_back(err);
    }

    //return std::make_pair(fbkg_vals, fbkg_vals);
    return fbkg_val_pairs;
}

void LFAFitter(){

    
}