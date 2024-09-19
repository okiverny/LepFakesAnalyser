# Tau fakes estimation for tHq analysis

[ToC]

## 1. Introduction
Default scripts to get tau fake scale-factors for tHq analysis.

## 2. How to run the script

### 2.1. Dileptau channel

To get the tau fakes numbers in dileptau channel, run the following:
```bash
 python RunAnalysis.py -c config_2l1tau_nominal.yaml
```
**Important:** Please, note that the estimates of tau fakes for events with 2 b-jets not yet implemented as this requires to run BDT training of quark-jets against gluon-jets.

### 2.2. Lepditau channel

To get the tau fakes numbers in lepditau channel, run the following:
```bash
 python RunAnalysis.py -c config_1l2tau_nominal.yaml
```

## 3. Inputs

The directory with ntuples are specified in ```LFAConfig.py``` file (To be ported to the cmd args).
```
lephad: /Users/okiverny/workspace/tH/3l1tau_loose_august/
hadhad: /Users/okiverny/workspace/tH/3l2tau_loose_april/
```

These ntuples will be moved to BAF, so that everyone can access them (TBD).
