# Strock_Nghiem_bioRxiv_2025

This project explores how feature attribution methods can identify the source of excitation/inhibition (E/I) imbalances in simulated fMRI data.

## Setting up environment

For TVB simulations, use `requirements_tvb.txt`, for classification training and feature attribution, use `requirements_torch.txt` and prior to running scripts run:
```bash
source environment.sh
```

## RNN simulation

Training all the models
```bash
bash model/rnn_exc/train_all_parameter.sh
```
Test all the models and computing feature attribution
```bash
bash model/rnn_exc/test_all_parameter.sh
bash model/rnn_exc/test_all_parameter_baseline.sh
```

## TVB human simulation

Generate simulated data
```bash
python dataset/tvb/EXAMPLE_human.py
python dataset/tvb/EXAMPLE_mouse.py
```
Training all the models
```bash
bash model/tvb/train_all_parameter_human.sh
bash model/tvb/train_all_parameter_mouse.sh
```
Test all the models and computing feature attribution
```bash
bash model/tvb/test_all_parameter_human.sh
bash model/tvb/test_all_parameter_baseline_human.sh
bash model/tvb/test_all_parameter_mouse.sh
bash model/tvb/test_all_parameter_baseline_mouse.sh
```