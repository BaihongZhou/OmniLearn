# Forked Repository for the OmniLearn paper
![Model Architecture](./assets/PET_arch.png)

This is the repository used for Quantum Entanglement measurement for $pp \rightarrow \tau \tau$ system. For citing the OmniLearn, please: 

```
@article{Mikuni:2024qsr,
    author = "Mikuni, Vinicius and Nachman, Benjamin",
    title = "{OmniLearn: A Method to Simultaneously Facilitate All Jet Physics Tasks}",
    eprint = "2404.16091",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "4",
    year = "2024"
}
```


# Installation

The list of packages needed to train/evaluate the model is found in the ```requirements.txt``` file. Alternatively, you can use the Docker container found in this [link](https://hub.docker.com/layers/vmikuni/tensorflow/ngc-23.12-tf2-v1/images/sha256-7aa143ab71775056f1ed3f74f1b7624e55f38108739374af958015dafea45eb3?context=repo).

Our **recommendation** is to use the docker container.

# Data

## $pp \rightarrow \tau^+ \tau^- \rightarrow \pi^+ \bar{\nu_{\tau}} \pi \nu_{\tau}$ 

You can copy the file in Cluster@INPAC:
```bash
cp /lustre/collider/zhoubaihong/QE_study/pptautau/pi_pi_recon_total* <your-file-path>
```
### Data Preprocess
> The codes are stored in 'preprocessing/preprocess_pipi.py';

The input contains three parts:
- "X": $\tau$ visible part and small-R jets; -> Shape:[-1, 7, 9];
  
  Contains: [ $p_T, \eta ,\phi,$ E, Charge, is_electron, is_muon, is_charged_pion, is_neutral_part]
- "MET": MET and MET_Phi information;
- "nu": Our reconstrction target: $(\nu_1, \nu_2)$ -> Shape[-1, 6];

    Contains: [ $p_x^{\nu_1}, p_y^{\nu_1}, p_z^{\nu_1}, p_x^{\nu_2}, p_y^{\nu_2}, p_z^{\nu_2}$ ]



# Training OmniLearn for neutrino reconstruction
For the training run:

```bash
cd scripts
python train.py --dataset "pipi" --layer_scale --local
```

## Evaluation

The evaluation of the trained $pp \rightarrow \tau^+ \tau^- \rightarrow \pi^+ \bar{\nu_{\tau}} \pi \nu_{\tau}$ samples:

```bash
python evaluate_recon.py ---layer_scale --local
```