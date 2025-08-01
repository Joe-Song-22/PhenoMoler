# PhenoMoler
Here is a template for your `README.md` file written entirely in English, suitable for a drug generation project based on gene expression profiles:

---

# Gene-Driven Molecular Generation

This repository contains code for generating drug-like molecules conditioned on gene expression profiles using a Transformer-based encoder-decoder model. The project explores how transcriptional responses (e.g., from perturbation experiments or disease states) can guide the de novo design of candidate compounds.

## 🚀 Project Overview

Given a gene expression profile representing a target cellular state (such as a disease phenotype or perturbed transcriptome), our model generates molecular structures that are predicted to induce or reverse such states.

* **Encoder**: Maps a 978-dimensional gene expression profile into a latent representation.
* **Decoder**: Autoregressively generates molecular structures in the form of SELFIES strings.
* **Training Objective**: Minimize the reconstruction error between real and generated molecules, optionally integrating reinforcement or adversarial objectives.

---

## 🧬 Data

* **Gene Expression**: 978-dimensional L1000 profiles (from LINCS).
* **Molecular Structures**: Represented as SMILES or SELFIES.
* **Drug Perturbation Data**: Pre-processed drug-induced expression profiles.

Data should be organized as follows:

```
data/
├── gene_expression.csv     # Rows: samples; Columns: genes
├── molecules.csv           # Corresponding SMILES/SELFIES
├── vocab.txt               # Token vocabulary for molecules
```

---

## 🧠 Model Architecture

* Transformer-based Encoder
* Transformer Decoder with masked self-attention
* Optional: Structural masking using Murcko scaffolds
* Optional: Reinforcement reward for valid/novel molecules (RDKit)

---

## 🛠️ Installation

```bash
conda create -n druggen python=3.9
conda activate druggen
pip install -r requirements.txt
```

Make sure you have [RDKit](https://www.rdkit.org/) installed. You may need to use conda:

```bash
conda install -c rdkit rdkit
```

---

## 💻 Usage

### Training

```bash
python 97encoder_97decoder.py --config configs/train_config.yaml
```



