# Software Environment and Dependencies

To reproduce the results in this study, please ensure that all required packages are installed:

```bash
pip install -r requirements.txt
```

The key dependencies include:

  * `numpy`
  * `pandas`
  * `torch`
  * `scikit-learn`
  * `scipy`
  * `openpyxl`

For the complete list of dependencies, please refer to the `requirements.txt` file included in the supplementary materials.

# Data Files Description

**SPT-curve.xlsx**

This file contains the Small Punch Test (SPT) simulation data used for training and testing across Experiments 1–4.

**Stress-Strain.xlsx**

This file contains the uniaxial tensile stress–strain data from FEM simulations used in Experiments 1–4.

# Finite Element Simulation Files

All finite element simulations were conducted using LS-DYNA R2023.

**SPT-FEM folder:** This folder contains the main `Main.key` driver deck along with associated control, contact, boundary, and material definition files. Users may execute the model directly or substitute different material cards by modifying the `INCLUDE Parameter material` entry, which links to the `DATABASE MATERIAL` section. This modular structure allows users to simulate different high-strength steels.

**Tensile-FEM folder:** This folder contains the FEM files for uniaxial tensile simulations. By running these files, users can obtain the stress–strain curves under standard tensile loading conditions for the studied materials.

# Model Code and Usage

**MFDNN.py**

This file contains the model definition for the Multi-Fidelity Deep Neural Network (MFDNN) architecture used in this work.

**Usage Notes:**

  * Use `SPT-curve.xlsx` as the model input.

  * Prior to training, data preprocessing is required to extract physically meaningful features.

  * For details on the preprocessing workflow, please refer to:

    Yang, Zheng-Ni, et al. “Machine learning-based extraction of mechanical properties from multi-fidelity small punch test data.” *Advances in Manufacturing*, 2025.

# Real Experimental Data Disclaimer

Due to commercial confidentiality agreements with our industrial partners, the real experimental material data used in this study cannot be publicly released. We appreciate your understanding.
