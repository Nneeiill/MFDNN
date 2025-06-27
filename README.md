pip install -r requirements.txt
See requirements.txt for the full list
numpy
pandas
torch (PyTorch)
scikit-learn
scipy
openpyxl
Data File Description
SPT_curve_selected.xlsx
Contains the SPT (Small Punch Test) data from FEM training and testing datasets used in Experiments 1-4 of this study.

Strain-stress_selected.xlsx
Contains the stress-strain data from FEM training and testing datasets used in Experiments 1-4 of this study.

MFDNN.py
The model definition file for the Multi-Fidelity Deep Neural Network (MFDNN) architecture used in this work.

Data & Model Usage Notes
Please use SPT_curve_selected.xlsx as the model input. Data preprocessing to obtain physically meaningful features is required before training. For details on the preprocessing workflow, please refer to:

Yang, Zheng-Ni, et al. "Machine learning-based extraction of mechanical properties from multi-fidelity small punch test data." Advances in Manufacturing (2025): 1-14.

The real experimental material data used in this study cannot be provided due to commercial confidentiality agreements.

To reproduce the model training workflow, please make sure all required dependencies are installed (see requirements.txt if available), and refer to the code comments in MFDNN.py.
