# -*- coding: utf-8 -*-
"""
Created on 06Feb2026_13-38

@author: Pradnya Kamble
"""

# multitask_predict.py
import os
import subprocess
import pandas as pd
from keras.models import load_model
import joblib

# --- Step 0: Define dynamic paths relative to current directory ---
cwd = os.getcwd()
padel_jar = os.path.join(cwd, 'PaDEL-Descriptor', 'PaDEL-Descriptor.jar')
config_file = os.path.join(cwd, 'padel_config_file')
descriptor_file = os.path.join(cwd, 'padel_descriptor_file.xml')
input_smi_file = os.path.join(cwd, 'input', 'sample.smi')
desc_output_file = os.path.join(cwd, 'padel_desc_output.csv')
X_train_file = os.path.join(cwd, 'X_train.csv')
scaler_file = os.path.join(cwd, 'padel_scaler.pkl')
model_file = os.path.join(cwd, 'gsk3bmt_model.h5')
output_file = os.path.join(cwd, 'multitask_prediction_results.csv')

# --- Step 1: Update padel_config_file with current paths ---
print("Updating PaDEL configuration file...")
with open(config_file, 'w') as f:
    f.write(f"""Compute2D=true
Compute3D=false
ComputeFingerprints=true
Convert3D=No
DescriptorFile={desc_output_file.replace('\\', '/')}
DetectAromaticity=true
Directory={os.path.join(cwd, 'input').replace('\\', '/')}
Log=true
MaxCpdPerFile=0
MaxJobsWaiting=-1
MaxRunTime=-1
MaxThreads=-1
RemoveSalt=true
Retain3D=false
RetainOrder=true
StandardizeNitro=true
StandardizeTautomers=false
TautomerFile=
UseFilenameAsMolName=false
""")

# --- Step 2: Run PaDEL descriptor calculation ---
print("Running PaDEL descriptor calculation...")
padel_cmd = [
    'java', '-jar', padel_jar,
    '-config', config_file,
    '-descriptortypes', descriptor_file
]
subprocess.call(padel_cmd)

# --- Step 3: Load descriptors and SMILES ---
print("Loading descriptors and SMILES...")
padel_desc_output = pd.read_csv(desc_output_file)
input_smiles = pd.read_csv(input_smi_file, header=None, names=["SMILES"])
padel_desc_output.insert(0, 'SMILES', input_smiles['SMILES'])  # Ensure SMILES is 1st column

# --- Step 4: Match and clean columns with training data ---
X_train = pd.read_csv(X_train_file)
common_columns = [col for col in padel_desc_output.columns if col in X_train.columns]
filtered_descriptors = padel_desc_output[common_columns]

print("Cleaning and normalizing descriptors...")
non_numeric_columns = filtered_descriptors.select_dtypes(exclude=['number']).columns
for col in non_numeric_columns:
    train_numeric = pd.to_numeric(X_train[col], errors='coerce')
    mean_value = train_numeric.mean()
    filtered_descriptors[col] = pd.to_numeric(filtered_descriptors[col], errors='coerce')
    filtered_descriptors[col] = filtered_descriptors[col].fillna(mean_value)

cleaned_descriptors = filtered_descriptors.apply(pd.to_numeric, errors='coerce')
cleaned_descriptors = cleaned_descriptors[X_train.columns]

# --- Step 5: Apply scaler ---
scaler = joblib.load(scaler_file)
cleaned_descriptors_scaled = pd.DataFrame(
    scaler.transform(cleaned_descriptors),
    columns=X_train.columns
)

# --- Step 6: Load model and predict ---
print("Loading model and making predictions...")
model = load_model(model_file, compile=False)
predictions = model.predict(cleaned_descriptors_scaled)

# --- Step 7: Postprocess outputs ---
pred_class_prob = predictions[0].ravel()
pred_pic50 = predictions[1].ravel()
pred_class_label = (pred_class_prob >= 0.5).astype(int)
pred_class_name = pd.Series(pred_class_label).map({1: 'Inhibitor', 0: 'Non-inhibitor'})
pred_ic50_uM = 10 ** (6 - pred_pic50)
pred_ic50_M = 10 ** (-pred_pic50)

results_df = pd.DataFrame({
    'SMILES': input_smiles['SMILES'],
    'Predicted_Prob': pred_class_prob,
    'Predicted_Class': pred_class_name,
    # 'Predicted_pIC50': pred_pic50,
    'Predicted_IC50_uM': pred_ic50_uM,
    # 'Predicted_IC50': pred_ic50_M
})

results_df.to_csv(output_file, index=False)
print(f"Predictions saved to: {output_file}")
