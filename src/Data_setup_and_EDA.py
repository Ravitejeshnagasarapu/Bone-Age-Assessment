import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
dataset_dir = r"c:/Users/ravit/Downloads/Projects/PRML_project/PRML_Dataset" 
csv_filename = 'boneage-training-dataset.csv'
image_folder_name = 'boneage-training-dataset' 

# --- 1. CHECK GPU & PATHS ---
print(f"Checking System...")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: GPU not detected.")

csv_path = os.path.join(dataset_dir, csv_filename)
img_dir = os.path.join(dataset_dir, image_folder_name)

# Handle nested folder issue
nested_check = os.path.join(img_dir, 'boneage-training-dataset')
if os.path.exists(nested_check):
    img_dir = nested_check

print(f"CSV Path: {csv_path}")

# --- 2. LOAD DATA & FIX COLUMNS ---
df = pd.read_csv(csv_path)

print(f"Original Columns found: {df.columns.tolist()}")

# --- FIX: Standardize Column Names ---
# The dataset often has 'boneage' instead of 'bone_age'
if 'boneage' in df.columns:
    print("  -> Renaming 'boneage' column to 'bone_age'")
    df.rename(columns={'boneage': 'bone_age'}, inplace=True)

# The dataset often has 'male' (True/False) instead of 'sex' (M/F)
if 'male' in df.columns and 'sex' not in df.columns:
    print("  -> Converting 'male' column to 'sex' (M/F)")
    df['sex'] = df['male'].apply(lambda x: 'M' if x else 'F')
# ---------------------------------------

# Validate that we now have the required columns
required_cols = ['id', 'bone_age', 'sex']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"CRITICAL ERROR: Could not find or create column '{col}'. Columns are: {df.columns}")

# Create image paths
df['path'] = df['id'].apply(lambda x: os.path.join(img_dir, str(x) + '.png'))

# --- 3. VISUALIZATION (EDA) ---
print("Generating Plots...")
plt.figure(figsize=(12, 5))

# Age Distribution
plt.subplot(1, 2, 1)
sns.histplot(df['bone_age'], kde=True, bins=30)
plt.title('Distribution of Bone Age (Months)')
plt.xlabel('Age (Months)')

# Sex Distribution
plt.subplot(1, 2, 2)
sns.countplot(x='sex', data=df)
plt.title('Distribution by Sex')
plt.show()

# --- 4. DATA SPLIT ---
# PDF requires 70% Train, 15% Val, 15% Test
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['sex'])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['sex'])

print("-" * 30)
print(f"Training Set:   {len(train_df)} images")
print(f"Validation Set: {len(val_df)} images")
print(f"Test Set:       {len(test_df)} images")
print("-" * 30)