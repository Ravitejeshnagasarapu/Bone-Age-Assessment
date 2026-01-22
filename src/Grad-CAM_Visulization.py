import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
dataset_dir = r"c:/Users/ravit/Downloads/Projects/PRML_project/PRML_Dataset"
csv_filename = 'boneage-training-dataset.csv'
image_folder_name = 'boneage-training-dataset'
checkpoint_path = 'best_bone_age_model.pth'

# --- 1. SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Evaluation on: {device}")

csv_path = os.path.join(dataset_dir, csv_filename)
img_dir = os.path.join(dataset_dir, image_folder_name)
if os.path.exists(os.path.join(img_dir, 'boneage-training-dataset')):
    img_dir = os.path.join(img_dir, 'boneage-training-dataset')

# Load and Fix Data
df = pd.read_csv(csv_path)
if 'boneage' in df.columns: df.rename(columns={'boneage': 'bone_age'}, inplace=True)
if 'male' in df.columns and 'sex' not in df.columns: df['sex'] = df['male'].apply(lambda x: 'M' if x else 'F')
df['path'] = df['id'].apply(lambda x: os.path.join(img_dir, str(x) + '.png'))

# Re-create the 15% Test Split
_, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['sex'])
_, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['sex'])

# --- 2. DATASET & MODEL ---
test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class BoneAgeDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try: image = Image.open(row['path']).convert('RGB')
        except: image = Image.new('RGB', (224, 224))
        sex_val = 1.0 if row['sex'] == 'M' else 0.0
        return self.transform(image), torch.tensor(row['bone_age'], dtype=torch.float32), torch.tensor(sex_val, dtype=torch.float32)

test_loader = DataLoader(BoneAgeDataset(test_df, test_transforms), batch_size=32, shuffle=False)

print("Loading saved model weights...")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# --- 3. EVALUATE ---
true_ages, pred_ages, sexes_list = [], [], []

print("Calculating metrics on Test Set...")
with torch.no_grad():
    for inputs, ages, sexes in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        true_ages.extend(ages.cpu().numpy().flatten())
        pred_ages.extend(outputs.cpu().numpy().flatten())
        sexes_list.extend(sexes.cpu().numpy().flatten())

true_ages = np.array(true_ages)
pred_ages = np.array(pred_ages)
sexes_list = np.array(sexes_list)

# --- 4. PRINT & PLOT ---
mae = mean_absolute_error(true_ages, pred_ages)
rmse = np.sqrt(mean_squared_error(true_ages, pred_ages))
r2 = r2_score(true_ages, pred_ages)

print("\n" + "="*30)
print(f"Test Set MAE:  {mae:.2f} months")
print(f"Test Set RMSE: {rmse:.2f} months")
print(f"R2 Score:      {r2:.4f}")
print("="*30 + "\n")

# Plots
plt.figure(figsize=(12, 5))

# Scatter
plt.subplot(1, 2, 1)
plt.scatter(true_ages, pred_ages, alpha=0.5, color='blue', s=10)
plt.plot([0, 240], [0, 240], 'r--', label='Ideal')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.title(f'Prediction Scatter (MAE: {mae:.2f})')
plt.legend()
plt.grid(True)

# Bias
mae_male = mean_absolute_error(true_ages[sexes_list == 1], pred_ages[sexes_list == 1])
mae_female = mean_absolute_error(true_ages[sexes_list == 0], pred_ages[sexes_list == 0])

plt.subplot(1, 2, 2)
bars = plt.bar(['Male', 'Female'], [mae_male, mae_female], color=['cyan', 'pink'])
plt.ylabel('MAE (Months)')
plt.title('Gender Bias Analysis')
plt.bar_label(bars, fmt='%.2f')

plt.tight_layout()
plt.show()