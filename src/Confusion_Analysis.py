import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- CONFIGURATION ---
dataset_dir = r"c:/Users/ravit/Downloads/Projects/PRML_project/PRML_Dataset"
csv_filename = 'boneage-training-dataset.csv'
image_folder_name = 'boneage-training-dataset'
checkpoint_path = 'best_bone_age_model.pth'

# --- 1. SETUP DATA (Same as before) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv_path = os.path.join(dataset_dir, csv_filename)
img_dir = os.path.join(dataset_dir, image_folder_name)
if os.path.exists(os.path.join(img_dir, 'boneage-training-dataset')):
    img_dir = os.path.join(img_dir, 'boneage-training-dataset')

df = pd.read_csv(csv_path)
if 'boneage' in df.columns: df.rename(columns={'boneage': 'bone_age'}, inplace=True)
if 'male' in df.columns and 'sex' not in df.columns: df['sex'] = df['male'].apply(lambda x: 'M' if x else 'F')
df['path'] = df['id'].apply(lambda x: os.path.join(img_dir, str(x) + '.png'))

# Re-create the Test Split (Using same random_state=42 ensures it's the exact same images)
from sklearn.model_selection import train_test_split
_, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['sex'])
_, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['sex'])

# --- 2. DEFINE STAGES (BUCKETS) ---
# We define 3 stages based on months:
# Child: 0 - 12 years (< 144 months)
# Adolescent: 12 - 18 years (144 - 216 months)
# Young Adult: 18+ years (> 216 months)
def get_stage(age_months):
    if age_months < 144: return 0 # Child
    elif age_months < 216: return 1 # Adolescent
    else: return 2 # Adult

stage_names = ['Child', 'Adolescent', 'Adult']

# --- 3. LOAD MODEL ---
print("Loading Best Model...")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# --- 4. PREDICT ---
test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class InferenceDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try: image = Image.open(row['path']).convert('RGB')
        except: image = Image.new('RGB', (224, 224))
        return self.transform(image), row['bone_age']

test_loader = DataLoader(InferenceDataset(test_df, test_transforms), batch_size=32, shuffle=False)

true_bins = []
pred_bins = []

print("Running Classification Analysis...")
with torch.no_grad():
    for inputs, ages in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        # Get raw ages
        batch_true_ages = ages.cpu().numpy()
        batch_pred_ages = outputs.cpu().numpy().flatten()
        
        # Convert to Bins (Classes)
        for age in batch_true_ages: true_bins.append(get_stage(age))
        for age in batch_pred_ages: pred_bins.append(get_stage(age))

# --- 5. METRICS & PLOT ---
acc = accuracy_score(true_bins, pred_bins)
print(f"\nClassification Accuracy: {acc:.2%}")
print("\nClassification Report:")
print(classification_report(true_bins, pred_bins, target_names=stage_names))

# Confusion Matrix
cm = confusion_matrix(true_bins, pred_bins)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=stage_names, yticklabels=stage_names)
plt.xlabel('Predicted Stage')
plt.ylabel('True Stage')
plt.title('Confusion Matrix: Developmental Stages')
plt.show()