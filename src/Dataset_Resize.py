import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import copy
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- CONFIGURATION ---
dataset_dir = r"c:/Users/ravit/Downloads/Projects/PRML_project/PRML_Dataset" 
csv_filename = 'boneage-training-dataset.csv'
image_folder_name = 'boneage-training-dataset' 

# --- 1. SETUP & LOAD DATA ---
print("--- Step 1: Loading Data ---")
csv_path = os.path.join(dataset_dir, csv_filename)
img_dir = os.path.join(dataset_dir, image_folder_name)

# Handle nested folder issue
nested_check = os.path.join(img_dir, 'boneage-training-dataset')
if os.path.exists(nested_check):
    img_dir = nested_check

# Load CSV and Standardize Columns
df = pd.read_csv(csv_path)
if 'boneage' in df.columns:
    df.rename(columns={'boneage': 'bone_age'}, inplace=True)
if 'male' in df.columns and 'sex' not in df.columns:
    df['sex'] = df['male'].apply(lambda x: 'M' if x else 'F')

df['path'] = df['id'].apply(lambda x: os.path.join(img_dir, str(x) + '.png'))

# Data Split
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['sex'])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['sex'])

print(f"Training Set:   {len(train_df)}")
print(f"Validation Set: {len(val_df)}")
print(f"Test Set:       {len(test_df)}")

# --- 2. TRANSFORMS & DATASET ---
print("\n--- Step 2: Preparing Data Loaders ---")
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),        
    transforms.RandomRotation(10),        
    transforms.CenterCrop(224),           
    transforms.ToTensor(),                
    transforms.Normalize(mean=[0.5], std=[0.5]) 
])

val_test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class BoneAgeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        age = row['bone_age']
        sex_str = row['sex'] 
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(age, dtype=torch.float32), torch.tensor(1.0 if sex_str == 'M' else 0.0, dtype=torch.float32)

BATCH_SIZE = 32
train_loader = DataLoader(BoneAgeDataset(train_df, train_transforms), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(BoneAgeDataset(val_df, val_test_transforms), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(BoneAgeDataset(test_df, val_test_transforms), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- 3. MODEL SETUP ---
print("\n--- Step 3: Building the Model ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, 1) 
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- RESUME LOGIC ---
checkpoint_path = 'best_bone_age_model.pth'
best_mae = float('inf')

if os.path.exists(checkpoint_path):
    print(f"\nFound saved model '{checkpoint_path}'! Loading weights...")
    try:
        model.load_state_dict(torch.load(checkpoint_path))
        print("Model weights loaded. Resuming training...")
        # If we load a model, we assume its MAE is the baseline to beat
        # We set this high so it continues to save if it improves further
        best_mae = 30.0 
    except Exception as e:
        print(f"Could not load model: {e}. Starting fresh.")
else:
    print("\nNo saved model found. Starting fresh.")

# --- 4. TRAINING LOOP ---
print("\n--- Step 4: Training (Press Ctrl+C to Stop Early & Evaluate) ---")
NUM_EPOCHS = 5
history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

try:
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        
        for i, (inputs, ages, sexes) in enumerate(train_loader):
            inputs, ages = inputs.to(device), ages.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            loss = criterion(model(inputs), ages)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            if (i + 1) % 50 == 0:
                print(f"  > Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.2f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        print(f'Train Loss: {epoch_loss:.4f}')

        # Validation
        print("  Validating...", end='\r')
        model.eval()
        val_mae = 0.0
        
        with torch.no_grad():
            for inputs, ages, sexes in val_loader:
                inputs, ages = inputs.to(device), ages.to(device).float().view(-1, 1)
                outputs = model(inputs)
                val_mae += torch.abs(outputs - ages).sum().item()

        epoch_val_mae = val_mae / len(val_loader.dataset)
        history['val_mae'].append(epoch_val_mae)
        print(f'Val MAE:    {epoch_val_mae:.4f} months')

        if epoch_val_mae < best_mae:
            best_mae = epoch_val_mae
            torch.save(model.state_dict(), checkpoint_path)
            print("  -> New Best Model Saved!")
            
except KeyboardInterrupt:
    print("\n\nTraining stopped by user! Proceeding to Evaluation...")

# --- 5. FINAL EVALUATION ---
print("\n--- Step 5: Final Evaluation on Test Set ---")
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print("Loaded best model for evaluation.")

model.eval()
true_ages, pred_ages, sexes_list = [], [], []

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

# Metrics
mae = mean_absolute_error(true_ages, pred_ages)
rmse = np.sqrt(mean_squared_error(true_ages, pred_ages))
r2 = r2_score(true_ages, pred_ages)

print(f"Test Set MAE:  {mae:.2f} months")
print(f"Test Set RMSE: {rmse:.2f} months")
print(f"R2 Score:      {r2:.4f}")

# Plots
plt.figure(figsize=(12, 5))

# Scatter Plot
plt.subplot(1, 2, 1)
plt.scatter(true_ages, pred_ages, alpha=0.5, color='blue', s=10)
plt.plot([0, 240], [0, 240], 'r--', label='Ideal')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.title(f'Prediction Scatter (MAE: {mae:.2f})')
plt.legend()
plt.grid(True)

# Bias Analysis
mae_male = mean_absolute_error(true_ages[sexes_list == 1], pred_ages[sexes_list == 1])
mae_female = mean_absolute_error(true_ages[sexes_list == 0], pred_ages[sexes_list == 0])

plt.subplot(1, 2, 2)
bars = plt.bar(['Male', 'Female'], [mae_male, mae_female], color=['cyan', 'pink'])
plt.ylabel('MAE (Months)')
plt.title('Gender Bias Analysis')
plt.bar_label(bars, fmt='%.2f')

plt.tight_layout()
plt.show()

print("\nDONE With the Layout")