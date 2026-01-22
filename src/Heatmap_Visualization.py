import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

# --- CONFIGURATION ---
dataset_dir = r"c:/Users/ravit/Downloads/Projects/PRML_project/PRML_Dataset"
csv_filename = 'boneage-training-dataset.csv'
image_folder_name = 'boneage-training-dataset'
checkpoint_path = 'best_bone_age_model.pth'

# --- 1. SETUP (FORCE CPU FOR SAFETY) ---
device = torch.device("cpu") 
print(f"Running Visualization on: {device}")

csv_path = os.path.join(dataset_dir, csv_filename)
img_dir = os.path.join(dataset_dir, image_folder_name)
if os.path.exists(os.path.join(img_dir, 'boneage-training-dataset')):
    img_dir = os.path.join(img_dir, 'boneage-training-dataset')

# --- LOAD & FIX DATA ---
df = pd.read_csv(csv_path)

if 'boneage' in df.columns:
    df.rename(columns={'boneage': 'bone_age'}, inplace=True)
if 'male' in df.columns and 'sex' not in df.columns:
    df['sex'] = df['male'].apply(lambda x: 'M' if x else 'F')

df['path'] = df['id'].apply(lambda x: os.path.join(img_dir, str(x) + '.png'))

# Use a few random samples
sample_df = df.sample(n=4, random_state=42) 

# --- 2. GRAD-CAM CLASS ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        self.model.zero_grad()
        output = self.model(x)
        
        output.backward(gradient=torch.ones_like(output), retain_graph=True)
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activations[0].clone() 
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activation, dim=0).detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        
        max_val = np.max(heatmap)
        if max_val != 0:
            heatmap /= max_val
            
        return heatmap, output.item()

# --- 3. LOAD MODEL ---
print("Loading Model...")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

grad_cam = GradCAM(model, model.layer4[1].conv2)

# --- 4. VISUALIZE ---
print("Generating Heatmaps...")
plt.figure(figsize=(15, 5))

for i, (idx, row) in enumerate(sample_df.iterrows()):
    try:
        img_pil = Image.open(row['path']).convert('RGB')
    except:
        continue
        
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
    
    try:
        heatmap, predicted_age = grad_cam(img_tensor)
        
        if heatmap is None or np.isnan(heatmap).any():
            heatmap = np.nan_to_num(heatmap)
            
    except Exception as e:
        print(f"Error: {e}")
        continue
    
    # Resize heatmap
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay
    img_resized = img_pil.resize((224, 224))
    img_np = np.array(img_resized)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    # Plot
    plt.subplot(1, 4, i + 1)
    plt.imshow(superimposed_img)
    plt.title(f"True: {row['bone_age']}m\nPred: {predicted_age:.1f}m")
    plt.axis('off')

plt.tight_layout()
plt.show()

print("Heatmaps Generated! Look for Red/Yellow areas on the bones.")