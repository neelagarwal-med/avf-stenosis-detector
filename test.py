import os
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
import joblib

# ==========================================
# 1. ARCHITECTURE
# ==========================================
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid() 
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

# ==========================================
# 2. CLINICALLY OPTIMIZED PREPROCESSING
# ==========================================
def process_audio(file_path, sr=22050, n_mels=64):
    """
    Loads a file and converts it to a normalized Log-Mel Spectrogram.
    Crucial Update: Bruits exist in low frequencies. We bound the analysis to 20Hz-2000Hz 
    to prevent the neural network from wasting convolutions on empty high-frequency space.
    """
    y, _ = librosa.load(file_path, sr=sr)
    
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)))
    else:
        y = y[:sr]
        
    # Zoomed-in Frequency Bounds
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmin=20, fmax=2000)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    log_mel_min = log_mel.min()
    log_mel_max = log_mel.max()
    normalized_mel = (log_mel - log_mel_min) / (log_mel_max - log_mel_min + 1e-8)
    
    return normalized_mel

def load_dataset_from_folder(folder_path, label):
    specs = []
    labels = []
    files = glob.glob(os.path.join(folder_path, "*.wav"))
    print(f"Found {len(files)} files in {folder_path}...")
    
    for f in files:
        try:
            spec = process_audio(f)
            specs.append(spec)
            labels.append(label)
        except Exception as e:
            print(f"Skipping {f} due to error: {e}")
            
    return specs, labels

# ==========================================
# 3. MAIN TRAINING PIPELINE
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    healthy_specs, healthy_labels = load_dataset_from_folder("healthy", 0)
    stenotic_specs, stenotic_labels = load_dataset_from_folder("stenosed", 1)
    
    X_healthy = torch.tensor(np.array(healthy_specs), dtype=torch.float32).unsqueeze(1)
    
    all_specs = np.array(healthy_specs + stenotic_specs)
    all_labels = np.array(healthy_labels + stenotic_labels)
    X_all = torch.tensor(all_specs, dtype=torch.float32).unsqueeze(1)
    
    print("\n--- Training Denoising Autoencoder ---")
    dae_model = DenoisingAutoencoder().to(device)
    optimizer = optim.Adam(dae_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    healthy_dataset = TensorDataset(X_healthy)
    train_loader = DataLoader(healthy_dataset, batch_size=16, shuffle=True)
    
    noise_factor = 0.2
    epochs = 50 
    
    dae_model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            x_clean = batch[0].to(device)
            
            x_noisy = x_clean + noise_factor * torch.randn(x_clean.shape).to(device)
            x_noisy = torch.clamp(x_noisy, 0., 1.)
            
            optimizer.zero_grad()
            _, reconstructed = dae_model(x_noisy)
            
            if reconstructed.shape != x_clean.shape:
                 reconstructed = torch.nn.functional.interpolate(reconstructed, size=x_clean.shape[2:])
                 
            loss = criterion(reconstructed, x_clean)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.5f}")
            
    torch.save(dae_model.state_dict(), "dae_weights.pth")
    print("Saved zoomed-in DAE weights to 'dae_weights.pth'")
    
    print("\n--- Extracting Features & Training Random Forest ---")
    dae_model.eval()
    
    features = []
    with torch.no_grad():
        for i in range(len(X_all)):
            x = X_all[i:i+1].to(device) 
            latent, reconstructed = dae_model(x)
            
            if reconstructed.shape != x.shape:
                reconstructed = torch.nn.functional.interpolate(reconstructed, size=x.shape[2:])
            
            mse_loss = criterion(reconstructed, x).item()
            latent_flat = latent.cpu().numpy().flatten()
            
            target_size = 128
            if len(latent_flat) > target_size:
                latent_flat = latent_flat[:target_size]
            else:
                latent_flat = np.pad(latent_flat, (0, target_size - len(latent_flat)))
                
            feature_vector = np.append(latent_flat, mse_loss)
            features.append(feature_vector)
            
    X_features = np.array(features)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_features, all_labels)
    
    joblib.dump(rf_model, "rf_classifier.pkl")
    print("Saved Random Forest to 'rf_classifier.pkl'")
    print("\nPhase 1 Optimization complete! Launch Streamlit now.")

if __name__ == "__main__":
    main()