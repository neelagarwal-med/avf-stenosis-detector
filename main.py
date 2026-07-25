import streamlit as st
import numpy as np
import librosa
import librosa.display
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tempfile
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AVF Stenosis Detector",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# NEURAL NETWORK ARCHITECTURE
# ==========================================
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Decoder
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
# CACHED MODEL LOADING & PROCESSING
# ==========================================
@st.cache_resource
def load_models():
    """Loads the trained PyTorch DAE weights and Random Forest classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load trained PyTorch DAE
    dae_model = DenoisingAutoencoder().to(device)
    dae_model.load_state_dict(torch.load('dae_weights.pth', map_location=device))
    dae_model.eval()
    
    # 2. Load trained Random Forest Classifier
    rf_model = joblib.load('rf_classifier.pkl')
    
    return dae_model, rf_model, device

def process_audio(file_path, sr=22050, n_mels=64):
    """Loads audio and converts to normalized Log-Mel Spectrogram."""
    y, sr = librosa.load(file_path, sr=sr)
    
    # Ensure audio is at least 1 second long (pad if necessary)
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)))
    else:
        y = y[:sr] # truncate to 1 second for consistent input size
        
    # MODIFIED: Added fmin=20 and fmax=2000 for clinical acoustic zoom
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmin=20, fmax=2000)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Min-Max Normalization
    log_mel_min = log_mel.min()
    log_mel_max = log_mel.max()
    normalized_mel = (log_mel - log_mel_min) / (log_mel_max - log_mel_min + 1e-8)
    
    return normalized_mel, y, sr

def generate_spectrogram_plot(original, reconstructed, sr):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # MODIFIED: Added fmin=20 and fmax=2000 so the Y-axis correctly displays the clinical zoom
    img1 = librosa.display.specshow(original, sr=sr, x_axis='time', y_axis='mel', fmin=20, fmax=2000, ax=ax[0])
    fig.colorbar(img1, ax=ax[0], format='%+2.0f dB')
    ax[0].set_title('Original Uploaded Bruit')
    
    img2 = librosa.display.specshow(reconstructed, sr=sr, x_axis='time', y_axis='mel', fmin=20, fmax=2000, ax=ax[1])
    fig.colorbar(img2, ax=ax[1], format='%+2.0f dB')
    ax[1].set_title('DAE Filtered/Reconstructed')
    
    plt.tight_layout()
    return fig

# ==========================================
# MAIN APP UI
# ==========================================
st.title("Arteriovenous Fistula (AVF) Stenosis Detection Pipeline")
st.markdown("A Deep Learning approach to non-invasive vascular monitoring using Denoising Autoencoders.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎙️ Upload & Analyze", 
    "🧮 Methodology & Math", 
    "🩺 Clinical Interpretation", 
    "📚 References", 
    "👤 About the Author"
])

dae_model, rf_model, device = load_models()

# --- TAB 1: UPLOAD & ANALYZE ---
with tab1:
    st.header("Acoustic Bruit Analysis")
    
    # MODIFIED: Added side-by-side columns to support both File Upload and Live Microphone Capture
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        uploaded_file = st.file_uploader("Upload AVF Audio File (Stethoscope recording)", type=["wav", "mp3", "m4a", "ogg"])
    with col_input2:
        recorded_file = st.audio_input("Or Record via Stethoscope/Mic")
        
    # Determine which source to use (prioritize live recording if both exist)
    audio_source = recorded_file if recorded_file is not None else uploaded_file
    
    if audio_source is not None:
        st.audio(audio_source)
        
        # FIX 2: Reset the file pointer after st.audio reads it
        audio_source.seek(0)
        
        with st.spinner('Processing signal and running inference...'):
            # FIX 3: Get the actual file extension to prevent Librosa/audioread decode errors
            file_extension = os.path.splitext(audio_source.name)[1] if hasattr(audio_source, 'name') else '.wav'
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(audio_source.read())
                tmp_file_path = tmp_file.name
            
            try:
                # 1. Preprocess
                spec, raw_y, sr = process_audio(tmp_file_path)
                
                # 2. Tensor Conversion
                x_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                
                # 3. Model Inference
                with torch.no_grad():
                    latent, reconstructed = dae_model(x_tensor)
                    
                    if reconstructed.shape != x_tensor.shape:
                        reconstructed = torch.nn.functional.interpolate(reconstructed, size=x_tensor.shape[2:])
                    
                    # Calculate MSE
                    criterion = nn.MSELoss()
                    mse_loss = criterion(reconstructed, x_tensor).item()
                    
                    # Format features for RF
                    latent_flat = latent.cpu().numpy().flatten()
                    
                    # Truncate or Pad to explicitly match the 128 size expectation
                    target_size = 128
                    if len(latent_flat) > target_size:
                        latent_flat = latent_flat[:target_size]
                    else:
                        latent_flat = np.pad(latent_flat, (0, target_size - len(latent_flat)))
                        
                    feature_vector = np.append(latent_flat, mse_loss).reshape(1, -1)
                    
                    # Predict
                    prediction = rf_model.predict(feature_vector)[0]
                    prediction_prob = rf_model.predict_proba(feature_vector)[0]
            
            finally:
                # Clean up the temp file
                os.remove(tmp_file_path)
                
            # 4. Display Results
            st.subheader("Diagnostic Results")
            col1, col2 = st.columns(2)
            
            status = "Stenotic / Abnormal" if prediction == 1 else "Healthy / Normal"
            color = "red" if prediction == 1 else "green"
            
            col1.markdown(f"**Classification:** <span style='color:{color}; font-size:24px'>{status}</span>", unsafe_allow_html=True)
            col1.metric("Reconstruction Error (MSE)", f"{mse_loss:.5f}")
            
            col2.metric("Confidence Score", f"{max(prediction_prob) * 100:.1f}%")
            
            st.subheader("Spectral Analysis")
            fig = generate_spectrogram_plot(spec, reconstructed.squeeze().cpu().numpy(), sr)
            st.pyplot(fig)

# --- TAB 2: METHODOLOGY ---
with tab2:
    st.header("Methodology: Signal Processing & Deep Learning")
    
    st.subheader("1. Spectral Transformation")
    st.markdown("""
    Raw acoustic waveforms from a stethoscope are difficult to interpret computationally due to background noise and complex overlapping frequencies. We utilize the **Short-Time Fourier Transform (STFT)** to transition from the time domain to the frequency domain, mapping the data into a Log-Mel Spectrogram.
    """)
    st.latex(r"X(m, k) = \sum_{n=0}^{N-1} x(n + mH) w(n) e^{-j 2\pi kn/N}")
    st.markdown("""
    Where $x(n)$ is the raw signal, $w(n)$ is the window function, and $H$ is the hop length. This maps human-audible frequencies to the Mel scale, emphasizing the relevant vascular acoustic bands.
    """)
    
    st.subheader("2. The Denoising Autoencoder (DAE)")
    st.markdown("""
    The DAE acts as an intelligent, non-linear filter trained exclusively on **healthy** vascular hemodynamics. During training, synthetic noise is added to the healthy signal, and the model attempts to reconstruct the clean signal. 
    
    The Signal-to-Noise Ratio defines the perturbation during training:
    """)
    st.latex(r"SNR_{dB} = 10\log_{10}\left(\frac{P_{signal}}{P_{noise}}\right)")
    
    # MODIFIED: Added the 'r' prefix here to prevent the Python 3.12 escape sequence warning
    st.markdown(r"""
    The autoencoder minimizes the **Mean Squared Error (MSE)** between the clean input $x_i$ and the reconstructed output $\hat{x}_i$:
    """)
    st.latex(r"L(\theta) = \frac{1}{n}\sum_{i=1}^{n}\|x_i - f_\theta(\tilde{x}_i)\|^2")
    
    st.subheader("3. Feature Extraction and Classification")
    st.markdown("""
    Because the model only understands "healthy" blood flow, the presence of a stenotic bruit (high-frequency turbulence) causes a massive spike in the Reconstruction Error (MSE). We extract the latent space vector from the DAE's bottleneck layer, append the MSE, and pass this combined tensor into a Random Forest Classifier to make the final clinical distinction.
    """)

# --- TAB 3: CLINICAL INTERPRETATION ---
with tab3:
    st.header("Clinical Interpretation & Significance")
    st.markdown("""
    ### Hemodynamics of an AVF
    An Arteriovenous Fistula (AVF) is the gold-standard vascular access for hemodialysis patients. A healthy AVF produces a continuous, low-pitched, rumbling "machinery" bruit, indicative of laminar, high-volume flow. 

    ### The Pathology of Stenosis
    As neointimal hyperplasia develops (commonly at the venous anastomosis), the lumen narrows, increasing the velocity of the blood flow. This transition from laminar to turbulent flow alters the acoustic signature of the bruit, shifting the energy toward higher frequencies—often before changes in flow volume are detected on standard dialysis machine venous pressure monitors.
    
    ### Interpreting Pipeline Metrics
    * **Low MSE (Healthy):** The DAE successfully recognized and reconstructed the signal. The flow is likely laminar. Continue standard clinical monitoring.
    * **High MSE (Stenotic):** The DAE failed to reconstruct the signal, indicating anomalous high-frequency components. This is a massive "red flag" for a developing clot or stricture. 
    
    ### Recommended Next Steps for High MSE
    If the pipeline flags a sample as **Stenotic**, the provider should consider:
    1.  **Physical Examination:** Re-evaluate for abnormal thrill (e.g., water-hammer pulse) or prolonged bleeding post-dialysis.
    2.  **Duplex Ultrasound:** To quantify blood flow volume (Qa) and visualize the anatomical narrowing.
    3.  **Fistulogram:** The gold-standard angiographic assessment and potential intervention (angioplasty) to salvage the access.
    """)

# --- TAB 4: REFERENCES ---
with tab4:
    st.header("References")
    st.markdown("""
    1.  **KDOQI Clinical Practice Guideline for Vascular Access:** 2019 Update. American Journal of Kidney Diseases. Provides guidelines on routine physical examination and screening for access dysfunction.
    2.  Wang, C., et al. (2020). *Non-invasive detection of arteriovenous fistula stenosis using acoustic analysis and machine learning.* Journal of Vascular Access.
    3.  Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press. (Foundational architecture for Autoencoders and representation learning).
    4.  McFee, B., et al. (2015). *librosa: Audio and Music Signal Analysis in Python.* Proceedings of the 14th Python in Science Conference.
    """)

# --- TAB 5: ABOUT THE AUTHOR ---
with tab5:
    st.header("About the Author")
    st.markdown("""
    **Neel Agarwal** is a third-year medical student (MS3) and MD Candidate at The Ohio State University College of Medicine. 
    
    Neel has a sustained professional and research focus in Urology, Nephrology, and Geriatric Medicine, particularly at the intersection of clinical research and data science. 
    
    He is highly proficient in R and Python, utilizing these languages for medical informatics, statistical modeling, and the development of clinical tools. He has developed and launched several healthcare applications using Streamlit, including the Geriatric Desert Mapper, the NephroFlow Pad Optimizer, and the Nocturia Risk Dashboard. Currently, he leads the Geriatric Education and Medicine Initiative for New Internists (G.E.M.I.N.I.) at the OSU College of Medicine.
    """)