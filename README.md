Markdown
# Arteriovenous Fistula (AVF) Stenosis Detection Pipeline 🩺

A clinical-grade, deep learning application designed to non-invasively detect stenosis in Arteriovenous Fistulas (AVF) using acoustic hemodynamics. 

Deployed live via Streamlit Community Cloud.

## 📖 Clinical Background
An Arteriovenous Fistula (AVF) is the gold-standard vascular access for hemodialysis patients. A healthy AVF produces a continuous, low-pitched, rumbling "machinery" bruit, indicative of laminar, high-volume blood flow. 

As neointimal hyperplasia develops (commonly at the venous anastomosis), the lumen narrows, increasing blood flow velocity. This transition from laminar to turbulent flow alters the acoustic signature of the bruit, shifting the acoustic energy toward higher frequencies. This pipeline acts as an early-warning anomaly detection system to flag these frequency shifts before physical flow volume drops or the access fails.

## 🧠 Pipeline Architecture
This tool utilizes a highly specialized two-step machine learning pipeline:

1. **Spectral Transformation & Zoom:**
   Raw audio from a digital stethoscope is processed using `librosa`. The Short-Time Fourier Transform (STFT) maps the audio into a Log-Mel Spectrogram. The frequencies are strictly bounded between **20 Hz and 2000 Hz** to isolate vascular acoustics and eliminate high-frequency environmental noise.

2. **Denoising Autoencoder (DAE) - PyTorch:**
   A convolutional Denoising Autoencoder is trained exclusively on *healthy* vascular flow. When presented with a stenotic bruit, the Autoencoder fails to reconstruct the turbulent high-frequency components, resulting in a spike in the Mean Squared Error (MSE).

3. **Latent Space Classifier - Scikit-Learn:**
   A Random Forest Classifier ingests the 128-dimensional bottleneck (latent space) vector from the Autoencoder, along with the Reconstruction Error (MSE), to output a final diagnostic confidence score.

## 🚀 Running the App Locally

If you want to run this application on your local machine:

**1. Clone the repository:**
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/avf-stenosis-detector.git](https://github.com/YOUR_GITHUB_USERNAME/avf-stenosis-detector.git)
cd avf-stenosis-detector
2. Install dependencies:
Bash
pip install -r requirements.txt
3. Launch Streamlit:
Bash
streamlit run main.py
👤 About the Author
Neel Agarwal is a third-year medical student (MS3) and MD Candidate at The Ohio State University College of Medicine.
Prior to medical school, he served as a Peace Corps volunteer in Mali, West Africa. Neel has a sustained professional and research focus in Urology, Nephrology, and Geriatric Medicine, particularly at the intersection of clinical research and data science. He leads the Geriatric Education and Medicine Initiative for New Internists (G.E.M.I.N.I.) at the OSU College of Medicine.

*(Note: In the "Running the App Locally" section above, just make sure to swap out `YOUR_GITHUB_USERNAME` with your actual GitHub handle before you save it!)*
