import numpy as np
import scipy.io.wavfile as wav

def generate_test_wavs(sr=22050, duration=2.0):
    # Time array for 2 seconds of audio
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Healthy: Low-frequency rhythmic flow (simulated by modulated noise)
    healthy = np.random.normal(0, 0.05, len(t)) * np.sin(2 * np.pi * 1.5 * t)
    
    # Stenotic: High-frequency turbulence added to the healthy base
    bruit = 0.5 * np.sin(2 * np.pi * 400 * t) * np.exp(-10 * (t - 1.0)**2)
    stenotic = healthy + bruit
    
    # Write to local directory
    wav.write("test_healthy_AVF.wav", sr, np.float32(healthy))
    wav.write("test_stenotic_AVF.wav", sr, np.float32(stenotic))
    print("Test files generated successfully.")

if __name__ == "__main__":
    generate_test_wavs()