import os  # Importing os for file and directory operations
import pandas as pd  # Importing pandas for handling data in tabular format
import numpy as np  # Importing numpy for numerical operations
from pyAudioAnalysis import MidTermFeatures as aF  # Importing mid-term feature extraction functions
from pyAudioAnalysis import audioBasicIO  # Importing audio file reading utilities

def extract_mfcc_features_to_csv(audios):
    """
    Extract MFCC and other audio features from a folder of MP3 files and save them to a CSV file.

    Args:
        audios (str): Path to the folder containing audio files.
    """
    
    print("Extracting MFCC features...")

    # List all MP3 files in the specified folder
    audio_files = [os.path.join(audios, f) for f in os.listdir(audios) if f.endswith('.mp3')]

    # Check if there are valid audio files
    if not audio_files:
        raise ValueError("No MP3 files found in the specified folder.")

    all_features = []  # List to store extracted features from each file
    file_names = []  # List to store corresponding file names

    for file_path in audio_files:
        # Read the audio file (returns sampling rate and signal data)
        sampling_rate, signal = audioBasicIO.read_audio_file(file_path)

        if signal is None:
            print(f"Error reading file {file_path}, skipping...")
            continue  # Skip processing for unreadable files

        # Convert stereo audio to mono (if applicable)
        signal = audioBasicIO.stereo_to_mono(signal)

        """
        Extract mid-term features from the audio.
        Mid-term features summarize characteristics of the audio over larger time windows compared to short-term features.
        These features can include spectral and energy-related properties.
        
        Parameters:
        - Window size: 45 seconds (converted to sample frames)
        - Step size: 22.5 seconds (half of the window size)
        - Short-term window and step: 0.05 seconds
        """

        mt_features, _, feature_names = aF.mid_feature_extraction(
            signal, sampling_rate, 
            45.0 * sampling_rate,  # Mid-term window length
            22.5 * sampling_rate,  # Mid-term step size
            0.05 * sampling_rate,  # Short-term window length
            0.05 * sampling_rate   # Short-term step size
        )

        # Compute the average feature values across windows
        avg_features = np.mean(mt_features, axis=1)

        # Store extracted features and the corresponding filename
        all_features.append(avg_features)
        file_names.append(os.path.basename(file_path))

    # Create a DataFrame with extracted features
    mfcc = pd.DataFrame(all_features, columns=feature_names)
    mfcc.insert(0, 'File', file_names)  # Insert file names as the first column

    # Save extracted features to a CSV file
    mfcc.to_csv("mfcc_45_sec.csv", index=False)

    print(f"Features saved to mfcc_45_sec.csv")
