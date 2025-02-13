import os
import numpy as np
import pandas as pd
import librosa
import scipy.stats as stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class CognitiveDeclineSpeechAnalyzer:
    @staticmethod
    def extract_cognitive_speech_features(audio_path):
        """
        Emma
        Extract speech features from an audio file that may indicate cognitive decline.
        The features include fluency, pitch variability, linguistic complexity, and spectral characteristics.
        """
        try:
            # Load the audio file (up to 60 seconds) with a sample rate of 8000 Hz
            y, sr = librosa.load(audio_path, duration=60, sr=8000)

            # Dictionary to store the extracted features
            features = {}

            # 1. Fluency and Pause Indicators
            # Root Mean Square (RMS) energy: to measure speech volume
            rms = librosa.feature.rms(y=y)[0]
            # Zero-crossing rate: measure of how frequently the signal changes sign
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]

            # Detect onsets (when speech starts or pauses)
            onsets = librosa.onset.onset_detect(y=y, sr=sr)
            # Calculate the intervals between onsets to identify pauses in speech
            pause_intervals = np.diff(onsets) / sr

            # Add pause-related and fluency features
            features.update({
                'mean_segment_duration': np.mean(pause_intervals),  # Average pause duration
                'segment_duration_variation': np.std(pause_intervals),  # Standard deviation of pause duration
                'pause_frequency': len(onsets) / (len(y) / sr),  # Frequency of pauses per second

                # Speech stability features
                'amplitude_stability': np.std(rms),  # Variability in speech loudness
                'zero_crossing_variation': np.std(zero_crossing_rate)  # Variation in the zero-crossing rate
            })

            # 2. Pitch and Voice Quality Markers
            # Pitch estimation using YIN algorithm (between C2 and C7 in Hz)
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'),
                             fmax=librosa.note_to_hz('C7'))
            f0 = f0[~np.isnan(f0)]  # Remove NaN values from pitch estimates

            # If pitch is detected, calculate features related to pitch
            if len(f0) > 0:
                features.update({
                    'pitch_entropy': stats.entropy(f0),  # Entropy of pitch values (measures unpredictability)
                    'pitch_variability': np.std(f0),  # Variability of pitch (range)
                    'pitch_range': np.ptp(f0),  # Range of pitch (max - min)

                    # Pitch distribution skewness and kurtosis as tremor indicators
                    'pitch_skewness': stats.skew(f0),  # Asymmetry in pitch distribution
                    'pitch_kurtosis': stats.kurtosis(f0)  # Peakedness of pitch distribution
                })

            # 3. Linguistic Complexity Proxies
            # Extract 13 MFCCs (Mel Frequency Cepstral Coefficients) that capture speech characteristics
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.update({
                'mfcc_complexity': np.mean(np.std(mfccs, axis=1)),  # Complexity based on MFCC standard deviation
                'mfcc_variability': np.std(np.mean(mfccs, axis=0))  # Variability in MFCC means across frames
            })

            # 4. Spectral Features
            # Spectral Centroid measures the "brightness" of a sound
            spectral_centroids = librosa.feature.spectral_centroid(y=y)[0]
            features.update({
                'spectral_variability': np.std(spectral_centroids),  # Variability in spectral centroid
                'spectral_irregularity': stats.skew(spectral_centroids)  # Skewness in spectral centroid
            })

            return features  # Return the extracted features

        except Exception as e:
            # In case of an error (e.g., file issues), print the error and return None
            print(f"Error processing {audio_path}: {e}")
            return None


def process_audio_files_in_parallel(folder_path, max_workers=None):
    """
    Emma
    Process all audio files in the train folder in parallel to extract cognitive speech features.
    The function uses multiprocessing to speed up the processing.
    """
    # List all mp3 files in the given folder
    audio_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.mp3')  # Only consider mp3 files
    ]

    # Set the number of workers to use for parallel processing (default to CPU count)
    max_workers = max_workers or os.cpu_count()

    results = []  # List to store results
    # Use a ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor to process each file in parallel
        future_to_file = {
            executor.submit(CognitiveDeclineSpeechAnalyzer.extract_cognitive_speech_features, file): file
            for file in audio_files
        }

        # Collect results as each task finishes
        for future in as_completed(future_to_file):
            result = future.result()
            if result is not None:
                result['filename'] = os.path.basename(future_to_file[future])  # Add filename to result
                results.append(result)  # Add the result to the list

    # Convert results list into a DataFrame
    return pd.DataFrame(results)


def create_cognitive_decline_visualizations(df):
    """
    Emma
    Create visualizations to illustrate speech patterns related to potential cognitive decline.
    This includes heatmaps, scatter plots, and histograms.
    """
    # Set up the figure size for the plots
    plt.figure(figsize=(24, 18))
    plt.suptitle('Cognitive Decline Speech Pattern Analysis', fontsize=20)

    # Define the key features to assess cognitive decline
    decline_features = [
        'mean_segment_duration',
        'pause_frequency',
        'pitch_variability',
        'pitch_entropy',
        'mfcc_complexity',
        'spectral_variability'
    ]

    # 1. Cognitive Decline Risk Heatmap
    plt.subplot(2, 3, 1)
    # Normalize features for visualization
    risk_data = (df[decline_features] - df[decline_features].mean()) / df[decline_features].std()
    risk_score = risk_data.sum(axis=1)  # Calculate a cumulative risk score for each file

    # Create a heatmap of the normalized features
    sns.heatmap(risk_data.T, cmap='YlOrRd', center=0,
                yticklabels=decline_features,
                cbar_kws={'label': 'Deviation from Normal'})
    plt.title('Potential Cognitive Decline Indicators')

    # 2. Pause Characteristics Distribution
    plt.subplot(2, 3, 2)
    # Boxplot for the mean segment duration and pause frequency
    sns.boxplot(data=df[['mean_segment_duration', 'pause_frequency']])
    plt.title('Speech Pause Characteristics')
    plt.xticks(rotation=45)

    # 3. Pitch Variability vs. Pitch Entropy
    plt.subplot(2, 3, 3)
    plt.scatter(df['pitch_variability'], df['pitch_entropy'],
                c=risk_score, cmap='coolwarm')
    plt.xlabel('Pitch Variability')
    plt.ylabel('Pitch Entropy')
    plt.title('Pitch Stability Indicators')
    plt.colorbar(label='Cognitive Decline Risk')

    # 4. MFCC Complexity vs. Spectral Variability
    plt.subplot(2, 3, 4)
    plt.scatter(df['mfcc_complexity'], df['spectral_variability'],
                c=risk_score, cmap='YlOrRd')
    plt.xlabel('MFCC Complexity')
    plt.ylabel('Spectral Variability')
    plt.title('Linguistic Complexity Markers')
    plt.colorbar(label='Cognitive Decline Risk')

    # 5. Distribution of Risk Scores
    plt.subplot(2, 3, 5)
    sns.histplot(risk_score, kde=True)
    plt.title('Distribution of Cognitive Decline Risk Scores')
    plt.xlabel('Risk Score')

    # 6. Top 10 High-Risk Files
    plt.subplot(2, 3, 6)
    top_risk_files = df.iloc[risk_score.nlargest(10).index]
    plt.bar(top_risk_files['filename'], risk_score.nlargest(10))
    plt.title('Top 10 High-Risk Files')
    plt.xticks(rotation=90)

    # Tight layout to avoid overlap of plots
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('cognitive_decline_analysis.png', dpi=300)  # Save the figure as a PNG image

    # Generate a detailed risk report CSV
    risk_report = pd.DataFrame({
        'filename': df['filename'],
        'risk_score': risk_score
    }).sort_values('risk_score', ascending=False)

    risk_report.to_csv('cognitive_decline_risk_report.csv', index=False)  # Save the risk report

    return risk_report


def main(folder_path):
    """
    Main function
    """
    print(f"Starting cognitive speech pattern analysis on folder: {folder_path}")

    # Process audio files and extract features
    df = process_audio_files_in_parallel(folder_path)

    # Generate visualizations and create a risk report
    risk_report = create_cognitive_decline_visualizations(df)

    # Print analysis summary
    print("\nCognitive Decline Analysis Summary:")
    print("-" * 50)
    print(f"Total Files Analyzed: {len(df)}")
    print("\nTop 5 High-Risk Files:")
    print(risk_report.head())

    return df, risk_report


if __name__ == "__main__":
    folder_path = "train"  # Path to the folder containing the audio files
    main(folder_path)
