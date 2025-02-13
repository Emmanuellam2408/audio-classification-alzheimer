import os
import numpy as np
import pandas as pd
import librosa
import scipy.stats as stats
import xgboost as xgb
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')


class EnhancedCognitiveDeclineSpeechAnalyzer:
    @staticmethod
    def extract_cognitive_speech_features(audio_path):
        """
        Enhanced feature extraction from speech to detect cognitive decline
        """
        try:
            y, sr = librosa.load(audio_path, sr=16000)  # Higher sampling rate
            features = {}

            # Extracting fluency, pause, and spectral features
            rms = librosa.feature.rms(y=y)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            onsets = librosa.onset.onset_detect(y=y, sr=sr)
            pause_intervals = np.diff(onsets) / sr

            features['mean_segment_duration'] = np.mean(pause_intervals)
            features['pause_frequency'] = len(onsets) / (len(y) / sr)

            # Pitch features with additional voice tremor analysis
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                                         fmax=librosa.note_to_hz('C7'))
            features['pitch_variability'] = np.nanstd(f0)
            features['harmonics_to_noise_ratio'] = np.mean(librosa.effects.harmonic(y) / librosa.effects.percussive(y))

            # Formant-based features (placeholder, can be extended with more advanced formant extraction techniques)

            # MFCCs for linguistic complexity
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_complexity'] = np.mean(np.std(mfccs, axis=1))

            # Harmonics-to-Noise Ratio (HNR)
            features['hnr'] = np.mean(librosa.effects.harmonic(y) / librosa.effects.percussive(y))

            # Extracting Jitter and Shimmer (frequency and amplitude variations)
            jitter = np.std(f0)
            shimmer = np.std(rms)
            features['jitter'] = jitter
            features['shimmer'] = shimmer

            # Spectral entropy for irregularity
            spectral_centroids = librosa.feature.spectral_centroid(y=y)[0]
            features['spectral_entropy'] = stats.entropy(spectral_centroids)

            return features
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None


def process_audio_files_in_parallel(folder_path, max_workers=None):
    """
    Process audio files with enhanced feature extraction
    """
    audio_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.mp3')
    ]

    max_workers = max_workers or os.cpu_count()

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(EnhancedCognitiveDeclineSpeechAnalyzer.extract_cognitive_speech_features, file): file
            for file in audio_files
        }

        for future in as_completed(future_to_file):
            result = future.result()
            if result is not None:
                result['filename'] = os.path.basename(future_to_file[future])
                results.append(result)

    return pd.DataFrame(results)


def create_cognitive_decline_visualizations(df):
    """
    Create specialized visualizations for potential cognitive decline indicators
    """
    plt.figure(figsize=(24, 18))
    plt.suptitle('Cognitive Decline Speech Pattern Analysis', fontsize=20)

    # Select cognitive decline relevant features
    decline_features = [
        'mean_segment_duration',
        'pause_frequency',
        'pitch_variability',
        'harmonics_to_noise_ratio',
        'mfcc_complexity',
        'spectral_entropy'
    ]

    # 1. Potential Cognitive Decline Risk Heatmap
    plt.subplot(2, 3, 1)
    # Normalize features for risk scoring
    risk_data = (df[decline_features] - df[decline_features].mean()) / df[decline_features].std()
    risk_score = risk_data.sum(axis=1)

    sns.heatmap(risk_data.T, cmap='YlOrRd', center=0,
                yticklabels=decline_features,
                cbar_kws={'label': 'Deviation from Normal'})
    plt.title('Potential Cognitive Decline Indicators')

    # 2. Pause Characteristics Distribution
    plt.subplot(2, 3, 2)
    sns.boxplot(data=df[['mean_segment_duration', 'pause_frequency']])
    plt.title('Speech Pause Characteristics')
    plt.xticks(rotation=45)

    # 3. Pitch Variability vs Harmonics-to-Noise Ratio
    plt.subplot(2, 3, 3)
    plt.scatter(df['pitch_variability'], df['harmonics_to_noise_ratio'],
                c=risk_score, cmap='coolwarm')
    plt.xlabel('Pitch Variability')
    plt.ylabel('Harmonics-to-Noise Ratio')
    plt.title('Pitch and Voice Quality Indicators')
    plt.colorbar(label='Cognitive Decline Risk')

    # 4. MFCC Complexity and Spectral Entropy
    plt.subplot(2, 3, 4)
    plt.scatter(df['mfcc_complexity'], df['spectral_entropy'],
                c=risk_score, cmap='YlOrRd')
    plt.xlabel('MFCC Complexity')
    plt.ylabel('Spectral Entropy')
    plt.title('Linguistic and Spectral Complexity')
    plt.colorbar(label='Cognitive Decline Risk')

    # 5. Risk Score Distribution
    plt.subplot(2, 3, 5)
    sns.histplot(risk_score, kde=True)
    plt.title('Distribution of Cognitive Decline Risk Scores')
    plt.xlabel('Risk Score')

    # 6. Top Risk Files Identification
    plt.subplot(2, 3, 6)
    top_risk_files = df.iloc[risk_score.nlargest(10).index]
    plt.bar(top_risk_files['filename'], risk_score.nlargest(10))
    plt.title('Top 10 High-Risk Files')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('cognitive_decline_analysis.png', dpi=300)

    # Create a detailed risk report
    risk_report = pd.DataFrame({
        'filename': df['filename'],
        'risk_score': risk_score
    }).sort_values('risk_score', ascending=False)

    risk_report.to_csv('cognitive_decline_risk_report.csv', index=False)

    return risk_report


def main(folder_path):
    """
    Main processing pipeline for cognitive decline detection
    """
    print(f"Starting cognitive speech pattern analysis on folder: {folder_path}")

    # Process files
    df = process_audio_files_in_parallel(folder_path)

    # Create visualizations and generate risk report
    risk_report = create_cognitive_decline_visualizations(df)

    # Print summary
    print("\nCognitive Decline Analysis Summary:")
    print("-" * 50)
    print(f"Total Files Analyzed: {len(df)}")
    print("\nTop 5 High-Risk Files:")
    print(risk_report.head())

    return df, risk_report


if __name__ == "__main__":
    folder_path = "train"
    df, risk_report = main(folder_path)

    # Train a model on the extracted features
    X = df.drop(['filename', 'risk_score'], axis=1)
    y = pd.cut(risk_report['risk_score'], bins=[-np.inf, 0, np.inf], labels=[0, 1])  # Binary classification

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a model
    model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\nModel Evaluation Report:")
    print(classification_report(y_test, y_pred))

    # Save model for later use
    model.save_model("cognitive_decline_model.xgb")
