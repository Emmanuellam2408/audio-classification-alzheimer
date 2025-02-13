# Cognitive Decline Speech Analyzer - ML Alzheimer's Detection
ML project for binary classification of patient recordings with probable Alzheimer's disease.


## Overview

This project analyzes speech patterns from audio recordings to identify potential indicators of cognitive decline. It extracts various acoustic and linguistic features from speech and visualizes the results through statistical analysis and machine learning techniques.

## Key Features:

- Fluency & Pause Analysis: Measures speech fluency, pause frequency, and segment duration.

- Pitch & Voice Quality Metrics: Evaluates pitch variability, entropy, and tremor markers.

- Linguistic Complexity Analysis: Uses Mel Frequency Cepstral Coefficients (MFCCs) to assess speech complexity.

- Spectral Features: Analyzes spectral centroid and variability.

- Parallel Processing: Uses multiprocessing to speed up audio feature extraction.

- Data Visualization: Generates heatmaps, scatter plots, and histograms to interpret cognitive risk factors

## Prerequisites

Ensure you have Python 3.7+ installed on your system

## Usage

### Running the Analysis

1. Place the audio files (.mp3 format) in the `train/` directory.
2. Execute the following command:

`python CDS.py`

## Output

- A risk analysis report is saved as `cognitive_decline_risk_report.csv`

- A visualization image of speech analysis is saved as `cognitive_decline_analysis.png`

- Console output summarizes high-risk speech patterns.

## Future Enhancements

- Additional Speech Features: Jitter, Shimmer, Harmonic-to-Noise Ratio (HNR), Formants (F1-F3), and Spectral Roll-off.

- Machine Learning Integration: Train models to predict cognitive decline based on speech features.

- Speech-to-Text Analysis: Use NLP techniques to analyze linguistic complexity.

- Web-Based Dashboard: Interactive UI to visualize results in real time.

