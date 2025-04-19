# Grammar Scoring Engine - Audio-based Grammar Assessment

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.com)

An optimized machine learning solution for scoring grammar quality in spoken audio samples, developed for Kaggle competitions with GPU acceleration support.

## Features

- 🚀 **GPU-Optimized Pipeline**: Leverages CUDA-enabled models through RAPIDS cuML
- 🎙️ **Advanced Audio Analysis**: Extracts 51 acoustic features including:
  - MFCCs (Mel-frequency cepstral coefficients)
  - Spectral characteristics (centroid, bandwidth, rolloff)
  - Speech rhythm metrics (pause rate, tempo)
- 🤖 **Multi-Model Comparison**: Tests Ridge Regression, Random Forest, and Gradient Boosting
- 📊 **Interactive Visualization**: Built-in Streamlit interface for real-time analysis
- ⚡ **Batch Processing**: Handles large datasets with memory-efficient batch feature extraction

## Installation

```bash
# Core requirements
pip install librosa scikit-learn pandas numpy matplotlib seaborn

# GPU support (optional)
pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com

# Streamlit interface (optional)
pip install streamlit soundfile
