# Grammar Scoring Engine for Kaggle Competition - Optimized for Accelerators
# This notebook implements an accelerated solution for scoring grammar in spoken audio samples

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import joblib
from scipy.stats import pearsonr
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from functools import partial
import pickle
import time
import itertools  # Added missing import

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Check for GPU availability
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("No GPU found, using CPU")
    device = torch.device("cpu")

# 1. Data Loading and Exploration
# ------------------------------

# Define paths
train_audio_path = '/kaggle/input/shl-intern-hiring-assessment/Dataset/audios/train'  
test_audio_path = '/kaggle/input/shl-intern-hiring-assessment/Dataset/audios/test'    
train_csv_path = '/kaggle/input/shl-intern-hiring-assessment/Dataset/train.csv'
test_csv_path = '/kaggle/input/shl-intern-hiring-assessment/Dataset/test.csv'
cache_dir = '/kaggle/working/feature_cache'  # Cache directory

# Create cache directory if it doesn't exist
os.makedirs(cache_dir, exist_ok=True)

# Load metadata
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")

# Display first few rows of the training data
print("\nTraining data preview:")
print(train_df.head())

# 2. Optimized Audio Feature Extraction
# -----------------------------------

# Modified extract_features_simple function with improved error handling and timeout
def extract_features_simple(filenames, audio_path):
    """Ultra-simple feature extraction with timeouts and better error handling"""
    import signal
    import contextlib
    
    # Create a timeout context manager
    @contextlib.contextmanager
    def timeout(seconds):
        def handler(signum, frame):
            raise TimeoutError(f"Feature extraction timed out after {seconds} seconds")
        
        original_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)
    
    # Pre-allocate output array with correct shape
    features_array = np.zeros((len(filenames), 51), dtype=np.float32)
    
    for i, filename in enumerate(filenames):
        try:
            file_path = os.path.join(audio_path, filename)
            
            # Skip if file doesn't exist
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
                
            # Check cache first
            cache_file = os.path.join(cache_dir, f"{os.path.splitext(filename)[0]}.npy")
            if os.path.exists(cache_file):
                try:
                    features_array[i, :] = np.load(cache_file)
                    if (i + 1) % 5 == 0 or i == len(filenames) - 1:
                        print(f"Processed {i+1}/{len(filenames)} files (cached)")
                    continue
                except Exception as e:
                    print(f"Cache error for {filename}, recomputing: {str(e)}")
            
            # Process with timeout
            with timeout(30):  # 30 second timeout per file
                # Load audio with more flexible error handling
                try:
                    y, sr = librosa.load(file_path, sr=22050, res_type='kaiser_fast', duration=30)
                    
                    # Skip if audio is empty
                    if len(y) == 0:
                        print(f"Warning: Empty audio in {filename}")
                        continue
                        
                    # Basic features (13 features)
                    features_array[i, 0] = librosa.get_duration(y=y, sr=sr)
                    features_array[i, 1] = np.sum(y**2) / len(y)  # energy
                    features_array[i, 2] = np.mean(librosa.feature.zero_crossing_rate(y=y))
                    features_array[i, 3] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
                    features_array[i, 4] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
                    features_array[i, 5] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
                    
                    # Calculate tempo
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    features_array[i, 6] = tempo
                    
                    # Simple speech rate and pauses (3 features)
                    zcr = librosa.feature.zero_crossing_rate(y)[0]
                    features_array[i, 7] = np.std(zcr)  # speech rate proxy
                    
                    # Calculate pauses
                    rms = librosa.feature.rms(y=y)[0]
                    is_silent = rms < 0.01
                    transitions = np.diff(is_silent.astype(int))
                    num_pauses = np.sum(transitions != 0) / 2  # Count transitions
                    features_array[i, 8] = num_pauses
                    features_array[i, 9] = num_pauses / features_array[i, 0] if features_array[i, 0] > 0 else 0  # pause rate
                    
                    # Simple additional features (3 features)
                    features_array[i, 10] = np.mean(y)
                    features_array[i, 11] = np.std(y)
                    features_array[i, 12] = np.max(np.abs(y))
                    
                    # MFCCs (26 features - 13 means, 13 variances)
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    features_array[i, 13:26] = np.mean(mfccs, axis=1)
                    features_array[i, 26:39] = np.var(mfccs, axis=1)
                    
                    # Chroma (12 features)
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
                    features_array[i, 39:51] = np.mean(chroma, axis=1)
                    
                    # Cache the computed features
                    np.save(cache_file, features_array[i, :])
                    
                except Exception as e:
                    print(f"Error processing audio in {filename}: {str(e)}")
                    # Row already initialized with zeros by np.zeros()
        
        except TimeoutError as e:
            print(f"Timeout while processing {filename}: {str(e)}")
            # Row already initialized with zeros
        except Exception as e:
            print(f"Unexpected error with {filename}: {str(e)}")
            # Row already initialized with zeros
        
        # Print progress
        if (i + 1) % 5 == 0 or i == len(filenames) - 1:
            print(f"Processed {i+1}/{len(filenames)} files")
    
    return features_array

# Modified feature extractor class with batch processing
class BatchAudioFeatureExtractor(BaseEstimator, TransformerMixin):
    """Feature extractor with batch processing to prevent memory issues"""
    
    def __init__(self, audio_path, batch_size=50):
        self.audio_path = audio_path
        self.batch_size = batch_size
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        start_time = time.time()
        filenames = X['filename'].tolist()
        total_files = len(filenames)
        
        print(f"Extracting features for {total_files} files in batches of {self.batch_size}...")
        
        # Initialize output array
        all_features = []
        
        # Process in batches
        for i in range(0, total_files, self.batch_size):
            batch_filenames = filenames[i:min(i+self.batch_size, total_files)]
            print(f"Processing batch {i//self.batch_size + 1}/{(total_files-1)//self.batch_size + 1} ({len(batch_filenames)} files)")
            
            # Extract features for this batch
            batch_features = extract_features_simple(batch_filenames, self.audio_path)
            all_features.append(batch_features)
            
            print(f"Batch {i//self.batch_size + 1} completed ({len(batch_filenames)} files)")
        
        # Combine all batches
        X_features = np.vstack(all_features)
        
        # Handle NaN values
        X_features = np.nan_to_num(X_features)
        
        print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
        return X_features

# Process test data with batch processing
def process_test_data(test_df, test_audio_path, scaler, tuned_model, batch_size=50):
    """Process test data with batch processing and better error handling"""
    print("\nExtracting features for test data...")
    
    # Use the batch processor
    extractor = BatchAudioFeatureExtractor(test_audio_path, batch_size=batch_size)
    
    try:
        # Extract features
        X_test_features = extractor.transform(test_df[['filename']])
        
        # Scale features
        X_test_scaled = scaler.transform(X_test_features)
        
        # Make predictions
        print("\nMaking predictions on test data...")
        test_predictions = tuned_model.predict(X_test_scaled)
        
        # Create submission file
        submission_df = pd.DataFrame({
            'filename': test_df['filename'],
            'label': test_predictions
        })
        
        # Ensure predictions are within the valid range
        submission_df['label'] = submission_df['label'].clip(0, 5)
        
        submission_df.to_csv('submission.csv', index=False)
        print("\nSubmission file created: submission.csv")
        
        return submission_df
        
    except Exception as e:
        print(f"Error during test data processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 3. Model Building with GPU Acceleration
# -------------------------------------

# Split the training data for validation
X = train_df[['filename']]
y = train_df['label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

# Create feature names for reference (used later)
basic_feature_names = [
    'duration', 'energy', 'zero_crossing_rate',
    'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
    'tempo', 'speech_rate_proxy', 'num_pauses', 'pause_rate',
    'harmonic_energy', 'percussive_energy', 'voice_activity_ratio'
]

mfcc_mean_names = [f'mfcc{i+1}_mean' for i in range(13)]
mfcc_var_names = [f'mfcc{i+1}_var' for i in range(13)]
chroma_names = [f'chroma{i+1}' for i in range(12)]

all_feature_names = basic_feature_names + mfcc_mean_names + mfcc_var_names + chroma_names

# Define models to try
models = {
    'Ridge Regression': Ridge(random_state=42, solver='auto'),
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),  # Using all cores
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Optional: Add GPU-accelerated models if available
if USE_GPU:
    try:
        import cuml
        from cuml.ensemble import RandomForestRegressor as cuRFR
        from cuml.linear_model import Ridge as cuRidge
        
        models['GPU Ridge'] = cuRidge(alpha=1.0)
        models['GPU Random Forest'] = cuRFR(random_state=42)
        print("Added GPU-accelerated models from cuML")
    except ImportError:
        print("cuML not available, using CPU models only")

# Extract features for all training data
print("\nExtracting features for all training data...")
# Fixed: Create feature extractor instance properly
feature_extractor = BatchAudioFeatureExtractor(train_audio_path, batch_size=50)
X_all_features = feature_extractor.transform(train_df[['filename']])

# Split features into train and validation sets using the same indices
train_indices = X_train.index
val_indices = X_val.index
X_train_features = X_all_features[train_indices]
X_val_features = X_all_features[val_indices]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_val_scaled = scaler.transform(X_val_features)

# Save scaler for later use
joblib.dump(scaler, os.path.join(cache_dir, 'scaler.pkl'))

best_rmse = float('inf')
best_model_name = None
best_model = None
results = {}

# Train and evaluate models
for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    
    # Fit the model directly on scaled features
    model.fit(X_train_scaled, y_train)
    
    # Predict on validation set
    y_val_pred = model.predict(X_val_scaled)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae = mean_absolute_error(y_val, y_val_pred)
    corr, _ = pearsonr(y_val, y_val_pred)
    
    training_time = time.time() - start_time
    
    # Store results
    results[name] = {
        'RMSE': rmse, 
        'MAE': mae, 
        'Pearson Correlation': corr,
        'Training Time': f"{training_time:.2f}s"
    }
    
    print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, Correlation: {corr:.4f}, Time: {training_time:.2f}s")
    
    # Save model
    joblib.dump(model, os.path.join(cache_dir, f"{name.replace(' ', '_').lower()}_model.pkl"))
    
    # Check if this is the best model so far
    if rmse < best_rmse:
        best_rmse = rmse
        best_model_name = name
        best_model = model

# Display results in a DataFrame
results_df = pd.DataFrame(results).T
print("\nModel comparison:")
print(results_df)

# 4. Hyperparameter Tuning - Optimized
# ----------------------------------

print(f"\nFine-tuning the best model: {best_model_name}")

# Check if we're using a GPU model
is_gpu_model = 'GPU' in best_model_name

if 'Ridge' in best_model_name:
    param_grid = {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    }
elif 'Random Forest' in best_model_name:
    # Simplified parameter grid for faster tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5]
    }
else:  # Gradient Boosting
    # Simplified parameter grid for faster tuning
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }

if is_gpu_model:
    # Manual grid search for GPU models (since GridSearchCV might not work with cuML)
    print("Performing manual grid search for GPU model...")
    
    best_score = float('-inf')
    best_params = {}
    best_tuned_model = None
    
    for param_combination in [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]:
        print(f"Testing parameters: {param_combination}")
        
        # Create model with these parameters
        if 'Ridge' in best_model_name:
            model = cuml.linear_model.Ridge(**param_combination)
        else:  # Random Forest
            model = cuml.ensemble.RandomForestRegressor(random_state=42, **param_combination)
        
        # Train and evaluate
        model.fit(X_train_scaled, y_train)
        y_val_pred = model.predict(X_val_scaled)
        score = -mean_squared_error(y_val, y_val_pred)  # Negative MSE for consistency with GridSearchCV
        
        print(f"Score: {score}")
        
        if score > best_score:
            best_score = score
            best_params = param_combination
            best_tuned_model = model
    
    print(f"Best parameters: {best_params}")
    print(f"Best score (neg MSE): {best_score:.4f}")
    
    tuned_model = best_tuned_model
    
else:
    # Standard GridSearchCV for CPU models
    grid_search = GridSearchCV(
        models[best_model_name],
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score (neg MSE): {grid_search.best_score_:.4f}")
    
    tuned_model = grid_search.best_estimator_

# Evaluate the tuned model
y_val_pred_tuned = tuned_model.predict(X_val_scaled)

rmse_tuned = np.sqrt(mean_squared_error(y_val, y_val_pred_tuned))
mae_tuned = mean_absolute_error(y_val, y_val_pred_tuned)
corr_tuned, _ = pearsonr(y_val, y_val_pred_tuned)

print(f"\nTuned model - RMSE: {rmse_tuned:.4f}, MAE: {mae_tuned:.4f}, Correlation: {corr_tuned:.4f}")

# Save the tuned model
joblib.dump(tuned_model, os.path.join(cache_dir, 'tuned_model.pkl'))

# 5. Final Model Training and Prediction
# -----------------------------------

# Use process_test_data for consistent feature extraction and prediction
print("\nProcessing test data for final prediction...")
submission_df = process_test_data(test_df, test_audio_path, scaler, tuned_model, batch_size=20)

# Format the submission values properly
if submission_df is not None:
    submission_df['label'] = submission_df['label'].clip(0, 5).round(4)
    submission_df = submission_df[['filename', 'label']]
    
    submission_df.to_csv('submission.csv', 
                         index=False, 
                         encoding='utf-8-sig', 
                         float_format='%.4f')
    print("Final submission file created successfully.")

# 6. Feature Importance Analysis (if applicable)
# ------------------------------------------

if hasattr(tuned_model, 'feature_importances_'):
    # Get feature importances
    feature_importances = tuned_model.feature_importances_
    
    # Create DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    # Plot top features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')  # Save plot to file
    plt.close()
    
    print("\nTop 10 most important features:")
    print(importance_df.head(10))
    print("Feature importance plot saved as 'feature_importance.png'")

# 7. Summary and Conclusion
# ----------------------
print("\n" + "="*50)
print("GRAMMAR SCORING ENGINE - OPTIMIZED SUMMARY")
print("="*50)
print(f"Total training samples: {train_df.shape[0]}")
print(f"Total test samples: {test_df.shape[0]}")
print(f"Best model: {best_model_name} (Tuned)")
print(f"GPU Acceleration: {'Enabled' if USE_GPU else 'Disabled'}")
print(f"Validation RMSE: {rmse_tuned:.4f}")
print(f"Validation Pearson Correlation: {corr_tuned:.4f}")
print("="*50)
