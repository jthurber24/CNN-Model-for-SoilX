#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


#complete interpolation of moisture data for complete dataset for data preprocessing

# Load the dataset
file_path = 'C:/Moisture_Probe/Moisture_Original.xlsx'
df = pd.read_excel(file_path, index_col='Date & Time').transpose()

# Convert the index to datetime to ensure proper time series handling
df.index = pd.to_datetime(df.index, errors='coerce')

# Interpolate missing values linearly
df_interpolated = df.interpolate(method='linear', axis=0)  # Interpolates along the date axis

# For areas where interpolation might not fill (e.g., at the start or end)
df_interpolated.ffill(inplace=True)  # Forward fill
df_interpolated.bfill(inplace=True)  # Backward fill

print(df_interpolated.head())

output_csv_path = 'C:/Moisture_Probe/Moisture_Interpolated.csv'

# Save the interpolated DataFrame to a CSV file
df_interpolated.to_csv(output_csv_path)

print(f'DataFrame saved to {output_csv_path}')


# In[2]:


import pandas as pd

# Load the moisture probe data
moisture_probe_data_path = 'C:/Moisture_Probe/Moisture_Interpolated.csv'
moisture_probe_data = pd.read_csv(moisture_probe_data_path)


percentage_columns = [col for col in moisture_probe_data.columns if col.endswith('%')]
if percentage_columns:
    moisture_probe_data['Average_Moisture_%'] = moisture_probe_data[percentage_columns].mean(axis=1)
else:
    # This placeholder column name should be replaced with the actual moisture column name
    moisture_probe_data['Average_Moisture_%'] = moisture_probe_data['Moisture']

# Define a threshold for grouping similar moisture levels (e.g., within 1% of each other)
moisture_probe_data['Moisture_Level_Group'] = moisture_probe_data['Average_Moisture_%'].round()

# Count how many entries fall into each moisture level group
moisture_level_distribution = moisture_probe_data['Moisture_Level_Group'].value_counts().sort_index()

moisture_level_distribution


# In[3]:


import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to preprocess a single CSV file
def preprocess_csv(filename):
    data = pd.read_csv(filename)
    amplitude_min = data['Amplitude(mdb)'].min()
    amplitude_max = data['Amplitude(mdb)'].max()
    data['Amplitude(mdb)_normalized'] = (data['Amplitude(mdb)'] - amplitude_min) / (amplitude_max - amplitude_min)
    return data

# Function to generate and plot spectrogram for each sweep
def plot_spectrogram(data, filename, height, save_directory):
    unique_sweeps = data['Sweep'].unique()
    sweep_counter = 0  # Initialize counter for sweep number
    for sweep in unique_sweeps:
        sweep_data = data[data['Sweep'] == sweep]
        frequency = filename.split('_')[3]  # Extract transmitted frequency from filename
        plt.figure(figsize=(10, 6))
        plt.specgram(sweep_data['Amplitude(mdb)_normalized'], NFFT=2048, Fs=22050, noverlap=1024, cmap='viridis')
        plt.title(f"Spectrogram - Transmitted Frequency: {frequency} Hz, Height: {height}")
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Amplitude (Normalized)')
        # Construct filename with sweep number
        save_filename = f"{filename[:-4]}_sweep_{sweep_counter}_spectrogram.png"
        save_path = os.path.join(save_directory, save_filename)
        plt.savefig(save_path)
        plt.close()
        sweep_counter += 1  # Increment sweep counter

# Function to load datasets from CSV files in a directory
def load_datasets(directory, save_directory):
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            height = filename.split('_')[4].split('.')[0]  # Extract height from filename
            file_path = os.path.join(directory, filename)
            data = preprocess_csv(file_path)
            plot_spectrogram(data, filename, height, save_directory)

# Specify directories
csv_directory = 'C:/SA44B_2024March22'
spectrogram_directory = 'C:/spectrograms'

# Load and preprocess datasets, saving spectrograms
load_datasets(csv_directory, spectrogram_directory)


# In[4]:


from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os

# Directory where the spectrograms are stored
spectrogram_dir = "C:/Spectrograms"
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Function to load and normalize spectrogram images
def load_and_normalize_spectrograms(spectrogram_dir, img_height, img_width):
    X = []  # List to hold the normalized spectrogram data
    for filename in os.listdir(spectrogram_dir):
        if filename.endswith('.png'):  # Assuming spectrograms are in PNG format
            img_path = os.path.join(spectrogram_dir, filename)
            img = imread(img_path, as_gray=True)  # Load image in grayscale
            img_resized = resize(img, (img_height, img_width))  # Resize the image
            img_normalized = img_resized / 255.0  # Normalize pixel values to [0, 1]
            X.append(img_normalized)
    return np.array(X)

# Load and preprocess the spectrogram data
X = load_and_normalize_spectrograms(spectrogram_dir, IMG_HEIGHT, IMG_WIDTH)


# In[5]:


from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Normalize the moisture levels
scaler = MinMaxScaler(feature_range=(0, 1))
moisture_probe_data['Normalized_Moisture'] = scaler.fit_transform(moisture_probe_data[['Average_Moisture_%']])

# Save the scaler for later use to inverse transform model predictions
import joblib
scaler_filename = "moisture_scaler.save"
joblib.dump(scaler, scaler_filename)


# In[7]:


import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import os
import joblib
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import scipy.stats as stats  # Import scipy.stats for QQ plot

# Load and preprocess moisture data
moisture_data_path = 'C:/Moisture_Probe/Moisture_Interpolated.csv'
moisture_data = pd.read_csv(moisture_data_path, index_col='Date & Time', parse_dates=True)

# Select moisture data for a specific date (2024-03-22)
target_moisture_data = moisture_data.loc['2024-03-22'] 

# Extract feature columns
percentage_columns = [col for col in target_moisture_data.index if col.endswith('%')]
mv_columns = [col for col in target_moisture_data.index if 'mV' in col]
feature_columns = percentage_columns + mv_columns

# Reshape moisture features for normalization
moisture_features = target_moisture_data[feature_columns].values.reshape(-1, len(feature_columns))

# Normalize moisture data
scaler = MinMaxScaler(feature_range=(0, 1))
moisture_features = scaler.fit_transform(moisture_features)

# Save the scaler for inverse transformation after prediction
scaler_filename = "moisture_scaler.pkl"
joblib.dump(scaler, scaler_filename)

# Function to load and preprocess spectrogram images
def load_spectrograms(spectrogram_dir, img_height, img_width):
    X, filenames = [], []
    for filename in sorted(os.listdir(spectrogram_dir)):
        if filename.endswith('.png'):
            filepath = os.path.join(spectrogram_dir, filename)
            img = imread(filepath, as_gray=True)
            img_resized = resize(img, (img_height, img_width))
            X.append(img_resized)
            filenames.append(filename)
    return np.array(X), filenames

# Load spectrogram data with fixed size
spectrogram_dir = "C:/Spectrograms"
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Set your desired height and width
X, filenames = load_spectrograms(spectrogram_dir, IMG_HEIGHT, IMG_WIDTH)

# Use the same moisture probe data for all spectrograms
random_moisture_features = np.repeat(moisture_features, len(X), axis=0)

# Define k-fold cross-validation
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Define CNN model
model = Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(len(feature_columns), activation='linear')  # Predicting all moisture features
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Perform k-fold cross-validation
for fold, (train_index, test_index) in enumerate(kfold.split(X)):
    print(f"Fold {fold + 1}:")
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = random_moisture_features[train_index], random_moisture_features[test_index]
    
    # Train the model
    history = model.fit(X_train, Y_train, validation_split=0.2, epochs=10, batch_size=32, callbacks=[early_stopping])
    
    # Evaluate the model
    loss = model.evaluate(X_test, Y_test)
    print("Test Loss:", loss)
    
    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Actual vs. Predicted Values Plot
    for i, feature in enumerate(feature_columns):
        plt.figure(figsize=(8, 6))
        plt.scatter(Y_test[:, i], predictions[:, i], color='blue', alpha=0.5)
        plt.plot(Y_test[:, i], Y_test[:, i], color='red')
        plt.title(f'Actual vs. Predicted Values for {feature}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.show()

        # Residual Plot
        residuals = Y_test[:, i] - predictions[:, i]
        plt.figure(figsize=(8, 6))
        plt.scatter(Y_test[:, i], residuals, color='green', alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Actual Values')
        plt.ylabel('Residuals')
        plt.grid(True)
        plt.show()

        # Learning Curves (if available)
        # Assuming you have arrays of training and validation losses over epochs
        train_losses = np.array(history.history['loss'])  # training losses over epochs
        val_losses = np.array(history.history['val_loss'])    # validation losses over epochs
        epochs = np.arange(1, len(train_losses) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.title('Learning Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        # QQ Plot (if desired)
        plt.figure(figsize=(8, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('QQ Plot of Residuals')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Ordered Residuals')
        plt.grid(True)
        plt.show()


# In[8]:


# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae_scores = []
mse_scores = []
rmse_scores = []
r2_scores = []

for i, column in enumerate(feature_columns):
    y_true = Y_test[:, i]
    y_pred = predictions[:, i]
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    mae_scores.append(mae)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    r2_scores.append(r2)
    
    # Print scores for current moisture feature
    print(f"Metrics for {column}:")
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R-squared (R²):", r2)
    print()

# Calculate and print average scores across all moisture features
print("Average scores across all moisture features:")
print("Mean Absolute Error:", np.mean(mae_scores))
print("Mean Squared Error:", np.mean(mse_scores))
print("Root Mean Squared Error:", np.mean(rmse_scores))
print("Average R-squared (R²):", np.mean(r2_scores))


# In[ ]:




