# Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('ai4i2020.csv')

# Display the first few rows
print(data.head())

# Data Overview
# The dataset contains the following features:
#  - Air temperature, Process temperature, Rotational speed, Torque, Tool wear, etc.

# Check for missing values (if any)
print(data.isnull().sum())

# Drop the 'UDI' and 'Product ID' as they are not useful for prediction
data = data.drop(columns=['UDI', 'Product ID'])

# Separate features and the target
X = data.drop(columns=['Machine failure'])  # Features
y = data['Machine failure']                 # Target

# One-hot encode the categorical 'Type' column
# Column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Air temperature [K]', 'Process temperature [K]',
                                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']),
        ('cat', OneHotEncoder(), ['Type'])
    ]
)

# Transform the features
X_preprocessed = preprocessor.fit_transform(X)

# Split the dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Build the neural network model
model = models.Sequential([
    layers.InputLayer(input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification for failure prediction
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Accuracy and Confusion Matrix
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc * 100:.2f}%')

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Visualize the training process
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualize the loss process
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
