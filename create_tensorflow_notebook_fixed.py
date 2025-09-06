#!/usr/bin/env python3
"""
Script to create TensorFlow/Keras version of diabetic retinopathy detection notebook
"""

import json

def create_cell(cell_type, source, metadata=None):
    """Create a notebook cell"""
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source if isinstance(source, list) else [source]
    }
    
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    
    return cell

# Define all notebook cells using TensorFlow
cells = []

# Title and setup
cells.append(create_cell("code", [
    "# ============================================================================\n",
    "# DIABETIC RETINOPATHY DETECTION PIPELINE - TENSORFLOW VERSION\n",
    "# ============================================================================\n",
    "# Complete ML Pipeline using TensorFlow/Keras and ResNet50\n",
    "# IET Codefest 2025 - Optimized for TensorFlow deployment\n",
    "#\n",
    "# Pipeline Overview:\n",
    "# 1. Dataset Understanding & Label Cleaning\n",
    "# 2. Exploratory Data Analysis (EDA)\n",
    "# 3. Preprocessing & Augmentation (tf.data)\n",
    "# 4. Model Training (Two-phase ResNet50)\n",
    "# 5. Explainability (GradCAM)\n",
    "# 6. Model Export (SavedModel & TensorFlow.js)\n",
    "# 7. Comprehensive Evaluation\n",
    "# ============================================================================\n",
    "\n",
    "print(\"🚀 Starting TensorFlow Diabetic Retinopathy Detection Pipeline\")\n",
    "print(\"📋 IET Codefest 2025 - TensorFlow/Keras Implementation\")\n",
    "print(\"⚡ Optimized for production deployment\")"
]))

# Install packages
cells.append(create_cell("code", [
    "# ============================================================================\n",
    "# TENSORFLOW PACKAGE INSTALLATION\n",
    "# ============================================================================\n",
    "# Install TensorFlow and related packages for the complete pipeline\n",
    "\n",
    "!pip install tensorflow>=2.13.0\n",
    "!pip install opencv-python-headless\n",
    "!pip install matplotlib seaborn plotly pandas numpy\n",
    "!pip install scikit-learn\n",
    "!pip install Pillow\n",
    "\n",
    "# Optional: TensorFlow.js conversion tools\n",
    "!pip install tensorflowjs\n",
    "\n",
    "print(\"✅ All TensorFlow packages installed successfully!\")"
]))

# Imports
cells.append(create_cell("code", [
    "# ============================================================================\n",
    "# TENSORFLOW IMPORTS AND SETUP\n",
    "# ============================================================================\n",
    "# Import all necessary libraries for TensorFlow-based pipeline\n",
    "\n",
    "import os\n",
    "import json\n",
    "import warnings\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from pathlib import Path\n",
    "from collections import Counter\n",
    "import re\n",
    "\n",
    "# TensorFlow imports\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers, models, optimizers, callbacks\n",
    "from tensorflow.keras.applications import ResNet50\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "\n",
    "# Image processing\n",
    "import cv2\n",
    "from PIL import Image\n",
    "\n",
    "# ML utilities\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import (\n",
    "    classification_report, confusion_matrix, roc_curve, auc,\n",
    "    precision_recall_curve, f1_score, accuracy_score\n",
    ")\n",
    "from sklearn.utils.class_weight import compute_class_weight\n",
    "\n",
    "# Configure environment\n",
    "warnings.filterwarnings('ignore')\n",
    "plt.style.use('default')\n",
    "sns.set_palette(\"husl\")\n",
    "\n",
    "# Set random seeds for reproducibility\n",
    "tf.random.set_seed(42)\n",
    "np.random.seed(42)\n",
    "\n",
    "# Configure TensorFlow\n",
    "print(f\"🔧 TensorFlow version: {tf.__version__}\")\n",
    "print(f\"🔧 Keras version: {keras.__version__}\")\n",
    "\n",
    "# Check for GPU\n",
    "gpus = tf.config.experimental.list_physical_devices('GPU')\n",
    "if gpus:\n",
    "    try:\n",
    "        # Enable memory growth to avoid allocating all GPU memory\n",
    "        for gpu in gpus:\n",
    "            tf.config.experimental.set_memory_growth(gpu, True)\n",
    "        print(f\"🎮 GPU available: {len(gpus)} device(s)\")\n",
    "        print(f\"🎮 GPU details: {gpus[0]}\")\n",
    "    except RuntimeError as e:\n",
    "        print(f\"GPU setup error: {e}\")\nelse:\n",
    "    print(\"💻 Using CPU for training\")\n",
    "\n",
    "print(\"✅ All TensorFlow imports loaded successfully!\")"
]))

# Configuration
cells.append(create_cell("code", [
    "# ============================================================================\n",
    "# TENSORFLOW PIPELINE CONFIGURATION\n",
    "# ============================================================================\n",
    "# Main configuration optimized for TensorFlow/Keras training\n",
    "\n",
    "CONFIG = {\n",
    "    # Data paths - UPDATE THESE FOR YOUR DATASET\n",
    "    'DATA_PATH': '/kaggle/input',          # Update this path to your dataset location\n",
    "    'LABELS_FILE': 'labels.csv',           # Update this to your labels file name\n",
    "    \n",
    "    # Model parameters\n",
    "    'IMAGE_SIZE': 224,                     # Input image size (224x224)\n",
    "    'BATCH_SIZE': 32,                      # Training batch size\n",
    "    'NUM_EPOCHS': 50,                      # Maximum training epochs\n",
    "    'LEARNING_RATE': 1e-4,                 # Initial learning rate\n",
    "    'NUM_CLASSES': 5,                      # Number of classification classes\n",
    "    'MODEL_NAME': 'resnet50',              # Model architecture\n",
    "    \n",
    "    # Training parameters\n",
    "    'PATIENCE': 10,                        # Early stopping patience\n",
    "    'MIN_DELTA': 0.001,                    # Minimum improvement for early stopping\n",
    "    'VALIDATION_SPLIT': 0.15,              # Validation split ratio\n",
    "    'TEST_SPLIT': 0.15,                    # Test split ratio\n",
    "    \n",
    "    # TensorFlow specific\n",
    "    'MIXED_PRECISION': True,               # Use mixed precision training\n",
    "    'PREFETCH_BUFFER': tf.data.AUTOTUNE,   # Dataset prefetch buffer\n",
    "    'CACHE_DATASET': True,                 # Cache dataset in memory\n",
    "    \n",
    "    # Paths\n",
    "    'SAVE_PATH': './models/',              # Model save directory\n",
    "    'TENSORBOARD_PATH': './logs/',         # TensorBoard logs\n",
    "    'RANDOM_SEED': 42                      # Random seed for reproducibility\n",
    "}\n",
    "\n",
    "# Enable mixed precision if supported\n",
    "if CONFIG['MIXED_PRECISION']:\n",
    "    try:\n",
    "        policy = tf.keras.mixed_precision.Policy('mixed_float16')\n",
    "        tf.keras.mixed_precision.set_global_policy(policy)\n",
    "        print(\"✅ Mixed precision training enabled\")\n",
    "    except:\n",
    "        print(\"⚠️  Mixed precision not supported, using float32\")\n",
    "        CONFIG['MIXED_PRECISION'] = False\n",
    "\n",
    "# Create necessary directories\n",
    "os.makedirs(CONFIG['SAVE_PATH'], exist_ok=True)\n",
    "os.makedirs('./outputs', exist_ok=True)\n",
    "os.makedirs(CONFIG['TENSORBOARD_PATH'], exist_ok=True)\n",
    "\n",
    "# Display configuration\n",
    "print(\"⚙️  TensorFlow Pipeline Configuration:\")\n",
    "print(\"=\" * 50)\n",
    "for key, value in CONFIG.items():\n",
    "    print(f\"{key:<20}: {value}\")\n",
    "print(\"=\" * 50)\n",
    "print(\"✅ TensorFlow configuration loaded successfully!\")"
]))

# Save the first part
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('/workspace/diabetic_retinopathy_tensorflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("TensorFlow notebook (Part 1) created successfully!")