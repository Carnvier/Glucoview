# Diabetic Retinopathy Detection - TensorFlow Version
## IET Codefest 2025 - TensorFlow/Keras Implementation

This is the **TensorFlow/Keras version** of the diabetic retinopathy detection pipeline, optimized for production deployment with multiple export formats.

## 🚀 Why TensorFlow?

### Advantages over PyTorch:
- **🌐 Better Web Deployment**: Native TensorFlow.js support
- **📱 Mobile Optimized**: TensorFlow Lite for mobile/edge devices
- **🏭 Production Ready**: Mature serving infrastructure (TF Serving)
- **🔧 Multiple Export Formats**: SavedModel, H5, TFLite, TFJS
- **⚡ Performance**: Optimized for inference and serving
- **📊 Easy Integration**: Better ecosystem for deployment

## 📊 Notebook Structure

- **Total Cells**: 16 (All code cells)
- **Lines of Code**: 1,146
- **Framework**: TensorFlow 2.13+ / Keras
- **Format**: Pure code with comprehensive comments

## 🏗️ Complete Pipeline Features

### 1. **TensorFlow-Native Implementation**
```python
# TensorFlow/Keras specific features:
- tf.data pipeline for efficient data loading
- Mixed precision training support
- Built-in callbacks (EarlyStopping, ReduceLROnPlateau)
- TensorBoard integration
- Multi-format model export
```

### 2. **Advanced Data Pipeline**
- **tf.data.Dataset**: Efficient data loading and preprocessing
- **Mixed Precision**: Faster training with automatic loss scaling
- **Prefetching & Caching**: Optimized data pipeline
- **Built-in Augmentation**: TensorFlow image operations

### 3. **Production-Ready Exports**
```
📦 Multiple Export Formats:
├── SavedModel/     # Production serving (TF Serving)
├── model.h5        # Keras native format
├── model.tflite    # Mobile/Edge deployment
└── tfjs_model/     # Web deployment
```

## 📋 Cell Structure

```
Cell 1:  Project Introduction & TensorFlow Setup
Cell 2:  TensorFlow Package Installation
Cell 3:  TensorFlow Imports and GPU Configuration
Cell 4:  TensorFlow Pipeline Configuration
Cell 5:  Label Cleaning and Data Loading
Cell 6:  TensorFlow Data Pipeline & Preprocessing
Cell 7:  TensorFlow/Keras Model Architecture
Cell 8:  Training Setup and Callbacks
Cell 9:  Phase 1 Training (Frozen Backbone)
Cell 10: Phase 2 Training (Fine-tuning)
Cell 11: Training History Visualization
Cell 12: Model Evaluation on Test Set
Cell 13: Confusion Matrix & Performance Visualization
Cell 14: Multi-Format Model Export
Cell 15: TensorFlow Configuration Export
Cell 16: Final Pipeline Summary
```

## ⚙️ Configuration

```python
CONFIG = {
    'DATA_PATH': '/kaggle/input',          # Update for your dataset
    'LABELS_FILE': 'labels.csv',           # Your labels file
    'IMAGE_SIZE': 224,                     # Input size
    'BATCH_SIZE': 32,                      # Batch size
    'NUM_EPOCHS': 50,                      # Max epochs
    'MIXED_PRECISION': True,               # Enable mixed precision
    'PREFETCH_BUFFER': tf.data.AUTOTUNE,   # Auto-tune prefetch
}
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install tensorflow>=2.13.0
pip install opencv-python-headless
pip install matplotlib seaborn pandas numpy scikit-learn
pip install tensorflowjs  # For web export
```

### 2. Update Configuration
```python
# Cell 4: Update these paths
CONFIG = {
    'DATA_PATH': '/path/to/your/dataset',    # UPDATE
    'LABELS_FILE': 'labels.csv',             # UPDATE
    # ... other settings
}
```

### 3. Run All Cells
Execute sequentially from top to bottom

## 📁 Generated Outputs

```
outputs/
├── diabetic_retinopathy_savedmodel/         # 🎯 Production model
├── diabetic_retinopathy_model.h5            # 📦 Keras model
├── diabetic_retinopathy_model.tflite        # 📱 Mobile model
├── diabetic_retinopathy_tfjs/               # 🌐 Web model
├── tensorflow_config.json                   # ⚙️ Configuration
├── tensorflow_training_history.png          # 📈 Training curves
├── tensorflow_confusion_matrix.png          # 🔢 Evaluation
└── tensorflow_roc_curves.png               # 📊 ROC analysis
```

## 🎯 Key TensorFlow Features

### 1. **Efficient Data Pipeline**
```python
# tf.data pipeline with automatic optimization
dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
```

### 2. **Mixed Precision Training**
```python
# Automatic mixed precision for faster training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

### 3. **Comprehensive Callbacks**
```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(...),
    tf.keras.callbacks.ReduceLROnPlateau(...),
    tf.keras.callbacks.ModelCheckpoint(...),
    tf.keras.callbacks.TensorBoard(...)
]
```

### 4. **Multi-Format Export**
```python
# Multiple deployment formats
model.save('savedmodel/', save_format='tf')      # Production
model.save('model.h5', save_format='h5')         # Keras
converter.convert()                               # TFLite
tensorflowjs_converter(...)                      # Web
```

## 🌐 Deployment Options

### 1. **Web Deployment (TensorFlow.js)**
```javascript
// Load and use in browser
const model = await tf.loadGraphModel('/path/to/tfjs_model/model.json');
const prediction = model.predict(preprocessedImage);
```

### 2. **Mobile Deployment (TensorFlow Lite)**
```python
# Mobile inference
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
```

### 3. **Server Deployment (SavedModel)**
```python
# Production serving
model = tf.saved_model.load('savedmodel/')
predictions = model(input_tensor)
```

## 🏆 Advantages for IET Codefest 2025

### **Production Ready**
- Multiple deployment formats
- Optimized for serving
- Industry-standard framework

### **Performance Optimized**
- Mixed precision training
- Efficient data pipeline
- GPU acceleration

### **Deployment Flexibility**
- Web (TensorFlow.js)
- Mobile (TensorFlow Lite)
- Server (SavedModel)
- Edge devices

### **Easy Integration**
- REST API ready
- Cloud platform support
- Microservices friendly

## 📊 Performance Features

- **🚀 Fast Training**: Mixed precision + optimized data pipeline
- **📱 Mobile Ready**: TensorFlow Lite quantization
- **🌐 Web Optimized**: TensorFlow.js for browser deployment
- **🏭 Production Scale**: SavedModel for TensorFlow Serving
- **📊 Monitoring**: TensorBoard integration

## 🔧 Technical Highlights

### **Modern TensorFlow Features**
- tf.data for data pipeline
- tf.keras for high-level API
- tf.saved_model for deployment
- tf.lite for mobile optimization

### **Best Practices**
- Mixed precision training
- Proper callback usage
- Efficient preprocessing
- Multiple export formats

---

**🎉 Perfect for IET Codefest 2025!**

This TensorFlow implementation provides:
- ✅ **Production-ready** deployment options
- ✅ **Multi-platform** support (web, mobile, server)
- ✅ **Optimized performance** with mixed precision
- ✅ **Industry-standard** framework
- ✅ **Complete pipeline** with comprehensive evaluation
- ✅ **Easy deployment** with multiple export formats

**Ready for real-world deployment and competition submission!**