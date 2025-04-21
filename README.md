# Continuum: Zero-Downtime ML System

A high-performance machine learning system built with Rust and Python that enables zero-downtime model updates and real-time adaptation to data drift.

## Performance Highlights

| Metric | Continuum (Rust) | Sklearn (Python) | Improvement |
|--------|-----------------|------------------|-------------|
| Avg Latency | 0.11 ms | 0.40 ms | 3.73x faster |
| Throughput | 158.86 pred/sec | 152.63 pred/sec | 1.04x higher |
| Memory Usage | 135.75 MB | 140.80 MB | 1.04x less |
| CPU Usage | 34.60% | 35.40% | 1.02x less |
| Model Updates | Zero-downtime | Requires restart | ∞ better |

## Key Features

- **Zero-downtime model updates**: Models can be updated without interrupting the prediction service
- **Real-time adaptation to data drift**: The system automatically detects and adapts to changing data patterns
- **High-performance Rust backend**: Significant speed and efficiency improvements over pure Python implementations
- **PyO3 Rust bindings**: Seamless integration with Python ecosystem while maintaining Rust performance
- **Concurrent prediction and training**: Model training happens in the background without blocking predictions

## Performance Visualizations

### Overall Performance Dashboard
![Performance Dashboard](py/plots/performance_dashboard.png)

### Latency Comparison
![Latency Comparison](py/plots/latency_comparison.png)

### Model Version Timeline (Zero-Downtime Updates)
![Model Versions](py/plots/model_versions.png)

## Getting Started

### Prerequisites

- Rust (1.67.0+)
- Python (3.8+)
- A C compiler (for building native extensions)

### Installation

#### Option 1: Install from PyPI (coming soon)
```bash
pip install continuum-ml
```

#### Option 2: Build from source

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/continuum.git
   cd continuum
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Install maturin (PyO3 build tool)**
   ```bash
   pip install maturin
   ```

5. **Build and install the package in development mode**
   ```bash
   # For default target (your current architecture)
   cd py
   maturin develop
   
   # For specific target
   maturin develop --target x86_64-apple-darwin  # Example for Intel Macs
   # OR
   maturin develop --target aarch64-apple-darwin  # Example for Apple Silicon (M1/M2)
   ```

6. **Verify installation**
   ```bash
   python -c "import continuum; print(continuum.__version__)"
   ```

### Wheel Building for Distribution

To build wheels for distribution:

```bash
cd py
maturin build --release
```

This creates wheels in the `target/wheels` directory.

## Usage Example

```python
import continuum
import numpy as np
import time

# Create a Continuum instance with frequent updates
config = continuum.LearningConfig.frequent_updates()
config.min_samples = 30  # Train after 30 samples
config.interval_sec = 3  # Check every 3 seconds

# Initialize the system
ml_system = continuum.Continuum(config)

# Register a linear regression model
ml_system.register_model(
    "adaptive_model",
    "linear",
    continuum.ModelParameters(
        with_bias=True, 
        learning_rate=0.01, 
        max_iterations=1000, 
        regularization=None
    )
)

# Add initial training data
for i in range(50):
    x = np.random.uniform(0, 10)
    y = 1.0 * x + np.random.normal(0, 0.1)  # y = x + noise
    ml_system.add_training_example("adaptive_model", [float(x)], float(y), False)

# Start continuous learning in the background
ml_system.start_continuous_learning()

# Wait for model to train
time.sleep(3)

# Make predictions
for i in range(10):
    x = np.random.uniform(0, 10)
    response = ml_system.predict("adaptive_model", [float(x)])
    print(f"Input: {x}, Prediction: {response.prediction}, Version: {response.model_version}")

# Add data with new pattern (data drift)
for i in range(50):
    x = np.random.uniform(0, 10)
    y = 2.0 * x + np.random.normal(0, 0.1)  # y = 2x + noise (slope changed)
    ml_system.add_training_example("adaptive_model", [float(x)], float(y), False)

# Wait for model update
time.sleep(5)

# Make more predictions with the updated model
for i in range(10):
    x = np.random.uniform(0, 10)
    response = ml_system.predict("adaptive_model", [float(x)])
    print(f"Input: {x}, Prediction: {response.prediction}, Version: {response.model_version}")

# Stop continuous learning
ml_system.stop_continuous_learning()
```

## Benchmark and Demo Script

To run the full performance comparison against scikit-learn:

```bash
python examples/enhanced_example.py
```

This will generate detailed performance metrics and visualizations comparing Continuum with scikit-learn.

## How It Works

Continuum uses a Rust backend with PyO3 bindings to provide high-performance machine learning capabilities to Python users. The system's architecture includes:

1. **Rust Core**: High-performance ML algorithms implemented in Rust
2. **Background Training**: A separate thread that monitors and trains models
3. **Atomic Model Swapping**: Zero-downtime model updates using atomic reference swapping
4. **PyO3 Bindings**: Seamless Python integration with minimal overhead
5. **Drift Detection**: Statistical methods to detect when data distributions change

This architecture provides substantial benefits:
- No prediction service interruptions during model updates
- Lower latency than pure Python implementations
- Reduced memory footprint
- Automatic adaptation to changing data distributions

## Advanced Configuration

```python
# Create custom learning configuration
config = continuum.LearningConfig(
    # Minimum samples before training
    min_samples=100,
    
    # Minimum time between training runs (seconds)
    interval_sec=10,
    
    # Maximum training queue size
    max_queue_size=10000,
    
    # Enable/disable drift detection
    drift_detection=True,
    
    # Drift detection sensitivity (0.0-1.0)
    drift_threshold=0.1
)

ml_system = continuum.Continuum(config)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The PyO3 team for making Rust-Python interoperability possible
- The Rust community for creating a fantastic language for systems programming
- All contributors who have helped improve this project