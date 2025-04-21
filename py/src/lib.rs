use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use tokio::runtime::Runtime;

use continuum::{
    ContinuumApi,
    ContinuousLearningConfig,
    ModelParameters,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfo,
    ApiError,
};

// Create a thread-local Tokio runtime for async operations
thread_local! {
    static RUNTIME: Runtime = Runtime::new().unwrap();
}

// Helper function to run async operations synchronously
fn run_sync<F, T>(future: F) -> PyResult<T>
where
    F: std::future::Future<Output = Result<T, ApiError>>,
{
    RUNTIME.with(|rt| {
        rt.block_on(future).map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

/// Python wrapper for ModelParameters
#[pyclass(subclass)]
#[derive(Clone)]
struct PyModelParameters {
    #[pyo3(get, set)]
    with_bias: bool,
    #[pyo3(get, set)]
    learning_rate: Option<f32>,
    #[pyo3(get, set)]
    max_iterations: Option<usize>,
    #[pyo3(get, set)]
    regularization: Option<f32>,
}

#[pymethods]
impl PyModelParameters {
    #[new]
    fn new(
        with_bias: bool,
        learning_rate: Option<f32>,
        max_iterations: Option<usize>,
        regularization: Option<f32>,
    ) -> Self {
        Self {
            with_bias,
            learning_rate,
            max_iterations,
            regularization,
        }
    }
}

impl From<PyModelParameters> for ModelParameters {
    fn from(params: PyModelParameters) -> Self {
        ModelParameters {
            with_bias: params.with_bias,
            learning_rate: params.learning_rate,
            max_iterations: params.max_iterations,
            regularization: params.regularization,
        }
    }
}

/// Python wrapper for PredictionResponse
#[pyclass(subclass)]
#[derive(Clone)]
struct PyPredictionResponse {
    #[pyo3(get)]
    prediction: f32,
    #[pyo3(get)]
    model_version: usize,
}

#[pymethods]
impl PyPredictionResponse {
    fn __repr__(&self) -> String {
        format!("PredictionResponse(prediction={}, model_version={})", 
                self.prediction, self.model_version)
    }
}

impl From<PredictionResponse> for PyPredictionResponse {
    fn from(resp: PredictionResponse) -> Self {
        Self {
            prediction: resp.prediction,
            model_version: resp.model_version,
        }
    }
}

/// Python wrapper for BatchPredictionResponse
#[pyclass(subclass)]
#[derive(Clone)]
struct PyBatchPredictionResponse {
    #[pyo3(get)]
    predictions: Vec<f32>,
    #[pyo3(get)]
    model_version: usize,
}

#[pymethods]
impl PyBatchPredictionResponse {
    fn __repr__(&self) -> String {
        format!("BatchPredictionResponse(predictions={:?}, model_version={})", 
                self.predictions, self.model_version)
    }
}

impl From<BatchPredictionResponse> for PyBatchPredictionResponse {
    fn from(resp: BatchPredictionResponse) -> Self {
        Self {
            predictions: resp.predictions,
            model_version: resp.model_version,
        }
    }
}

/// Python wrapper for ModelInfo
#[pyclass(subclass)]
#[derive(Clone)]
struct PyModelInfo {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    version: usize,
    #[pyo3(get)]
    is_training: bool,
    #[pyo3(get)]
    stats: String,
}

#[pymethods]
impl PyModelInfo {
    fn __repr__(&self) -> String {
        format!("ModelInfo(name='{}', version={}, is_training={}, stats='{}')", 
                self.name, self.version, self.is_training, self.stats)
    }
}

impl From<ModelInfo> for PyModelInfo {
    fn from(info: ModelInfo) -> Self {
        Self {
            name: info.name,
            version: info.version,
            is_training: info.is_training,
            stats: info.stats,
        }
    }
}

/// Python wrapper for ContinuousLearningConfig
#[pyclass(subclass)]
#[derive(Clone)]
struct PyLearningConfig {
    #[pyo3(get, set)]
    enabled: bool,
    #[pyo3(get, set)]
    interval_sec: u64,
    #[pyo3(get, set)]
    min_samples: usize,
    #[pyo3(get, set)]
    auto_swap: bool,
    #[pyo3(get, set)]
    validation_threshold: f32,
    #[pyo3(get, set)]
    use_kl_divergence: bool,
}

#[pymethods]
impl PyLearningConfig {
    #[new]
    fn new(
        enabled: bool,
        interval_sec: u64,
        min_samples: usize,
        auto_swap: bool,
        validation_threshold: f32,
        use_kl_divergence: bool,
    ) -> Self {
        Self {
            enabled,
            interval_sec,
            min_samples,
            auto_swap,
            validation_threshold,
            use_kl_divergence,
        }
    }
    
    #[staticmethod]
    fn default() -> Self {
        let config = ContinuousLearningConfig::default();
        Self {
            enabled: config.enabled,
            interval_sec: config.interval_sec,
            min_samples: config.min_samples,
            auto_swap: config.auto_swap,
            validation_threshold: config.validation_threshold,
            use_kl_divergence: config.use_kl_divergence,
        }
    }
    
    #[staticmethod]
    fn frequent_updates() -> Self {
        let config = ContinuousLearningConfig::frequent_updates();
        Self {
            enabled: config.enabled,
            interval_sec: config.interval_sec,
            min_samples: config.min_samples,
            auto_swap: config.auto_swap,
            validation_threshold: config.validation_threshold,
            use_kl_divergence: config.use_kl_divergence,
        }
    }
}

impl From<PyLearningConfig> for ContinuousLearningConfig {
    fn from(config: PyLearningConfig) -> Self {
        ContinuousLearningConfig::new(
            config.enabled,
            config.interval_sec,
            config.min_samples,
            config.auto_swap,
            config.validation_threshold,
            config.use_kl_divergence,
        )
    }
}

/// Main Python wrapper for the Continuum API
#[pyclass(subclass)]
struct PyContinuum {
    api: ContinuumApi,
}

#[pymethods]
impl PyContinuum {
    #[new]
    fn new(config: Option<PyLearningConfig>) -> Self {
        let rust_config = config
            .map(|c| c.into())
            .unwrap_or_else(ContinuousLearningConfig::default);
        
        Self {
            api: ContinuumApi::new(rust_config),
        }
    }
    
    #[staticmethod]
    fn default() -> Self {
        Self {
            api: ContinuumApi::default(),
        }
    }
    
    /// Register a new model
    fn register_model(
        &self,
        name: &str,
        model_type: &str,
        parameters: Option<PyModelParameters>,
    ) -> PyResult<()> {
        let rust_params = parameters.map(|p| p.into());
        run_sync(self.api.register_model(name, model_type, rust_params))
    }
    
    /// Make a prediction
    fn predict(&self, model_name: &str, features: Vec<f32>) -> PyResult<PyPredictionResponse> {
        let result = run_sync(self.api.predict(model_name, features))?;
        Ok(result.into())
    }
    
    /// Make batch predictions
    fn predict_batch(
        &self,
        model_name: &str,
        features: Vec<Vec<f32>>,
    ) -> PyResult<PyBatchPredictionResponse> {
        let result = run_sync(self.api.predict_batch(model_name, features))?;
        Ok(result.into())
    }
    
    /// Add a training example
    fn add_training_example(
        &self,
        model_name: &str,
        features: Vec<f32>,
        target: f32,
        is_validation: Option<bool>,
    ) -> PyResult<()> {
        run_sync(self.api.add_training_example(
            model_name,
            features,
            target,
            is_validation.unwrap_or(false),
        ))
    }
    
    /// Manually trigger training for a model
    fn train_model(&self, model_name: &str) -> PyResult<()> {
        run_sync(self.api.train_model(model_name))
    }
    
    /// Get model information
    fn get_model_info(&self, model_name: &str) -> PyResult<PyModelInfo> {
        let result = run_sync(self.api.get_model_info(model_name))?;
        Ok(result.into())
    }
    
    /// List all available models
    fn list_models(&self) -> PyResult<Vec<String>> {
        run_sync(self.api.list_models())
    }
    
    /// Start continuous learning
    fn start_continuous_learning(&self) -> PyResult<()> {
        run_sync(self.api.start_continuous_learning())
    }
    
    /// Stop continuous learning
    fn stop_continuous_learning(&self) -> PyResult<()> {
        self.api.stop_continuous_learning()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// The Python module definition
#[pymodule]
fn continuum_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyContinuum>()?;
    m.add_class::<PyModelParameters>()?;
    m.add_class::<PyLearningConfig>()?;
    m.add_class::<PyPredictionResponse>()?;
    m.add_class::<PyBatchPredictionResponse>()?;
    m.add_class::<PyModelInfo>()?;
    Ok(())
}