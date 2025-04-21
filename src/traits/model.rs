use std::error::Error;
use std::fmt::{write, Display, Formatter, Result as FmtResult};
use crate::traits::features::FeatureVector;

/// Custom error type for machine learning models
#[derive(Debug)]
pub enum ModelError {
    /// Errors during training
    TrainingError(String),
    /// Errors during prediction
    PredictionError(String),
    /// Data dimension mismatch errors
    DimensionMismatch {
        expected: usize,
        actual: usize,
        context: String,
    },
    /// Invalid parameter errors
    InvalidParameter(String),
    /// I/O errors (for model saving/loading)
    IoError(std::io::Error),
    /// Errors from serialization/deserialization
    SerializationError(String),

    ValidationError(String),
    /// Timeout errors
    Timeout(String),
}

impl Display for ModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            ModelError::TrainingError(msg) => write!(f, "Training error: {}", msg),
            ModelError::PredictionError(msg) => write!(f, "Prediction error: {}", msg),
            ModelError::DimensionMismatch { expected, actual, context } => 
                write!(f, "Dimension mismatch ({}): expected {}, got {}", context, expected, actual),
            ModelError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            ModelError::IoError(err) => write!(f, "I/O error: {}", err),
            ModelError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            ModelError::Timeout(msg) => write!(f, "Operation timed out: {}", msg),
            ModelError::ValidationError(msg) => write!(f, "Validation error: {}", msg)
        }
    }
}

impl Error for ModelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ModelError::IoError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ModelError {
    fn from(err: std::io::Error) -> Self {
        ModelError::IoError(err)
    }
}

/// Trait for model validation metrics
pub trait Metrics {
    /// Calculate mean squared error
    fn mse(&self, predictions: &[f32], targets: &[f32]) -> Result<f32, ModelError>;
    
    /// Calculate root mean squared error
    fn rmse(&self, predictions: &[f32], targets: &[f32]) -> Result<f32, ModelError>;
    
    /// Calculate mean absolute error
    fn mae(&self, predictions: &[f32], targets: &[f32]) -> Result<f32, ModelError>;
    
    /// Calculate R-squared (coefficient of determination)
    fn r_squared(&self, predictions: &[f32], targets: &[f32]) -> Result<f32, ModelError>;
}

/// Core trait for machine learning models
pub trait Model: Send + Sync {
    /// Train the model on a batch of data
    fn train(&mut self, features: &[FeatureVector], targets: &[f32]) -> Result<(), ModelError>;
    
    /// Make a prediction for a single feature vector
    fn predict(&self, feature: &FeatureVector) -> Result<f32, ModelError>;
    
    /// Make predictions for multiple feature vectors
    fn predict_batch(&self, features: &[FeatureVector]) -> Result<Vec<f32>, ModelError> {
        let mut predictions = Vec::with_capacity(features.len());
        for feature in features {
            predictions.push(self.predict(feature)?);
        }
        Ok(predictions)
    }
    
    /// Export model parameters
    fn export_parameters(&self) -> Result<Vec<f32>, ModelError>;
    
    /// Import model parameters
    fn import_parameters(&mut self, parameters: Vec<f32>) -> Result<(), ModelError>;
    
    /// Validate the model using test data
    fn validate(&self, features: &[FeatureVector], targets: &[f32]) -> Result<f32, ModelError>;
    
    /// Save the model to a file
    fn save(&self, path: &str) -> Result<(), ModelError>;
    
    /// Load the model from a file
    fn load(&mut self, path: &str) -> Result<(), ModelError>;
    
    /// Clone the model (needed for atomic swapping)
    fn clone_model(&self) -> Box<dyn Model>;
}

/// Trait for models that support asynchronous operations
#[async_trait::async_trait]
pub trait AsyncModel: Model {
    /// Train the model asynchronously
    async fn train_async(&mut self, features: &[FeatureVector], targets: &[f32]) -> Result<(), ModelError>;
    
    /// Make predictions asynchronously
    async fn predict_async(&self, feature: &FeatureVector) -> Result<f32, ModelError>;
    
    /// Make batch predictions asynchronously
    async fn predict_batch_async(&self, features: &[FeatureVector]) -> Result<Vec<f32>, ModelError>;
    
    /// Validate the model asynchronously
    async fn validate_async(&self, features: &[FeatureVector], targets: &[f32]) -> Result<f32, ModelError>;
}

/// Trait for models that can be updated incrementally (online learning)
pub trait IncrementalModel: Model {
    /// Update the model with new training examples without full retraining
    fn update(&mut self, features: &[FeatureVector], targets: &[f32]) -> Result<(), ModelError>;
    
    /// Set how much to weight new examples compared to existing model knowledge (0.0-1.0)
    fn set_learning_rate(&mut self, rate: f32) -> Result<(), ModelError>;
    
    /// Get current model parameters for incremental update
    fn get_parameters(&self) -> Vec<f32>;
}

/// Factory trait for creating new model instances
pub trait ModelFactory: Send + Sync {
    /// Create a new instance of the model with default parameters
    fn create(&self) -> Box<dyn Model>;
    
    /// Create a new instance with custom hyperparameters
    fn create_with_params(&self, params: &[f32]) -> Result<Box<dyn Model>, ModelError>;
}