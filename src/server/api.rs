use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::traits::features::FeatureVector;
use crate::traits::model::ModelError;
use crate::server::server::ModelServer;
use crate::server::continuous_learning::ContinuousLearningConfig;

/// API errors
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Model error: {0}")]
    ModelError(#[from] ModelError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Not found: {0}")]
    NotFound(String),
}

/// Result type for API operations
pub type ApiResult<T> = Result<T, ApiError>;

/// Model parameters for initialization
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ModelParameters {
    pub with_bias: bool,
    pub learning_rate: Option<f32>,
    pub max_iterations: Option<usize>,
    pub regularization: Option<f32>,
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            with_bias: true,
            learning_rate: Some(0.01),
            max_iterations: Some(1000),
            regularization: None,
        }
    }
}

/// Prediction response
#[derive(Debug, Serialize)]
pub struct PredictionResponse {
    pub prediction: f32,
    pub model_version: usize,
}

/// Batch prediction response
#[derive(Debug, Serialize)]
pub struct BatchPredictionResponse {
    pub predictions: Vec<f32>,
    pub model_version: usize,
}

/// Model information response
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: usize,
    pub is_training: bool,
    pub stats: String,
}

/// API for the ML system
pub struct ContinuumApi {
    server: ModelServer,
}

impl ContinuumApi {
    /// Create a new API instance
    pub fn new(config: ContinuousLearningConfig) -> Self {
        Self {
            server: ModelServer::new(config),
        }
    }
    
    /// Create a new API instance with default configuration
    pub fn default() -> Self {
        Self {
            server: ModelServer::default(),
        }
    }
    
    /// Register a new model
    pub async fn register_model(
        &self,
        name: &str,
        model_type: &str,
        parameters: Option<ModelParameters>,
    ) -> ApiResult<()> {
        let params = parameters.unwrap_or_default();
        
        match model_type {
            "linear" => {
                let model = crate::models::linears::LinearRegression::new(
                    params.with_bias,
                    params.learning_rate.unwrap_or(0.01),
                    params.max_iterations.unwrap_or(1000),
                );
                self.server.register_model(name, model).await?;
                Ok(())
            }
            "ridge" => {
                let model = crate::models::ridge::RidgeRegression::new(
                    params.with_bias,
                    params.regularization.unwrap_or(0.1),
                    params.learning_rate.unwrap_or(0.01),
                    params.max_iterations.unwrap_or(1000),
                );
                self.server.register_model(name, model).await?;
                Ok(())
            }
            _ => Err(ApiError::InvalidInput(format!("Unknown model type: {}", model_type))),
        }
    }
    
    /// Make a prediction
    pub async fn predict(&self, model_name: &str, features: Vec<f32>) -> ApiResult<PredictionResponse> {
        let feature_vector = FeatureVector::new(features);
        
        // Get model for version info
        let model = self.server.get_model(model_name).await?;
        let version = model.get_version();
        
        // Make prediction
        let prediction = self.server.predict(model_name, &feature_vector).await?;
        
        Ok(PredictionResponse {
            prediction,
            model_version: version,
        })
    }
    
    /// Make batch predictions
    pub async fn predict_batch(
        &self,
        model_name: &str,
        features: Vec<Vec<f32>>,
    ) -> ApiResult<BatchPredictionResponse> {
        let feature_vectors: Vec<FeatureVector> = features
            .into_iter()
            .map(FeatureVector::new)
            .collect();
        
        // Get model for version info
        let model = self.server.get_model(model_name).await?;
        let version = model.get_version();
        
        // Make predictions
        let predictions = self.server.predict_batch(model_name, &feature_vectors).await?;
        
        Ok(BatchPredictionResponse {
            predictions,
            model_version: version,
        })
    }
    
    /// Add a training example
    pub async fn add_training_example(
        &self,
        model_name: &str,
        features: Vec<f32>,
        target: f32,
        is_validation: bool,
    ) -> ApiResult<()> {
        let feature_vector = FeatureVector::new(features);
        self.server.add_training_example(
            model_name,
            feature_vector,
            target,
            is_validation,
        ).await?;
        Ok(())
    }
    
    /// Manually trigger training for a model
    pub async fn train_model(&self, model_name: &str) -> ApiResult<()> {
        self.server.train_now(model_name).await?;
        Ok(())
    }
    
    /// Get model information
    pub async fn get_model_info(&self, model_name: &str) -> ApiResult<ModelInfo> {
        let model = self.server.get_model(model_name).await?;
        let stats = self.server.get_model_stats(model_name).await?;
        
        Ok(ModelInfo {
            name: model_name.to_string(),
            version: model.get_version(),
            is_training: model.is_training(),
            stats,
        })
    }
    
    /// List all available models
    pub async fn list_models(&self) -> ApiResult<Vec<String>> {
        Ok(self.server.list_models().await)
    }
    
    /// Start continuous learning
    pub async fn start_continuous_learning(&self) -> ApiResult<()> {
        self.server.start_continuous_learning().await?;
        Ok(())
    }
    
    /// Stop continuous learning
    pub fn stop_continuous_learning(&self) -> ApiResult<()> {
        self.server.stop_continuous_learning();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_api_register_model() {
        let api = ContinuumApi::default();
        
        api.register_model("test_linear", "linear", None).await.unwrap();
        
        let models = api.list_models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0], "test_linear");
    }
    
    #[tokio::test]
    async fn test_api_unknown_model_type() {
        let api = ContinuumApi::default();
        
        let result = api.register_model("test_unknown", "unknown", None).await;
        assert!(result.is_err());
        
        if let Err(ApiError::InvalidInput(msg)) = result {
            assert!(msg.contains("Unknown model type"));
        } else {
            panic!("Expected InvalidInput error");
        }
    }
    
    #[tokio::test]
    async fn test_api_model_lifecycle() {
        let api = ContinuumApi::default();
        
        // Register a model
        api.register_model("test_model", "linear", None).await.unwrap();
        
        // Add training examples
        for i in 0..5 {
            api.add_training_example(
                "test_model",
                vec![i as f32],
                (i * 2) as f32,
                false,
            ).await.unwrap();
        }
        
        // Train the model
        api.train_model("test_model").await.unwrap();
        
        // Make a prediction
        let response = api.predict("test_model", vec![5.0]).await.unwrap();
        assert!(response.model_version >= 1);
        
        // Get model info
        let info = api.get_model_info("test_model").await.unwrap();
        assert_eq!(info.name, "test_model");
        assert!(info.version >= 1);
    }
}