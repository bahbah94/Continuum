use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::RwLock;

use crate::traits::features::FeatureVector;
use crate::traits::model::{Model, ModelError};
use crate::server::model_server::{AtomicModel, ModelWrapper};
use crate::server::continuous_learning::{TrainingBuffer, ContinuousLearningConfig};

/// Server for managing multiple models
pub struct ModelServer {
    /// Map of model name to atomic model instance
    models: Arc<RwLock<HashMap<String, Arc<dyn ModelWrapper>>>>,
    /// Map of model name to training data buffer
    training_buffers: Arc<RwLock<HashMap<String, TrainingBuffer>>>,
    /// Server configuration
    config: ContinuousLearningConfig,
    /// Is the server running?
    running: Arc<AtomicBool>,
}

impl ModelServer {
    /// Create a new model server
    pub fn new(config: ContinuousLearningConfig) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            training_buffers: Arc::new(RwLock::new(HashMap::new())),
            config,
            running: Arc::new(AtomicBool::new(false)),
        }
    }
    
    /// Create a new model server with default configuration
    pub fn default() -> Self {
        Self::new(ContinuousLearningConfig::default())
    }
    
    /// Register a new model with the server
    pub async fn register_model<M: Model + Clone + Send + Sync + 'static>(
        &self,
        name: &str,
        model: M,
    ) -> Result<(), ModelError> {
        let mut models = self.models.write().await;
        
        if models.contains_key(name) {
            return Err(ModelError::InvalidParameter(format!("Model '{}' already exists", name)));
        }
        
        // Create atomic model container
        let atomic_model = AtomicModel::new(model);
        models.insert(name.to_string(), Arc::new(atomic_model));
        
        // Initialize training buffer
        let mut buffers = self.training_buffers.write().await;
        buffers.insert(name.to_string(), TrainingBuffer::new());
        
        Ok(())
    }
    
    /// Unregister a model from the server
    pub async fn unregister_model(&self, name: &str) -> Result<(), ModelError> {
        let mut models = self.models.write().await;
        let mut buffers = self.training_buffers.write().await;
        
        if !models.contains_key(name) {
            return Err(ModelError::InvalidParameter(format!("Model '{}' not found", name)));
        }
        
        models.remove(name);
        buffers.remove(name);
        
        Ok(())
    }
    
    /// Get a reference to a model
    pub async fn get_model(&self, name: &str) -> Result<Arc<dyn ModelWrapper>, ModelError> {
        let models = self.models.read().await;
        
        match models.get(name) {
            Some(model) => Ok(Arc::clone(model)),
            None => Err(ModelError::InvalidParameter(format!("Model '{}' not found", name))),
        }
    }
    
    /// Make a prediction using a named model
    pub async fn predict(&self, name: &str, feature: &FeatureVector) -> Result<f32, ModelError> {
        let model = self.get_model(name).await?;
        model.predict(feature).await
    }
    
    /// Make batch predictions using a named model
    pub async fn predict_batch(&self, name: &str, features: &[FeatureVector]) -> Result<Vec<f32>, ModelError> {
        let model = self.get_model(name).await?;
        
        // Using the ModelWrapper trait, we need to convert the batch prediction to individual predictions
        let mut predictions = Vec::with_capacity(features.len());
        for feature in features {
            predictions.push(model.predict(feature).await?);
        }
        
        Ok(predictions)
    }
    
    /// Add a new training example (will be applied automatically by continuous learning)
    pub async fn add_training_example(
        &self,
        name: &str,
        feature: FeatureVector,
        target: f32,
        is_validation: bool,
    ) -> Result<(), ModelError> {
        let mut buffers = self.training_buffers.write().await;
        
        match buffers.get_mut(name) {
            Some(buffer) => {
                buffer.add(feature, target, is_validation);
                Ok(())
            }
            None => Err(ModelError::InvalidParameter(format!("Model '{}' not found", name))),
        }
    }
    
    /// Force training for a model immediately
    pub async fn train_now(&self, name: &str) -> Result<(), ModelError> {
        // Get the model
        let model = self.get_model(name).await?;
        
        // Get the training buffer
        let mut buffers = self.training_buffers.write().await;
        let buffer = match buffers.get_mut(name) {
            Some(buffer) => buffer,
            None => return Err(ModelError::InvalidParameter(format!("Model '{}' not found", name))),
        };
        
        // Skip if no training data
        if buffer.features.is_empty() {
            return Ok(());
        }
        
        // Clone the training data
        let features = buffer.features.clone();
        let targets = buffer.targets.clone();
        
        // Train the model
        model.train(&features, &targets).await?;
        
        // Clear the training buffer
        buffer.clear_training();
        
        // If auto-swap is enabled, swap models
        if self.config.auto_swap {
            // If validation data exists, validate before swapping
            if !buffer.val_features.is_empty() {
                // Validate current model
                let old_error = model.validate(&buffer.val_features, &buffer.val_targets).await?;
                
                // First swap to the new model
                model.swap_models()?;
                
                // Validate new model
                let new_error = model.validate(&buffer.val_features, &buffer.val_targets).await?;
                
                // If new model is not better by threshold, log warning
                if new_error > old_error * (1.0 - self.config.validation_threshold) {
                    println!("Warning: New model ({}) doesn't improve validation error by threshold (old: {}, new: {})",
                        name, old_error, new_error);
                }
            } else {
                // No validation data, just swap
                model.swap_models()?;
            }
        }
        
        Ok(())
    }
    
    /// Start the continuous learning background task
    pub async fn start_continuous_learning(&self) -> Result<(), ModelError> {
        if !self.config.enabled {
            return Ok(());
        }
        
        if self.running.load(Ordering::SeqCst) {
            return Ok(()); // Already running
        }
        
        self.running.store(true, Ordering::SeqCst);
        
        // Clone Arc references for the background task
        let models = Arc::clone(&self.models);
        let buffers = Arc::clone(&self.training_buffers);
        let config = self.config.clone();
        let running = Arc::clone(&self.running);
        
        // Spawn background task
        tokio::spawn(async move {
            while running.load(Ordering::SeqCst) {
                // Wait for next training interval
                tokio::time::sleep(Duration::from_secs(config.interval_sec)).await;
                
                // Get all model names
                let model_names: Vec<String> = {
                    let models = models.read().await;
                    models.keys().cloned().collect()
                };
                
                // Process each model
                for name in model_names {
                    // Check if model has enough training data
                    let should_train = {
                        let buffers = buffers.read().await;
                        match buffers.get(&name) {
                            Some(buffer) => buffer.has_min_samples(config.min_samples),
                            None => false,
                        }
                    };
                    
                    if should_train {
                        // Get the model
                        let model = match models.read().await.get(&name) {
                            Some(model) => Arc::clone(model),
                            None => continue,
                        };
                        
                        // Skip if already training
                        if model.is_training() {
                            continue;
                        }
                        
                        // Get training data
                        let (features, targets) = {
                            let mut buffers = buffers.write().await;
                            let buffer = match buffers.get_mut(&name) {
                                Some(buffer) => buffer,
                                None => continue,
                            };
                            
                            let features = buffer.features.clone();
                            let targets = buffer.targets.clone();
                            
                            // Clear the buffer
                            buffer.clear_training();
                            
                            (features, targets)
                        };
                        
                        // Train the model
                        if let Err(err) = model.train(&features, &targets).await {
                            println!("Error training model {}: {}", name, err);
                            continue;
                        }
                        
                        // Get validation data
                        let (val_features, val_targets) = {
                            let buffers = buffers.read().await;
                            let buffer = match buffers.get(&name) {
                                Some(buffer) => buffer,
                                None => continue,
                            };
                            
                            (buffer.val_features.clone(), buffer.val_targets.clone())
                        };
                        
                        // If auto-swap is enabled and validation data exists
                        if config.auto_swap && !val_features.is_empty() {
                            // Validate current model
                            let old_error = match model.validate(&val_features, &val_targets).await {
                                Ok(err) => err,
                                Err(_) => continue,
                            };
                            
                            // Swap models
                            if let Err(_) = model.swap_models() {
                                continue;
                            }
                            
                            // Validate new model
                            let new_error = match model.validate(&val_features, &val_targets).await {
                                Ok(err) => err,
                                Err(_) => continue,
                            };
                            
                            // Log improvement
                            println!("Model {} updated: Error changed from {} to {}", 
                                name, old_error, new_error);
                        } else if config.auto_swap {
                            // No validation data, just swap
                            if let Err(err) = model.swap_models() {
                                println!("Error swapping model {}: {}", name, err);
                            } else {
                                println!("Model {} updated to version {}", 
                                    name, model.get_version());
                            }
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Stop the continuous learning background task
    pub fn stop_continuous_learning(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
    
    /// Get list of all registered models
    pub async fn list_models(&self) -> Vec<String> {
        let models = self.models.read().await;
        models.keys().cloned().collect()
    }
    
    /// Get model statistics
    pub async fn get_model_stats(&self, name: &str) -> Result<String, ModelError> {
        let model = self.get_model(name).await?;
        Ok(model.get_stats_formatted())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::linears::LinearRegression;
    
    #[tokio::test]
    async fn test_model_server_register_unregister() {
        let server = ModelServer::default();
        
        // Register a model
        let model = LinearRegression::new(true, 0.01, 1000);
        server.register_model("test_model", model).await.unwrap();
        
        // List models
        let models = server.list_models().await;
        assert_eq!(models.len(), 1);
        assert_eq!(models[0], "test_model");
        
        // Unregister model
        server.unregister_model("test_model").await.unwrap();
        
        // List models again
        let models = server.list_models().await;
        assert_eq!(models.len(), 0);
    }
    
    #[tokio::test]
    async fn test_model_server_duplicate_registration() {
        let server = ModelServer::default();
        
        // Register a model
        let model1 = LinearRegression::new(true, 0.01, 1000);
        server.register_model("test_model", model1).await.unwrap();
        
        // Try to register another model with the same name
        let model2 = LinearRegression::new(true, 0.01, 1000);
        let result = server.register_model("test_model", model2).await;
        
        assert!(result.is_err());
        if let Err(ModelError::InvalidParameter(msg)) = result {
            assert!(msg.contains("already exists"));
        } else {
            panic!("Expected InvalidParameter error");
        }
    }
    
    #[tokio::test]
    async fn test_model_server_prediction() {
        let server = ModelServer::default();
        
        // Register and train a model
        let mut model = LinearRegression::new(true, 0.01, 1000);
        let features = vec![
            FeatureVector::new(vec![1.0]),
            FeatureVector::new(vec![2.0]),
            FeatureVector::new(vec![3.0]),
        ];
        let targets = vec![2.0, 4.0, 6.0]; // y = 2x
        
        // Train the model before registering
        model.train(&features, &targets).unwrap();
        
        server.register_model("test_model", model).await.unwrap();
        
        // Make a prediction
        let test_feature = FeatureVector::new(vec![4.0]);
        let prediction = server.predict("test_model", &test_feature).await.unwrap();
        
        // Should predict close to y = 2*4 = 8
        assert!((prediction - 8.0).abs() < 0.5);
    }
    
    #[tokio::test]
    async fn test_model_server_add_training_examples() {
        let server = ModelServer::default();
        
        // Register a model
        let model = LinearRegression::new(true, 0.01, 1000);
        server.register_model("test_model", model).await.unwrap();
        
        // Add training examples
        for i in 0..5 {
            let feature = FeatureVector::new(vec![i as f32]);
            server.add_training_example("test_model", feature, (i * 2) as f32, false).await.unwrap();
        }
        
        // Check that examples are buffered
        let buffers = server.training_buffers.read().await;
        let buffer = buffers.get("test_model").unwrap();
        assert_eq!(buffer.features.len(), 5);
        assert_eq!(buffer.targets.len(), 5);
    }
    
    #[tokio::test]
    async fn test_model_server_train_now() {
        let server = ModelServer::default();
        
        // Register a model
        let model = LinearRegression::new(true, 0.01, 1000);
        server.register_model("test_model", model).await.unwrap();
        
        // Add training examples
        for i in 0..5 {
            let feature = FeatureVector::new(vec![i as f32]);
            server.add_training_example("test_model", feature, (i * 2) as f32, false).await.unwrap();
        }
        
        // Train the model
        server.train_now("test_model").await.unwrap();
        
        // Check that buffer is cleared
        let buffers = server.training_buffers.read().await;
        let buffer = buffers.get("test_model").unwrap();
        assert_eq!(buffer.features.len(), 0);
        assert_eq!(buffer.targets.len(), 0);
    }
    
    #[tokio::test]
    async fn test_model_server_get_stats() {
        let server = ModelServer::default();
        
        // Register a model
        let model = LinearRegression::new(true, 0.01, 1000);
        server.register_model("test_model", model).await.unwrap();
        
        // Get stats
        let stats = server.get_model_stats("test_model").await.unwrap();
        assert!(stats.contains("Model v1"));
        assert!(stats.contains("Predictions: 0"));
        assert!(stats.contains("Training runs: 0"));
    }
}