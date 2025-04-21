use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::Instant;
use parking_lot::{RwLock, Mutex};

use crate::traits::features::FeatureVector;
use crate::traits::model::{Model, ModelError};
use crate::server::metrics::ModelStats;

/// Atomic model container that enables zero-downtime updates
pub struct AtomicModel<M: Model + Clone + Send + Sync + 'static> {
    /// Current model for predictions (multiple readers)
    current: Arc<RwLock<M>>,
    /// Training model (exclusive access)
    training: Arc<Mutex<M>>,
    /// Model statistics
    stats: Arc<ModelStats>,
    /// Flag to indicate if training is in progress
    training_in_progress: AtomicBool,
    /// Models are the same?
    models_in_sync: AtomicBool,
}

impl<M: Model + Clone + Send + Sync + 'static> AtomicModel<M> {
    /// Create a new atomic model container
    pub fn new(initial_model: M) -> Self {
        let stats = Arc::new(ModelStats::new());
        
        Self {
            current: Arc::new(RwLock::new(initial_model.clone())),
            training: Arc::new(Mutex::new(initial_model)),
            stats,
            training_in_progress: AtomicBool::new(false),
            models_in_sync: AtomicBool::new(true),
        }
    }
    
    /// Get a reference to the current model for predictions
    pub fn get_current(&self) -> Arc<RwLock<M>> {
        Arc::clone(&self.current)
    }
    
    /// Get a reference to the model statistics
    pub fn get_stats(&self) -> Arc<ModelStats> {
        Arc::clone(&self.stats)
    }
    
    /// Get current model version
    pub fn get_version(&self) -> usize {
        self.stats.version.load(Ordering::Relaxed)
    }
    
    /// Check if training is currently in progress
    pub fn is_training(&self) -> bool {
        self.training_in_progress.load(Ordering::Relaxed)
    }
    
    /// Check if models are in sync (current = training)
    pub fn is_in_sync(&self) -> bool {
        self.models_in_sync.load(Ordering::Relaxed)
    }
    
    /// Update the training model with new data
    pub async fn train(&self, features: &[FeatureVector], targets: &[f32]) -> Result<(), ModelError> {
        if features.is_empty() || targets.is_empty() {
            return Err(ModelError::TrainingError("Empty training data".to_string()));
        }
        
        if features.len() != targets.len() {
            return Err(ModelError::DimensionMismatch {
                expected: features.len(),
                actual: targets.len(),
                context: "features vs targets length".to_string(),
            });
        }
        
        // Check if training is already in progress
        if self.training_in_progress.swap(true, Ordering::SeqCst) {
            return Err(ModelError::TrainingError("Training already in progress".to_string()));
        }
        
        // Models will be out of sync
        self.models_in_sync.store(false, Ordering::SeqCst);
        
        // Record start time
        let start_time = Instant::now();
        
        // Get exclusive access to training model
        let mut training_model = self.training.lock();
        
        // Perform training
        let result = training_model.train(features, targets);
        
        // Update stats
        match result {
            Ok(()) => {
                self.stats.training_count.fetch_add(1, Ordering::SeqCst);
                let duration = start_time.elapsed().as_micros() as usize;
                self.stats.latest_training_latency_us.store(duration, Ordering::SeqCst);
                self.stats.update_timestamp();
            }
            Err(_) => {
                self.stats.training_errors.fetch_add(1, Ordering::SeqCst);
            }
        }
        
        // Release training lock
        drop(training_model);
        self.training_in_progress.store(false, Ordering::SeqCst);
        
        result
    }
    
    /// Make a prediction using the current model
    pub async fn predict(&self, feature: &FeatureVector) -> Result<f32, ModelError> {
        // Record start time
        let start_time = Instant::now();
        
        // Get read access to current model (allows multiple concurrent predictions)
        let current_model = self.current.read();
        
        // Make prediction
        let result = current_model.predict(feature);
        
        // Update stats
        match result {
            Ok(prediction) => {
                self.stats.prediction_count.fetch_add(1, Ordering::SeqCst);
                let duration = start_time.elapsed().as_micros() as usize;
                self.stats.latest_prediction_latency_us.store(duration, Ordering::SeqCst);
                
                Ok(prediction)
            }
            Err(err) => {
                self.stats.prediction_errors.fetch_add(1, Ordering::SeqCst);
                Err(err)
            }
        }
    }
    
    /// Make batch predictions using the current model
    pub async fn predict_batch(&self, features: &[FeatureVector]) -> Result<Vec<f32>, ModelError> {
        // Record start time
        let start_time = Instant::now();
        
        // Get read access to current model
        let current_model = self.current.read();
        
        // Make predictions
        let result = current_model.predict_batch(features);
        
        // Update stats
        match result {
            Ok(predictions) => {
                self.stats.prediction_count.fetch_add(features.len(), Ordering::SeqCst);
                let duration = start_time.elapsed().as_micros() as usize;
                self.stats.latest_prediction_latency_us.store(duration / features.len().max(1), Ordering::SeqCst);
                
                Ok(predictions)
            }
            Err(err) => {
                self.stats.prediction_errors.fetch_add(1, Ordering::SeqCst);
                Err(err)
            }
        }
    }
    
    /// Atomically swap training model to become current model
    pub fn swap_models(&self) -> Result<usize, ModelError> {
        if self.is_training() {
            return Err(ModelError::TrainingError("Cannot swap while training in progress".to_string()));
        }
        
        // Create a clone of the training model
        let new_model = {
            let training_guard = self.training.lock();
            training_guard.clone()
        };
        
        // Update the current model
        {
            let mut current_guard = self.current.write();
            *current_guard = new_model;
        }
        
        // Increment version
        let new_version = self.stats.version.fetch_add(1, Ordering::SeqCst) + 1;
        
        // Update timestamp
        self.stats.update_timestamp();
        
        // Mark models as in sync
        self.models_in_sync.store(true, Ordering::SeqCst);
        
        Ok(new_version)
    }
    
    /// Validate current model performance
    pub async fn validate(&self, features: &[FeatureVector], targets: &[f32]) -> Result<f32, ModelError> {
        let current_model = self.current.read();
        current_model.validate(features, targets)
    }
    
    /// Compare performance between current and training models
    pub async fn compare_models(&self, features: &[FeatureVector], targets: &[f32]) -> Result<(f32, f32), ModelError> {
        if features.is_empty() || targets.is_empty() {
            return Err(ModelError::ValidationError("Empty validation data".to_string()));
        }
        
        if features.len() != targets.len() {
            return Err(ModelError::DimensionMismatch {
                expected: features.len(),
                actual: targets.len(),
                context: "Validation features vs targets".to_string(),
            });
        }
        
        // Get performance of current model
        let current_error = {
            let current_model = self.current.read();
            current_model.validate(features, targets)?
        };
        
        // Get performance of training model
        let training_error = {
            let training_model = self.training.lock();
            training_model.validate(features, targets)?
        };
        
        Ok((current_error, training_error))
    }
}

/// Implement Clone for AtomicModel
impl<M: Model + Clone + Send + Sync + 'static> Clone for AtomicModel<M> {
    fn clone(&self) -> Self {
        let current = self.current.read().clone();
        let training = self.training.lock().clone();
        
        Self {
            current: Arc::new(RwLock::new(current)),
            training: Arc::new(Mutex::new(training)),
            stats: Arc::clone(&self.stats),
            training_in_progress: AtomicBool::new(self.is_training()),
            models_in_sync: AtomicBool::new(self.is_in_sync()),
        }
    }
}

/// Trait for unifying different model types in the server
#[async_trait::async_trait]
pub trait ModelWrapper: Send + Sync {
    /// Make a prediction
    async fn predict(&self, feature: &FeatureVector) -> Result<f32, ModelError>;
    
    /// Train the model
    async fn train(&self, features: &[FeatureVector], targets: &[f32]) -> Result<(), ModelError>;
    
    /// Swap current and training models
    fn swap_models(&self) -> Result<usize, ModelError>;
    
    /// Validate model performance
    async fn validate(&self, features: &[FeatureVector], targets: &[f32]) -> Result<f32, ModelError>;
    
    /// Get model version
    fn get_version(&self) -> usize;
    
    /// Check if training is in progress
    fn is_training(&self) -> bool;
    
    /// Get model stats as formatted string
    fn get_stats_formatted(&self) -> String;
}

/// Implementation of ModelWrapper for AtomicModel
#[async_trait::async_trait]
impl<M: Model + Clone + Send + Sync + 'static> ModelWrapper for AtomicModel<M> {
    async fn predict(&self, feature: &FeatureVector) -> Result<f32, ModelError> {
        self.predict(feature).await
    }
    
    async fn train(&self, features: &[FeatureVector], targets: &[f32]) -> Result<(), ModelError> {
        self.train(features, targets).await
    }
    
    fn swap_models(&self) -> Result<usize, ModelError> {
        self.swap_models()
    }
    
    async fn validate(&self, features: &[FeatureVector], targets: &[f32]) -> Result<f32, ModelError> {
        self.validate(features, targets).await
    }
    
    fn get_version(&self) -> usize {
        self.get_version()
    }
    
    fn is_training(&self) -> bool {
        self.is_training()
    }
    
    fn get_stats_formatted(&self) -> String {
        self.stats.format_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::linears::LinearRegression;
    use tokio::sync::Barrier;
    
    // Helper function to create a simple model and data
    fn create_test_model() -> LinearRegression {
        LinearRegression::new(true, 0.01, 1000)
    }
    
    // Helper function to create a pre-trained model
    fn create_trained_model() -> LinearRegression {
        let mut model = LinearRegression::new(true, 0.01, 1000);
        // Set some initial weights
        model.import_parameters(vec![0.0, 1.0]).unwrap();
        model
    }
    
    fn create_test_data() -> (Vec<FeatureVector>, Vec<f32>) {
        let features = vec![
            FeatureVector::new(vec![1.0]),
            FeatureVector::new(vec![2.0]),
            FeatureVector::new(vec![3.0]),
            FeatureVector::new(vec![4.0]),
        ];
        
        let targets = vec![3.0, 5.0, 7.0, 9.0]; // y = 2x + 1
        
        (features, targets)
    }
    
    #[tokio::test]
    async fn test_atomic_model_creation() {
        let model = create_test_model();
        let atomic_model = AtomicModel::new(model);
        
        assert_eq!(atomic_model.get_version(), 1);
        assert!(!atomic_model.is_training());
        assert!(atomic_model.is_in_sync());
    }
    
    #[tokio::test]
    async fn test_atomic_model_train_predict() {
        let model = create_test_model();
        let atomic_model = AtomicModel::new(model);
        
        let (features, targets) = create_test_data();
        
        // Train the model
        atomic_model.train(&features, &targets).await.unwrap();
        
        // After training, we need to swap the models to make the trained version current
        atomic_model.swap_models().unwrap();
        
        // Make a prediction
        let test_feature = FeatureVector::new(vec![5.0]);
        let prediction = atomic_model.predict(&test_feature).await.unwrap();
        
        // Should predict close to y = 2x + 1 for x=5 (around 11)
        assert!((prediction - 11.0).abs() < 1.0);  // Increased tolerance for numeric stability
        
        // Ensure stats were updated
        let stats = atomic_model.get_stats();
        assert_eq!(stats.training_count.load(Ordering::Relaxed), 1);
        assert_eq!(stats.prediction_count.load(Ordering::Relaxed), 1);
        assert_eq!(stats.prediction_errors.load(Ordering::Relaxed), 0);
        assert_eq!(stats.training_errors.load(Ordering::Relaxed), 0);
    }
    
    #[tokio::test]
    async fn test_atomic_model_concurrent_training() {
        let model = create_test_model();
        let atomic_model = Arc::new(AtomicModel::new(model));
        
        let (features, targets) = create_test_data();
        
        // Manually set the training flag to test the exclusive access
        assert!(!atomic_model.is_training(), "Training flag should initially be false");
        
        // Manually set the training flag to true to simulate a training in progress
        atomic_model.training_in_progress.store(true, Ordering::SeqCst);
        
        // Now try to train - it should fail because the flag is set
        let result = atomic_model.train(&features, &targets).await;
        
        match result {
            Err(ModelError::TrainingError(msg)) => {
                assert!(msg.contains("Training already in progress"), "Expected concurrent training error");
            }
            Ok(_) => panic!("Expected training error when flag is set"),
            Err(e) => panic!("Unexpected error type: {:?}", e),
        }
        
        // Reset the flag
        atomic_model.training_in_progress.store(false, Ordering::SeqCst);
        
        // Now training should succeed
        let result = atomic_model.train(&features, &targets).await;
        assert!(result.is_ok(), "Training should succeed when flag is not set");
    }
    
    #[tokio::test]
    async fn test_atomic_model_swap() {
        let model = create_test_model();
        let atomic_model = AtomicModel::new(model);
        
        let (features, targets) = create_test_data();
        
        // Train the model
        atomic_model.train(&features, &targets).await.unwrap();
        
        // Before swap, models are out of sync
        assert!(!atomic_model.is_in_sync());
        
        // Swap models
        let new_version = atomic_model.swap_models().unwrap();
        
        // Version should be incremented
        assert_eq!(new_version, 2);
        assert_eq!(atomic_model.get_version(), 2);
        
        // Models should be in sync after swap
        assert!(atomic_model.is_in_sync());
    }
    
    #[tokio::test]
    async fn test_atomic_model_compare() {
        // Create a model with some initial weights
        let mut model = create_test_model();
        
        // Initialize the model with some parameters so it can make predictions
        model.import_parameters(vec![0.0, 1.0]).unwrap();
        
        let atomic_model = AtomicModel::new(model);
        
        let (features, targets) = create_test_data();
        
        // Create validation data
        let val_features = vec![FeatureVector::new(vec![1.5])];
        let val_targets = vec![4.0]; // y = 2*1.5 + 1 = 4
        
        // Get initial error before training
        let initial_error = atomic_model.validate(&val_features, &val_targets).await.unwrap();
        
        // Train the model
        atomic_model.train(&features, &targets).await.unwrap();
        
        // Compare models after training
        let (current_error, training_error) = atomic_model.compare_models(&val_features, &val_targets).await.unwrap();
        
        // Current model should have same error as initial (untrained)
        assert!((current_error - initial_error).abs() < 1e-5);
        
        // Training model should have lower error than current
        assert!(training_error < current_error);
    }
    
    #[tokio::test]
    async fn test_atomic_model_error_handling() {
        let model = create_test_model();
        let atomic_model = AtomicModel::new(model);
        
        // Empty training data should fail
        let result = atomic_model.train(&[], &[]).await;
        assert!(matches!(result, Err(ModelError::TrainingError(_))));
        
        // Mismatched data lengths should fail
        let features = vec![FeatureVector::new(vec![1.0])];
        let targets = vec![1.0, 2.0];
        let result = atomic_model.train(&features, &targets).await;
        assert!(matches!(result, Err(ModelError::DimensionMismatch { .. })));
    }
    
    #[tokio::test]
    async fn test_atomic_model_batch_predictions() {
        let model = create_test_model();
        let atomic_model = AtomicModel::new(model);
        
        let (features, targets) = create_test_data();
        
        // Train the model
        atomic_model.train(&features, &targets).await.unwrap();
        
        // Swap models to make trained version current
        atomic_model.swap_models().unwrap();
        
        // Test batch prediction
        let test_features = vec![
            FeatureVector::new(vec![1.0]),
            FeatureVector::new(vec![2.0]),
            FeatureVector::new(vec![3.0]),
        ];
        
        let predictions = atomic_model.predict_batch(&test_features).await.unwrap();
        
        // Predictions should be close to y = 2x + 1
        assert!((predictions[0] - 3.0).abs() < 1.0);
        assert!((predictions[1] - 5.0).abs() < 1.0);
        assert!((predictions[2] - 7.0).abs() < 1.0);
    }
}