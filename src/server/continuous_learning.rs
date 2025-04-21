use std::time::SystemTime;
use crate::traits::features::FeatureVector;

/// Configuration for continuous learning
#[derive(Debug, Clone)]
pub struct ContinuousLearningConfig {
    /// Whether continuous learning is enabled
    pub enabled: bool,
    /// How often to check for new training data (in seconds)
    pub interval_sec: u64,
    /// Minimum number of samples before training
    pub min_samples: usize,
    /// Whether to automatically swap models after training
    pub auto_swap: bool,
    /// Validation threshold to determine if new model is better  
    pub validation_threshold: f32,
    /// Whether to use KL divergence for swap decisions
    pub use_kl_divergence: bool,
}

impl Default for ContinuousLearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_sec: 60,
            min_samples: 100,
            auto_swap: true,
            validation_threshold: 0.05, // 5% improvement required
            use_kl_divergence: false,
        }
    }
}

impl ContinuousLearningConfig {
    /// Create a new configuration with custom values
    pub fn new(
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
    
    /// Create a disabled configuration (useful for testing)
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
    
    /// Create a configuration optimized for frequent updates
    pub fn frequent_updates() -> Self {
        Self {
            interval_sec: 10,
            min_samples: 10,
            validation_threshold: 0.01,
            ..Default::default()
        }
    }
}

/// Buffer for accumulating training data
#[derive(Debug)]
pub struct TrainingBuffer {
    /// Feature vectors for training
    pub features: Vec<FeatureVector>,
    /// Target values for training
    pub targets: Vec<f32>,
    /// Validation feature vectors
    pub val_features: Vec<FeatureVector>,
    /// Validation target values
    pub val_targets: Vec<f32>,
    /// Last time the buffer was trained
    pub last_trained: SystemTime,
    /// Maximum buffer size (after which oldest entries are dropped)
    pub max_size: Option<usize>,
}

impl TrainingBuffer {
    /// Create a new training buffer
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            targets: Vec::new(),
            val_features: Vec::new(),
            val_targets: Vec::new(),
            last_trained: SystemTime::now(),
            max_size: None,
        }
    }
    
    /// Create a new training buffer with a maximum size
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            features: Vec::new(),
            targets: Vec::new(),
            val_features: Vec::new(),
            val_targets: Vec::new(),
            last_trained: SystemTime::now(),
            max_size: Some(max_size),
        }
    }
    
    /// Add a new training example
    pub fn add(&mut self, feature: FeatureVector, target: f32, is_validation: bool) {
        if is_validation {
            self.val_features.push(feature);
            self.val_targets.push(target);
            
            // Enforce max size for validation data
            if let Some(max_size) = self.max_size {
                if self.val_features.len() > max_size {
                    self.val_features.remove(0);
                    self.val_targets.remove(0);
                }
            }
        } else {
            self.features.push(feature);
            self.targets.push(target);
            
            // Enforce max size for training data
            if let Some(max_size) = self.max_size {
                if self.features.len() > max_size {
                    self.features.remove(0);
                    self.targets.remove(0);
                }
            }
        }
    }
    
    /// Check if buffer has enough samples for training
    pub fn has_min_samples(&self, min_samples: usize) -> bool {
        self.features.len() >= min_samples
    }
    
    /// Clear training data (but keep validation data)
    pub fn clear_training(&mut self) {
        self.features.clear();
        self.targets.clear();
        self.last_trained = SystemTime::now();
    }
    
    /// Clear validation data
    pub fn clear_validation(&mut self) {
        self.val_features.clear();
        self.val_targets.clear();
    }
    
    /// Clear all data
    pub fn clear_all(&mut self) {
        self.clear_training();
        self.clear_validation();
    }
    
    /// Get the time since last training in seconds
    pub fn time_since_last_training(&self) -> std::time::Duration {
        self.last_trained.elapsed().unwrap_or_default()
    }
    
    /// Get the current buffer sizes
    pub fn get_sizes(&self) -> (usize, usize) {
        (self.features.len(), self.val_features.len())
    }
    
    /// Get training data as references
    pub fn get_training_data(&self) -> (&[FeatureVector], &[f32]) {
        (&self.features, &self.targets)
    }
    
    /// Get validation data as references
    pub fn get_validation_data(&self) -> (&[FeatureVector], &[f32]) {
        (&self.val_features, &self.val_targets)
    }
}

impl Default for TrainingBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_continuous_learning_config_default() {
        let config = ContinuousLearningConfig::default();
        assert!(config.enabled);
        assert_eq!(config.interval_sec, 60);
        assert_eq!(config.min_samples, 100);
        assert!(config.auto_swap);
        assert_eq!(config.validation_threshold, 0.05);
        assert!(!config.use_kl_divergence);
    }
    
    #[test]
    fn test_continuous_learning_config_custom() {
        let config = ContinuousLearningConfig::new(
            false,
            30,
            50,
            false,
            0.1,
            true,
        );
        
        assert!(!config.enabled);
        assert_eq!(config.interval_sec, 30);
        assert_eq!(config.min_samples, 50);
        assert!(!config.auto_swap);
        assert_eq!(config.validation_threshold, 0.1);
        assert!(config.use_kl_divergence);
    }
    
    #[test]
    fn test_continuous_learning_config_disabled() {
        let config = ContinuousLearningConfig::disabled();
        assert!(!config.enabled);
    }
    
    #[test]
    fn test_continuous_learning_config_frequent_updates() {
        let config = ContinuousLearningConfig::frequent_updates();
        assert_eq!(config.interval_sec, 10);
        assert_eq!(config.min_samples, 10);
        assert_eq!(config.validation_threshold, 0.01);
    }
    
    #[test]
    fn test_training_buffer_add() {
        let mut buffer = TrainingBuffer::new();
        
        // Add training data
        let feature = FeatureVector::new(vec![1.0, 2.0, 3.0]);
        buffer.add(feature.clone(), 4.0, false);
        
        assert_eq!(buffer.features.len(), 1);
        assert_eq!(buffer.targets.len(), 1);
        assert_eq!(buffer.val_features.len(), 0);
        assert_eq!(buffer.val_targets.len(), 0);
        
        // Add validation data
        buffer.add(feature, 4.0, true);
        
        assert_eq!(buffer.features.len(), 1);
        assert_eq!(buffer.targets.len(), 1);
        assert_eq!(buffer.val_features.len(), 1);
        assert_eq!(buffer.val_targets.len(), 1);
    }
    
    #[test]
    fn test_training_buffer_max_size() {
        let mut buffer = TrainingBuffer::with_max_size(3);
        
        // Add 5 training samples (should only keep the last 3)
        for i in 0..5 {
            let feature = FeatureVector::new(vec![i as f32]);
            buffer.add(feature, i as f32, false);
        }
        
        assert_eq!(buffer.features.len(), 3);
        assert_eq!(buffer.targets.len(), 3);
        
        // Verify oldest entries were dropped
        assert_eq!(buffer.targets[0], 2.0);
        assert_eq!(buffer.targets[1], 3.0);
        assert_eq!(buffer.targets[2], 4.0);
    }
    
    #[test]
    fn test_training_buffer_has_min_samples() {
        let mut buffer = TrainingBuffer::new();
        
        assert!(!buffer.has_min_samples(1));
        
        let feature = FeatureVector::new(vec![1.0, 2.0, 3.0]);
        buffer.add(feature, 4.0, false);
        
        assert!(buffer.has_min_samples(1));
        assert!(!buffer.has_min_samples(2));
    }
    
    #[test]
    fn test_training_buffer_clear() {
        let mut buffer = TrainingBuffer::new();
        
        // Add both training and validation data
        let feature = FeatureVector::new(vec![1.0, 2.0, 3.0]);
        buffer.add(feature.clone(), 4.0, false);
        buffer.add(feature, 4.0, true);
        
        // Clear training data only
        buffer.clear_training();
        
        assert_eq!(buffer.features.len(), 0);
        assert_eq!(buffer.targets.len(), 0);
        assert_eq!(buffer.val_features.len(), 1);
        assert_eq!(buffer.val_targets.len(), 1);
        
        // Clear all data
        buffer.clear_all();
        
        assert_eq!(buffer.features.len(), 0);
        assert_eq!(buffer.targets.len(), 0);
        assert_eq!(buffer.val_features.len(), 0);
        assert_eq!(buffer.val_targets.len(), 0);
    }
    
    #[test]
    fn test_training_buffer_time_since_last_training() {
        let buffer = TrainingBuffer::new();
        
        // Sleep for a short duration
        std::thread::sleep(std::time::Duration::from_millis(10));
        
        let elapsed = buffer.time_since_last_training();
        assert!(elapsed >= std::time::Duration::from_millis(10));
    }
    
    #[test]
    fn test_training_buffer_get_sizes() {
        let mut buffer = TrainingBuffer::new();
        let feature = FeatureVector::new(vec![1.0, 2.0, 3.0]);
        
        // Add 3 training samples and 2 validation samples
        for i in 0..3 {
            buffer.add(feature.clone(), i as f32, false);
        }
        for i in 0..2 {
            buffer.add(feature.clone(), i as f32, true);
        }
        
        let (train_size, val_size) = buffer.get_sizes();
        assert_eq!(train_size, 3);
        assert_eq!(val_size, 2);
    }
    
    #[test]
    fn test_training_buffer_get_data() {
        let mut buffer = TrainingBuffer::new();
        let feature1 = FeatureVector::new(vec![1.0]);
        let feature2 = FeatureVector::new(vec![2.0]);
        
        buffer.add(feature1.clone(), 10.0, false);
        buffer.add(feature2.clone(), 20.0, true);
        
        let (features, targets) = buffer.get_training_data();
        assert_eq!(features.len(), 1);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0], 10.0);
        
        let (val_features, val_targets) = buffer.get_validation_data();
        assert_eq!(val_features.len(), 1);
        assert_eq!(val_targets.len(), 1);
        assert_eq!(val_targets[0], 20.0);
    }
}