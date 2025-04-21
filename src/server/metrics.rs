use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::SystemTime;

/// Container for tracking model statistics
#[derive(Debug)]
pub struct ModelStats {
    /// Total number of predictions made
    pub prediction_count: AtomicUsize,
    /// Total number of training batches processed
    pub training_count: AtomicUsize,
    /// Number of prediction errors
    pub prediction_errors: AtomicUsize,
    /// Number of training errors
    pub training_errors: AtomicUsize,
    /// Latest prediction latency in microseconds
    pub latest_prediction_latency_us: AtomicUsize,
    /// Latest training latency in microseconds
    pub latest_training_latency_us: AtomicUsize,
    /// Model version
    pub version: AtomicUsize,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last update timestamp
    pub last_updated_at: AtomicU64,
}

impl ModelStats {
    /// Create new model statistics
    pub fn new() -> Self {
        Self {
            prediction_count: AtomicUsize::new(0),
            training_count: AtomicUsize::new(0),
            prediction_errors: AtomicUsize::new(0),
            training_errors: AtomicUsize::new(0),
            latest_prediction_latency_us: AtomicUsize::new(0),
            latest_training_latency_us: AtomicUsize::new(0),
            version: AtomicUsize::new(1),
            created_at: SystemTime::now(),
            last_updated_at: AtomicU64::new(
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            ),
        }
    }
    
    /// Update the last updated timestamp
    pub fn update_timestamp(&self) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.last_updated_at.store(now, Ordering::SeqCst);
    }
    
    /// Get formatted statistics as a string
    pub fn format_stats(&self) -> String {
        format!(
            "Model v{} | Predictions: {} | Training runs: {} | Errors: {}/{} | Latency: {}μs/{}μs",
            self.version.load(Ordering::Relaxed),
            self.prediction_count.load(Ordering::Relaxed),
            self.training_count.load(Ordering::Relaxed),
            self.prediction_errors.load(Ordering::Relaxed),
            self.training_errors.load(Ordering::Relaxed),
            self.latest_prediction_latency_us.load(Ordering::Relaxed),
            self.latest_training_latency_us.load(Ordering::Relaxed),
        )
    }
    
    /// Reset error counters
    pub fn reset_error_counters(&self) {
        self.prediction_errors.store(0, Ordering::SeqCst);
        self.training_errors.store(0, Ordering::SeqCst);
    }
    
    /// Get uptime in seconds
    pub fn uptime_secs(&self) -> u64 {
        SystemTime::now()
            .duration_since(self.created_at)
            .unwrap_or_default()
            .as_secs()
    }
    
    /// Get time since last update in seconds
    pub fn time_since_update_secs(&self) -> u64 {
        let last_update = self.last_updated_at.load(Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(last_update)
    }
}

impl Default for ModelStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate KL divergence between two distributions
/// Note: both arguments should be normalized probability distributions
pub fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    if p.len() != q.len() {
        return f32::INFINITY;
    }
    
    p.iter()
        .zip(q.iter())
        .filter(|&(&p_i, &q_i)| p_i > 0.0 && q_i > 0.0)
        .map(|(&p_i, &q_i)| p_i * (p_i / q_i).ln())
        .sum()
}

/// Convert raw model outputs to probability distributions using softmax
pub fn to_probabilities(values: &[f32]) -> Vec<f32> {
    // Find max value for numerical stability
    let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    // Calculate exp(x - max) for each value
    let exp_values: Vec<f32> = values
        .iter()
        .map(|&x| (x - max_val).exp())
        .collect();
    
    // Calculate sum for normalization
    let sum: f32 = exp_values.iter().sum();
    
    // Normalize to get probabilities
    exp_values.into_iter().map(|v| v / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_stats_basics() {
        let stats = ModelStats::new();
        
        assert_eq!(stats.prediction_count.load(Ordering::Relaxed), 0);
        assert_eq!(stats.version.load(Ordering::Relaxed), 1);
        
        stats.prediction_count.fetch_add(10, Ordering::SeqCst);
        assert_eq!(stats.prediction_count.load(Ordering::Relaxed), 10);
        
        stats.version.fetch_add(1, Ordering::SeqCst);
        assert_eq!(stats.version.load(Ordering::Relaxed), 2);
    }
    
    #[test]
    fn test_kl_divergence() {
        // Two identical distributions should have KL divergence of 0
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        assert!((kl_divergence(&p, &q) - 0.0).abs() < 1e-6);
        
        // Test with slightly different distributions
        let p = vec![0.3, 0.2, 0.2, 0.3];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        let kl = kl_divergence(&p, &q);
        assert!(kl > 0.0);
        assert!(kl < 0.1); // Small difference
        
        // Test with very different distributions
        let p = vec![0.9, 0.1, 0.0, 0.0];
        let q = vec![0.1, 0.1, 0.4, 0.4];
        let kl = kl_divergence(&p, &q);
        assert!(kl > 1.0); // Large difference
    }
    
    #[test]
    fn test_to_probabilities() {
        // Test conversion of raw values to probabilities
        let raw = vec![1.0, 2.0, 3.0, 4.0];
        let probs = to_probabilities(&raw);
        
        // Sum should be close to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Values should be in ascending order
        for i in 1..probs.len() {
            assert!(probs[i] > probs[i-1]);
        }
        
        // Highest raw value should have highest probability
        assert_eq!(probs.iter().position(|&p| p == *probs.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()), Some(3));
    }
}