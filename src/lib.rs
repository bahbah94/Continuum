//! Continuum: Zero-downtime ML Model Training and Serving
//!
//! This library provides a framework for training machine learning models
//! while serving predictions, with atomic model swapping for zero-downtime updates.

pub mod traits;
pub mod models;
pub mod server;


// Re-export key types for ergonomic use
pub use traits::features::FeatureVector;
pub use traits::model::{Model, ModelError};
pub use models::linears::LinearRegression;
pub use models::ridge::RidgeRegression;
pub use server::metrics::ModelStats;
pub use server::model_server::AtomicModel;
pub use server::continuous_learning::ContinuousLearningConfig;

// Re-export API structures for ease of use
pub use server::api::{
    ModelParameters,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfo,
    ApiError,
    ApiResult,
    ContinuumApi,
};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_library_exports() {
        // Verify that key types are exported and accessible
        let _feature = FeatureVector::new(vec![1.0, 2.0, 3.0]);
        let _config = ContinuousLearningConfig::default();
        let _params = ModelParameters::default();
    }
}
