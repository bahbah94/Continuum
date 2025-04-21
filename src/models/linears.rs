use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use serde_json;

use crate::traits::features::FeatureVector;
use crate::traits::model::{Model, ModelError};

/// Linear regression model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegression {
    /// Weights including bias term
    weights: Vec<f32>,
    /// Whether to include a bias term
    with_bias: bool,
    /// Learning rate for gradient descent
    learning_rate: f32,
    /// Number of iterations for gradient descent
    max_iterations: usize,
    /// Whether the model has been trained
    trained: bool,
}

impl LinearRegression {
    /// Create a new Linear Regression model
    pub fn new(with_bias: bool, learning_rate: f32, max_iterations: usize) -> Self {
        Self {
            weights: Vec::new(),
            with_bias,
            learning_rate,
            max_iterations,
            trained: false,
        }
    }
    
    /// Create design matrix from feature vectors
    fn create_design_matrix(&self, features: &[FeatureVector]) -> Array2<f32> {
        let n_samples = features.len();
        let n_features = if features.is_empty() {
            0
        } else {
            features[0].dimension()
        };
        
        let mut design_matrix = if self.with_bias {
            Array2::ones((n_samples, n_features + 1))
        } else {
            Array2::zeros((n_samples, n_features))
        };
        
        for (i, feature) in features.iter().enumerate() {
            let feature_array = feature.as_array();
            if self.with_bias {
                // First column is all ones for bias
                for j in 0..n_features {
                    design_matrix[[i, j + 1]] = feature_array[j];
                }
            } else {
                for j in 0..n_features {
                    design_matrix[[i, j]] = feature_array[j];
                }
            }
        }
        
        design_matrix
    }
    
    /// Train using ordinary least squares
    fn fit_ols(&mut self, x: Array2<f32>, y: Array1<f32>) -> Result<(), ModelError> {
        // Calculate X^T * X
        let xt_x = x.t().dot(&x);
        
        // Calculate X^T * y
        let xt_y = x.t().dot(&y);
        
        // Solve (X^T * X) * w = X^T * y
        match xt_x.solve(&xt_y) {
            Ok(weights) => {
                self.weights = weights.to_vec();
                self.trained = true;
                Ok(())
            },
            Err(e) => Err(ModelError::TrainingError(format!("Failed to solve OLS: {}", e))),
        }
    }
    
    /// Train using gradient descent
    fn fit_gradient_descent(&mut self, x: &Array2<f32>, y: &Array1<f32>) -> Result<(), ModelError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        
        // Initialize weights
        let mut weights = Array1::zeros(n_features);
        
        for _ in 0..self.max_iterations {
            // Predictions: X * w
            let predictions = x.dot(&weights);
            
            // Errors: y - predictions
            let errors = y - &predictions;
            
            // Gradient: -2/n * X^T * errors
            let gradient = x.t().dot(&errors) * (-2.0 / n_samples as f32);
            
            // Update weights: w = w - learning_rate * gradient
            weights = &weights - &(self.learning_rate * gradient);
        }
        
        self.weights = weights.to_vec();
        self.trained = true;
        Ok(())
    }
}

impl Model for LinearRegression {
    fn train(&mut self, features: &[FeatureVector], targets: &[f32]) -> Result<(), ModelError> {
        if features.is_empty() || targets.is_empty() {
            return Err(ModelError::TrainingError("Empty training data".to_string()));
        }
        
        if features.len() != targets.len() {
            return Err(ModelError::DimensionMismatch {
                expected: features.len(),
                actual: targets.len(),
                context: "Number of feature vectors doesn't match number of targets".to_string(),
            });
        }
        
        // Create design matrix
        let x = self.create_design_matrix(features);
        let y = Array1::from(targets.to_vec());
        
        // Choose training method based on data size
        if x.ncols() < 1000 && x.nrows() > x.ncols() {
            // Use OLS for smaller problems
            self.fit_ols(x, y)
        } else {
            // Use gradient descent for larger problems or when X^T*X is singular
            self.fit_gradient_descent(&x, &y)
        }
    }
    
    fn predict(&self, feature: &FeatureVector) -> Result<f32, ModelError> {
        if !self.trained {
            return Err(ModelError::PredictionError("Model not trained".to_string()));
        }
        
        let expected_dim = if self.with_bias {
            self.weights.len() - 1
        } else {
            self.weights.len()
        };
        
        if feature.dimension() != expected_dim {
            return Err(ModelError::DimensionMismatch {
                expected: expected_dim,
                actual: feature.dimension(),
                context: "Feature dimension doesn't match model weights".to_string(),
            });
        }
        
        // Compute dot product
        let mut prediction = if self.with_bias {
            self.weights[0] // Bias term
        } else {
            0.0
        };
        
        let feature_array = feature.as_array();
        let offset = if self.with_bias { 1 } else { 0 };
        
        for i in 0..feature.dimension() {
            prediction += feature_array[i] * self.weights[i + offset];
        }
        
        Ok(prediction)
    }
    
    fn export_parameters(&self) -> Result<Vec<f32>, ModelError> {
        Ok(self.weights.clone())
    }
    
    fn import_parameters(&mut self, parameters: Vec<f32>) -> Result<(), ModelError> {
        if parameters.is_empty() {
            return Err(ModelError::InvalidParameter("Empty parameters".to_string()));
        }
        
        self.weights = parameters;
        self.trained = true;
        Ok(())
    }
    
    fn validate(&self, features: &[FeatureVector], targets: &[f32]) -> Result<f32, ModelError> {
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
        
        // Make predictions
        let predictions = self.predict_batch(features)?;
        
        // Calculate MSE
        let mut sum_squared_error = 0.0;
        for i in 0..predictions.len() {
            let error = predictions[i] - targets[i];
            sum_squared_error += error * error;
        }
        
        let mse = sum_squared_error / predictions.len() as f32;
        Ok(mse)
    }
    
    fn save(&self, path: &str) -> Result<(), ModelError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        
        match serde_json::to_writer(writer, self) {
            Ok(_) => Ok(()),
            Err(e) => Err(ModelError::SerializationError(e.to_string())),
        }
    }
    
    fn load(&mut self, path: &str) -> Result<(), ModelError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        match serde_json::from_reader(reader) {
            Ok(model) => {
                *self = model;
                Ok(())
            }
            Err(e) => Err(ModelError::SerializationError(e.to_string())),
        }
    }
    
    fn clone_model(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_linear_regression_train_predict() {
        // Create simple training data: y = 2*x + 3
        let features = vec![
            FeatureVector::new(vec![1.0]),
            FeatureVector::new(vec![2.0]),
            FeatureVector::new(vec![3.0]),
            FeatureVector::new(vec![4.0]),
        ];
        
        let targets = vec![5.0, 7.0, 9.0, 11.0];
        
        // Create and train model
        let mut model = LinearRegression::new(true, 0.01, 1000);
        model.train(&features, &targets).unwrap();
        
        // Verify model weights
        let weights = model.export_parameters().unwrap();
        // Weights should be close to [3.0, 2.0]
        assert!((weights[0] - 3.0).abs() < 0.1, "Bias should be close to 3.0");
        assert!((weights[1] - 2.0).abs() < 0.1, "Coefficient should be close to 2.0");
        
        // Test prediction
        let test_feature = FeatureVector::new(vec![5.0]);
        let prediction = model.predict(&test_feature).unwrap();
        
        // Expected: 2*5 + 3 = 13
        assert!((prediction - 13.0).abs() < 0.1, "Prediction should be close to 13.0");
        
        // Test validation
        let mse = model.validate(&features, &targets).unwrap();
        assert!(mse < 0.01, "MSE should be very small for perfect fit");
    }
    
    #[test]
    fn test_linear_regression_no_bias() {
        // Create simple training data: y = 2*x (no bias)
        let features = vec![
            FeatureVector::new(vec![1.0]),
            FeatureVector::new(vec![2.0]),
            FeatureVector::new(vec![3.0]),
            FeatureVector::new(vec![4.0]),
        ];
        
        let targets = vec![2.0, 4.0, 6.0, 8.0];
        
        // Create and train model without bias
        let mut model = LinearRegression::new(false, 0.01, 1000);
        model.train(&features, &targets).unwrap();
        
        // Verify model weight
        let weights = model.export_parameters().unwrap();
        assert!((weights[0] - 2.0).abs() < 0.1, "Coefficient should be close to 2.0");
        
        // Test prediction
        let test_feature = FeatureVector::new(vec![5.0]);
        let prediction = model.predict(&test_feature).unwrap();
        
        // Expected: 2*5 = 10
        assert!((prediction - 10.0).abs() < 0.1, "Prediction should be close to 10.0");
    }
    
    #[test]
    fn test_linear_regression_multidimensional_input() {
        // Create training data: y = 2*x1 + 3*x2 + 1
        let features = vec![
            FeatureVector::new(vec![1.0, 1.0]),
            FeatureVector::new(vec![2.0, 1.0]),
            FeatureVector::new(vec![1.0, 2.0]),
            FeatureVector::new(vec![2.0, 2.0]),
        ];
        
        let targets = vec![6.0, 8.0, 9.0, 11.0];
        
        // Train model
        let mut model = LinearRegression::new(true, 0.01, 1000);
        model.train(&features, &targets).unwrap();
        
        // Verify weights [1.0, 2.0, 3.0]
        let weights = model.export_parameters().unwrap();
        assert!(weights.len() == 3, "Should have 3 weights (bias + 2 features)");
        assert!((weights[0] - 1.0).abs() < 0.1, "Bias should be close to 1.0");
        assert!((weights[1] - 2.0).abs() < 0.1, "First coefficient should be close to 2.0");
        assert!((weights[2] - 3.0).abs() < 0.1, "Second coefficient should be close to 3.0");
        
        // Test prediction
        let test_feature = FeatureVector::new(vec![3.0, 4.0]);
        let prediction = model.predict(&test_feature).unwrap();
        
        // Expected: 1 + 2*3 + 3*4 = 19
        assert!((prediction - 19.0).abs() < 0.1, "Prediction should be close to 19.0");
    }
}