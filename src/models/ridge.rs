use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

use crate::traits::features::FeatureVector;
use crate::traits::model::{Model, ModelError};

/// Ridge regression model (Linear regression with L2 regularization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RidgeRegression {
    /// Weights including bias term
    weights: Vec<f32>,
    /// Whether to include a bias term
    with_bias: bool,
    /// Regularization strength (alpha)
    alpha: f32,
    /// Learning rate for gradient descent
    learning_rate: f32,
    /// Number of iterations for gradient descent
    max_iterations: usize,
    /// Whether the model has been trained
    trained: bool,
}

impl RidgeRegression {
    /// Create a new Ridge Regression model
    pub fn new(with_bias: bool, alpha: f32, learning_rate: f32, max_iterations: usize) -> Self {
        Self {
            weights: Vec::new(),
            with_bias,
            alpha,
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
    
    /// Train using closed-form ridge solution
    fn fit_closed_form(&mut self, x: Array2<f32>, y: Array1<f32>) -> Result<(), ModelError> {
        let n_features = x.ncols();
        
        // Calculate X^T * X
        let xt_x = x.t().dot(&x);
        
        // Add regularization: X^T*X + alpha*I
        // Note: If using bias, we typically don't regularize it
        let mut regularized = xt_x.clone();
        
        let offset = if self.with_bias { 1 } else { 0 };
        for i in offset..n_features {
            regularized[[i, i]] += self.alpha;
        }
        
        // Calculate X^T * y
        let xt_y = x.t().dot(&y);
        
        // Solve (X^T*X + alpha*I) * w = X^T*y
        match regularized.solve(&xt_y) {
            Ok(weights) => {
                self.weights = weights.to_vec();
                self.trained = true;
                Ok(())
            },
            Err(e) => Err(ModelError::TrainingError(format!("Failed to solve ridge regression: {}", e))),
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
            
            // L2 penalty gradient (don't regularize bias if present)
            let mut l2_grad = Array1::zeros(n_features);
            let offset = if self.with_bias { 1 } else { 0 };
            for i in offset..n_features {
                l2_grad[i] = self.alpha * weights[i];
            }
            
            // Gradient: -2/n * X^T * errors + alpha * w
            let gradient = x.t().dot(&errors) * (-2.0 / n_samples as f32) + &l2_grad;
            
            // Update weights: w = w - learning_rate * gradient
            weights = &weights - &(self.learning_rate * gradient);
        }
        
        self.weights = weights.to_vec();
        self.trained = true;
        Ok(())
    }
}

impl Model for RidgeRegression {
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
            // Use closed-form solution for smaller problems
            self.fit_closed_form(x, y)
        } else {
            // Use gradient descent for larger problems
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
    fn test_ridge_regression_train_predict() {
        // Create simple training data: y = 2*x + 3 + small noise
        let features = vec![
            FeatureVector::new(vec![1.0]),
            FeatureVector::new(vec![2.0]),
            FeatureVector::new(vec![3.0]),
            FeatureVector::new(vec![4.0]),
            FeatureVector::new(vec![5.0]),
        ];
        
        let targets = vec![5.1, 7.2, 8.9, 10.8, 13.2]; // Small noise added
        
        // Create and train model with mild regularization
        let mut model = RidgeRegression::new(true, 0.1, 0.01, 1000);
        model.train(&features, &targets).unwrap();
        
        // Verify model weights (should be close to [3.0, 2.0] but slightly different due to regularization)
        let weights = model.export_parameters().unwrap();
        assert!(weights.len() == 2, "Should have 2 weights (bias + coefficient)");
        assert!((weights[0] - 3.0).abs() < 0.5, "Bias should be close to 3.0");
        assert!((weights[1] - 2.0).abs() < 0.5, "Coefficient should be close to 2.0");
        
        // Test prediction
        let test_feature = FeatureVector::new(vec![6.0]);
        let prediction = model.predict(&test_feature).unwrap();
        
        // Expected: ~2*6 + 3 = ~15 (slightly different due to regularization)
        assert!((prediction - 15.0).abs() < 1.0, "Prediction should be close to 15.0");
    }
    
    #[test]
    fn test_ridge_vs_linear() {
        // Create training data with outliers to show ridge's advantage
        let features = vec![
            FeatureVector::new(vec![1.0]),
            FeatureVector::new(vec![2.0]),
            FeatureVector::new(vec![3.0]),
            FeatureVector::new(vec![4.0]),
            FeatureVector::new(vec![5.0]),
            FeatureVector::new(vec![6.0]), // outlier
        ];
        
        let targets = vec![2.0, 4.0, 6.0, 8.0, 10.0, 20.0]; // Last value is an outlier
        
        // Train linear regression
        let mut linear_model = super::super::linears::LinearRegression::new(true, 0.01, 1000);
        linear_model.train(&features, &targets).unwrap();
        
        // Train ridge regression with regularization
        let mut ridge_model = RidgeRegression::new(true, 1.0, 0.01, 1000);
        ridge_model.train(&features, &targets).unwrap();
        
        // Get weights from both models
        let linear_weights = linear_model.export_parameters().unwrap();
        let ridge_weights = ridge_model.export_parameters().unwrap();
        
        // Ridge should have smaller coefficient (less influenced by outlier)
        assert!(ridge_weights[1].abs() < linear_weights[1].abs(), 
                "Ridge coefficient should be smaller than linear coefficient");
        
        // Test on new data
        let test_feature = FeatureVector::new(vec![7.0]);
        let linear_pred = linear_model.predict(&test_feature).unwrap();
        let ridge_pred = ridge_model.predict(&test_feature).unwrap();
        
        // Ridge prediction should be closer to the true pattern (y=2x)
        assert!((ridge_pred - 14.0).abs() < (linear_pred - 14.0).abs(),
                "Ridge prediction should be closer to expected value");
    }
    
    #[test]
    fn test_ridge_with_high_regularization() {
        // Create training data
        let features = vec![
            FeatureVector::new(vec![1.0]),
            FeatureVector::new(vec![2.0]),
            FeatureVector::new(vec![3.0]),
            FeatureVector::new(vec![4.0]),
        ];
        
        let targets = vec![2.0, 4.0, 6.0, 8.0]; // y = 2x
        
        // Train with very high regularization
        let mut high_reg_model = RidgeRegression::new(false, 100.0, 0.01, 1000);
        high_reg_model.train(&features, &targets).unwrap();
        
        // Train with low regularization
        let mut low_reg_model = RidgeRegression::new(false, 0.01, 0.01, 1000);
        low_reg_model.train(&features, &targets).unwrap();
        
        // High regularization should push coefficient toward 0
        let high_reg_weights = high_reg_model.export_parameters().unwrap();
        let low_reg_weights = low_reg_model.export_parameters().unwrap();
        
        assert!(high_reg_weights[0].abs() < low_reg_weights[0].abs(),
                "High regularization should result in smaller weights");
    }
}