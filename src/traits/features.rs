use ndarray::Array1;

#[derive(Clone, Debug)]
pub struct FeatureVector{
    values: Array1<f32>,
}

impl FeatureVector {
    // Create new feature vector
    pub fn new(values: Vec<f32>) -> Self {
        Self {values: Array1::from(values)}
    }
     // get number of dimension in my case just length as 1-D array
    pub fn dimension(&self) -> usize {
        self.values.len()
    }

    pub fn as_array(&self) -> &Array1<f32>{
        &self.values
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_vector(){
        let vec = FeatureVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(vec.dimension(), 3);
        assert_eq!(vec.as_array()[0], 1.0);
        assert_eq!(vec.as_array()[1], 2.0);
        assert_eq!(vec.as_array()[2], 3.0);
    }
}