//! Dense row-major float32 and int64 tensors for the reference interpreter.

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntTensor {
    pub shape: Vec<usize>,
    pub data: Vec<i64>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "shape {shape:?} does not match {} elements", data.len());
        Tensor { shape, data }
    }

    pub fn zeros(shape: Vec<usize>) -> Tensor {
        let n = shape.iter().product();
        Tensor { shape, data: vec![0.0; n] }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn last(&self) -> usize {
        *self.shape.last().unwrap_or(&1)
    }

    /// (outer, n, inner) of the tensor around dimension `dim`.
    pub fn split_dims(&self, dim: usize) -> (usize, usize, usize) {
        (self.shape[..dim].iter().product(), self.shape[dim], self.shape[dim + 1..].iter().product())
    }
}
