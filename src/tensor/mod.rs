pub type Tensor = Vec<f32>;

pub fn zeros(len: usize) -> Tensor {
    vec![0.0; len]
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = 0.0;
    for i in 0..n {
        acc += a[i] * b[i];
    }
    acc
}
