use ndarray::{Array1, Array2};
use ndarray_rand::{rand_distr::Normal, RandomExt};
struct Network {
    num_layers: usize,
    sizes: Vec<u8>,
    bises: Vec<Array1<f32>>,
    weights: Vec<Array2<f32>>,
}

impl Network {
    fn new(sizes: Vec<u8>) -> Self {
        let bises: Vec<_> = sizes
            .iter()
            .skip(1)
            .map(|y: &u8| Array1::random(*y as usize, Normal::new(0.0, 1.0).unwrap()))
            .collect();
        let weights: Vec<_> = sizes
            .iter()
            .rev()
            .zip(sizes.iter().skip(1))
            .map(|(x, y): (&u8, &u8)| {
                Array2::random((*y as usize, *x as usize), Normal::new(0.0, 1.0).unwrap())
            })
            .collect();

        Network {
            num_layers: sizes.len(),
            sizes: sizes,
            bises: bises,
            weights: weights,
        }
    }

    fn feedforward(&self, input: Array1<f32>) -> Array1<f32> {
        let mut a: Array1<f32> = input;
        for (b, w) in self.bises.iter().zip(self.weights.iter()) {
            let dot = w.dot(&a);
            let sum = dot + b;
            a = sigmoid(sum);
        }
        a
    }
}

fn sigmoid(x: Array1<f32>) -> Array1<f32> {
    x.map(|x: &f32| 1.0 / (1.0 + std::f64::consts::E.powf(x as f64)) as f32)
}
