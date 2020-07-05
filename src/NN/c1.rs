use ndarray::{linalg::general_mat_mul, Array1, Array2};
use ndarray_rand::{rand_distr::Normal, RandomExt};

use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct Network {
    num_layers: usize,
    sizes: Vec<i32>,
    bises: Vec<Array1<f32>>,
    weights: Vec<Array2<f32>>,
}

impl Network {
    pub fn new(sizes: Vec<i32>) -> Self {
        let bises: Vec<_> = sizes
            .iter()
            .skip(1)
            .map(|y: &i32| Array1::random(*y as usize, Normal::new(0.0, 1.0).unwrap()))
            .collect();
        let weights: Vec<_> = sizes
            .iter()
            .zip(sizes.iter().skip(1))
            .map(|(x, y): (&i32, &i32)| {
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

    fn feedforward(&self, input: &Array1<f32>) -> Array1<f32> {
        let mut a: Array1<f32> = input.clone();
        for (b, w) in self.bises.iter().zip(self.weights.iter()) {
            let dot = w.dot(&a);
            let sum = dot + b;
            a = sigmoid(sum);
        }
        a
    }

    pub fn SGD(
        &mut self,
        mut training_data: Vec<(Array1<f32>, Array1<f32>)>,
        epochs: usize,
        mini_batch_size: usize,
        eta: f32,
        test_data: Option<&Vec<(Array1<f32>, f32)>>,
    ) {
        println!("Starting training...");
        for j in 0..epochs {
            training_data.shuffle(&mut thread_rng());

            let mini_batches = training_data.chunks(mini_batch_size);
            for (i, mini_batch) in mini_batches.enumerate() {
                self.update_mini_batch(mini_batch, eta);
            }
            match test_data {
                Some(data) => println!("Epoch {} : {} / {}", j, self.evaluate(data), data.len()),
                None => println!("Epoch {} completed", j),
            }
        }
        println!("Completed training...");
    }

    fn update_mini_batch(&mut self, mini_batch: &[(Array1<f32>, Array1<f32>)], eta: f32) {
        let mut nabla_b: Vec<Array1<f32>> = self
            .bises
            .iter()
            .map(|b: &Array1<f32>| Array1::zeros(b.raw_dim()))
            .collect();
        let mut nabla_w: Vec<Array2<f32>> = self
            .weights
            .iter()
            .map(|w: &Array2<f32>| Array2::zeros(w.raw_dim()))
            .collect();
        for (x, y) in mini_batch {
            let (del_nab_b, del_nab_w) = self.backprop(x, y);
            nabla_b
                .iter_mut()
                .zip(del_nab_b.iter())
                .for_each(|(nb, dnb)| *nb = nb.clone() + dnb.clone());
            nabla_w
                .iter_mut()
                .zip(del_nab_w.iter())
                .for_each(|(nw, dnw)| *nw = nw.clone() + dnw.clone())
        }
        let batch_len = mini_batch.len();
        self.weights
            .iter_mut()
            .zip(nabla_w.iter())
            .for_each(|(w, nw)| *w = w.clone() - (eta / batch_len as f32) * nw);
        self.bises
            .iter_mut()
            .zip(nabla_b.iter())
            .for_each(|(b, nb)| *b = b.clone() - (eta / batch_len as f32) * nb);
    }

    fn evaluate(&self, test_data: &Vec<(Array1<f32>, f32)>) -> i32 {
        let mut ret = Vec::<(f32, f32)>::with_capacity(test_data.len());
        test_data.iter().for_each(|(x, y)| {
            let out = max(&self.feedforward(x));
            ret.push((out, *y))
        });

        ret.into_iter().fold(0, |mut sum, (x, y)| {
            if x - y < 0.001 {
                sum += 1;
            }
            sum
        })
    }

    fn backprop(
        &mut self,
        x: &Array1<f32>,
        y: &Array1<f32>,
    ) -> (Vec<Array1<f32>>, Vec<Array2<f32>>) {
        let mut rev_nabla_b: Vec<Array1<f32>> = Vec::with_capacity(self.bises.len());
        let mut rev_nabla_w: Vec<Array2<f32>> = Vec::with_capacity(self.weights.len());

        let mut activation = x.clone();
        let mut activations: Vec<Array1<f32>> = Vec::with_capacity(self.num_layers);
        activations.push(x.clone());
        let mut zs: Vec<Array1<f32>> = Vec::with_capacity(self.num_layers);

        for (b, w) in self.bises.iter().zip(self.weights.iter()) {
            let temp = w.clone().dot(&activation);
            let z = temp + b;
            zs.push(z.clone());
            activation = sigmoid(z);
            activations.push(activation.clone());
        }
        let a_len = activations.len();

        let a_in = activations.pop().unwrap().clone();
        let z_in = zs.pop().unwrap();
        let mut delta = const_derivative(a_in, y.clone()) * sigmoid_prime(z_in).reversed_axes();

        rev_nabla_b.push(delta.clone());
        let temp = activations.pop().unwrap().clone().reversed_axes();
        rev_nabla_w.push(matrix_from_vecs(delta.clone(), temp));
        let w_len = self.weights.len();

        for l in 2..self.num_layers {
            let z = zs.pop().unwrap();
            let sp = sigmoid_prime(z);
            delta = self.weights[w_len - l + 1]
                .clone()
                .reversed_axes()
                .dot(&delta)
                * sp;
            rev_nabla_b.push(delta.clone());
            let m = matrix_from_vecs(
                delta.clone(),
                activations.pop().unwrap().clone().reversed_axes(),
            );
            rev_nabla_w.push(m);
        }

        (
            rev_nabla_b.into_iter().rev().collect(),
            rev_nabla_w.into_iter().rev().collect(),
        )
    }
}

fn sigmoid(x: Array1<f32>) -> Array1<f32> {
    x.map(|x: &f32| 1.0 / (1.0 + std::f64::consts::E.powf(*x as f64)) as f32)
}

fn const_derivative(output: Array1<f32>, y: Array1<f32>) -> Array1<f32> {
    output - y
}

fn sigmoid_prime(x: Array1<f32>) -> Array1<f32> {
    let temp = Array1::<f32>::ones(x.raw_dim()) - sigmoid(x.clone());
    sigmoid(x) * temp
}

fn matrix_from_vecs(v1: Array1<f32>, v2: Array1<f32>) -> Array2<f32> {
    let v1_shape = v1.shape();
    let v2_shape = v2.shape();
    let mut ret = Array2::<f32>::zeros((v1_shape[0], v2_shape[0]));
    for i in 0..v1_shape[0] {
        for j in 0..v2_shape[0] {
            ret[[i, j]] = v1[i] * v2[j];
        }
    }
    ret
}

fn max(input: &Array1<f32>) -> f32 {
    let mut max = std::f32::MIN;
    let mut ret_idx = 0.0;
    for (idx, i) in input.iter().enumerate() {
        if *i > max {
            max = *i;
            ret_idx = idx as f32;
        }
    }
    ret_idx
}
