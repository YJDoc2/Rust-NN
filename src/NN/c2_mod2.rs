use crate::util::c2::Cost;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
use rand::seq::SliceRandom;
use rand::thread_rng;

#[macro_use]
use serde::{Deserialize, Serialize};

pub enum Weight_Init {
    Default,
    Large,
}

#[derive(Serialize, Deserialize)]
pub struct WB {
    acc: f32,
    weights: Vec<Vec<Vec<f32>>>,
    biases: Vec<Vec<f32>>,
}

pub struct Network {
    acc: f32,
    num_layers: usize,
    sizes: Vec<i32>,
    biases: Vec<Array1<f32>>,
    weights: Vec<Array2<f32>>,
    cost: Cost,
}

impl Network {
    pub fn new(sizes: Vec<i32>, w_init: Weight_Init, cost: Cost) -> Self {
        let (biases, weights) = match w_init {
            Weight_Init::Default => (
                sizes
                    .iter()
                    .skip(1)
                    .map(|y: &i32| Array1::random(*y as usize, StandardNormal))
                    .collect(),
                sizes
                    .iter()
                    .zip(sizes.iter().skip(1))
                    .map(|(x, y): (&i32, &i32)| {
                        (1.0 / (*x as f32).sqrt())
                            * Array2::random((*y as usize, *x as usize), StandardNormal)
                    })
                    .collect(),
            ),
            Weight_Init::Large => (
                sizes
                    .iter()
                    .skip(1)
                    .map(|y: &i32| Array1::random(*y as usize, StandardNormal))
                    .collect(),
                sizes
                    .iter()
                    .zip(sizes.iter().skip(1))
                    .map(|(x, y): (&i32, &i32)| {
                        Array2::random((*y as usize, *x as usize), StandardNormal)
                    })
                    .collect(),
            ),
        };
        Network {
            acc: 0.0,
            num_layers: sizes.len(),
            sizes: sizes,
            biases: biases,
            weights: weights,
            cost: cost,
        }
    }

    fn feedforward(&self, input: &Array1<f32>) -> Array1<f32> {
        let mut a: Array1<f32> = input.clone();
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
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
        lambda: f32,
        eval_data: Option<&Vec<(Array1<f32>, f32)>>,
        auto_save: bool,
        monitor_eval_cost: bool,
        monitor_eval_acc: bool,
        monitor_train_cost: bool,
        monitor_train_acc: bool,
    ) {
        println!("Starting training...");
        println!();
        for j in 0..epochs {
            training_data.shuffle(&mut thread_rng());

            let mini_batches = training_data.chunks(mini_batch_size);
            for mini_batch in mini_batches {
                self.update_mini_batch(mini_batch, eta, lambda, training_data.len());
            }
            println!("Epoch {} completed", j);
            if monitor_train_cost {
                let cost = self.train_cost(&training_data, lambda);
                // triaining_cost.push(cost);
                println!("Cost on training data : {}", cost);
            }
            if monitor_train_acc {
                let accuracy = self.train_accuracy(&training_data);
                let p_acc = accuracy as f32 / training_data.len() as f32;
                if p_acc > self.acc {
                    self.acc = p_acc;
                }
                if auto_save {
                    println!("Accuracy = {} %, saving...", p_acc * 100.0);
                    self.save();
                }
                //training_acc.push(accuracy);
                println!(
                    "Accuracy on training data : {} / {} ",
                    accuracy,
                    training_data.len()
                );
            }
            if monitor_eval_cost && eval_data.is_some() {
                let cost = self.eval_cost(eval_data.unwrap(), lambda);
                // eval_cost.push(cost);
                println!("Cost on evaluation data : {}", cost);
            }
            if monitor_eval_acc && eval_data.is_some() {
                let accuracy = self.eval_accuracy(eval_data.unwrap());
                //eval_acc.push(accuracy);
                println!(
                    "Accuracy on evaluation data : {} / {} ",
                    accuracy,
                    eval_data.unwrap().len()
                );
            }
            println!();
        }
        println!("Completed training...");
    }

    fn update_mini_batch(
        &mut self,
        mini_batch: &[(Array1<f32>, Array1<f32>)],
        eta: f32,
        lambda: f32,
        n: usize,
    ) {
        let mut nabla_b: Vec<Array1<f32>> = self
            .biases
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
            nabla_b = nabla_b
                .iter()
                .zip(del_nab_b.iter())
                .map(|(nb, dnb)| nb + dnb)
                .collect();
            nabla_w = nabla_w
                .iter()
                .zip(del_nab_w.iter())
                .map(|(nw, dnw)| nw + dnw)
                .collect();
        }
        let batch_len = mini_batch.len() as f32;
        self.weights = self
            .weights
            .iter()
            .zip(nabla_w.into_iter())
            .map(|(w, nw)| (1.0 - eta * (lambda / n as f32)) * w.clone() - (eta / batch_len) * nw)
            .collect();
        self.biases = self
            .biases
            .iter()
            .zip(nabla_b.into_iter())
            .map(|(b, nb)| b.clone() - (eta / batch_len) * nb)
            .collect();
    }

    fn backprop(
        &mut self,
        x: &Array1<f32>,
        y: &Array1<f32>,
    ) -> (Vec<Array1<f32>>, Vec<Array2<f32>>) {
        let mut rev_nabla_b: Vec<Array1<f32>> = Vec::with_capacity(self.biases.len());
        let mut rev_nabla_w: Vec<Array2<f32>> = Vec::with_capacity(self.weights.len());

        let mut activation = x.clone();
        let mut activations: Vec<Array1<f32>> = Vec::with_capacity(self.num_layers);
        activations.push(x.clone());
        let mut zs: Vec<Array1<f32>> = Vec::with_capacity(self.num_layers);

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let temp = w.clone().dot(&activation);
            let z = temp + b;
            zs.push(z.clone());
            activation = sigmoid(z);
            activations.push(activation.clone());
        }
        let a_in = activations.pop().unwrap();
        let z_in = zs.pop().unwrap();
        let mut delta = (self.cost.delta)(z_in, a_in, y.clone());

        rev_nabla_b.push(delta.clone());
        let temp = activations.pop().unwrap();
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
            let temp = activations.pop().unwrap().clone().reversed_axes();
            let m = matrix_from_vecs(delta.clone(), temp);
            rev_nabla_w.push(m);
        }

        (
            rev_nabla_b.into_iter().rev().collect(),
            rev_nabla_w.into_iter().rev().collect(),
        )
    }

    pub fn save(&mut self) {
        let wt = self.weights.clone();
        let bs = self.biases.clone();
        let b_save: Vec<Vec<f32>> = bs.into_iter().map(|b| b.to_vec()).collect();
        let w_save: Vec<Vec<Vec<f32>>> = wt
            .into_iter()
            .map(|w| {
                let mut ret = Vec::new();
                for c in w.gencolumns() {
                    ret.push(c.to_vec());
                }
                ret
            })
            .collect();
        let save = WB {
            acc: self.acc,
            weights: w_save,
            biases: b_save,
        };
        let s = serde_json::to_string(&save).unwrap();
        std::fs::write("network.json", s).unwrap();
    }

    fn eval_cost(&self, data: &Vec<(Array1<f32>, f32)>, lambda: f32) -> f32 {
        let mut cost = 0.0;
        let len = data.len() as f32;
        for (x, y) in data.iter() {
            let a = self.feedforward(x);
            let out = vectorize_res(*y as usize);
            cost += (self.cost.cost)(a, out) / len;
        }
        let norm_sum: f32 = self
            .weights
            .iter()
            .map(|w| w.iter().fold(0.0, |sum, val| sum + val * val))
            .sum();
        cost += 0.5 * (lambda / len) * norm_sum;
        cost
    }
    fn train_cost(&self, data: &Vec<(Array1<f32>, Array1<f32>)>, lambda: f32) -> f32 {
        let mut cost = 0.0;
        let len = data.len() as f32;
        for (x, y) in data.iter() {
            let a = self.feedforward(x);
            cost += (self.cost.cost)(a, y.clone()) / len;
        }
        let norm_sum: f32 = self
            .weights
            .iter()
            .map(|w| w.iter().fold(0.0, |sum, val| sum + val * val))
            .sum();
        cost += 0.5 * (lambda / len) * norm_sum;
        cost
    }

    pub fn eval_accuracy(&self, data: &Vec<(Array1<f32>, f32)>) -> i32 {
        let mut ret = Vec::<(f32, f32)>::with_capacity(data.len());
        data.iter().for_each(|(x, y)| {
            let out = max(&self.feedforward(x));
            ret.push((out, *y))
        });

        ret.into_iter().fold(0, |mut sum, (x, y)| {
            if (x - y).abs() < 0.1 {
                sum += 1;
            }
            sum
        })
    }

    pub fn train_accuracy(&self, data: &Vec<(Array1<f32>, Array1<f32>)>) -> i32 {
        let mut ret = Vec::<(f32, f32)>::with_capacity(data.len());
        data.iter().for_each(|(x, y)| {
            let out = max(&self.feedforward(x));
            ret.push((out, max(y)))
        });

        ret.into_iter().fold(0, |mut sum, (x, y)| {
            if (x - y).abs() < 0.1 {
                sum += 1;
            }
            sum
        })
    }
}

fn sigmoid(x: Array1<f32>) -> Array1<f32> {
    x.map(|x: &f32| 1.0 / (1.0 + std::f64::consts::E.powf(-*x as f64)) as f32)
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

fn confidence(out: &Array1<f32>) -> f32 {
    let max = max(out);
    let mut sum = 0.0;
    for i in 0..10 {
        if i != max as usize {
            sum += out[i];
        }
    }
    let avg = sum / 9.0;

    (out[max as usize] - avg) * 100.0
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

fn vectorize_res(idx: usize) -> Array1<f32> {
    let mut ret = Array1::zeros(10);
    if idx > 9 {
        panic!("array value out of bounds : {}", idx);
    }
    for i in 0..10 {
        if i == idx {
            ret[i] = 1.0;
        }
    }
    ret
}

pub fn load_from_string(sizes: Vec<i32>, cost: Cost, s: String) -> Network {
    let wb: WB = serde_json::from_str(&s).unwrap();
    let mut biases = Vec::new();
    let mut weights = Vec::new();
    for b in wb.biases {
        let b_in = Array1::from(b);
        biases.push(b_in);
    }
    for w_vec in wb.weights {
        let mut a: Array2<f32> = Array2::zeros([w_vec[0].len(), w_vec.len()]);
        for i in 0..w_vec[0].len() {
            for j in 0..w_vec.len() {
                a[[i, j]] = w_vec[j][i];
            }
        }
        weights.push(a);
    }
    Network {
        acc: wb.acc,
        num_layers: sizes.len(),
        sizes: sizes,
        weights: weights,
        biases: biases,
        cost: cost,
    }
}
