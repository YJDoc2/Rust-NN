use ndarray::{Array1, Array2};
pub struct Cost {
    pub cost: Box<dyn Fn(Array1<f32>, Array1<f32>) -> f32>,
    pub delta: Box<dyn Fn(Array1<f32>, Array1<f32>, Array1<f32>) -> Array1<f32>>,
}

fn cec_cost(output: Array1<f32>, y: Array1<f32>) -> f32 {
    let log_a = output.mapv(|x: f32| x.ln());
    let log_one_minus_a = output.mapv(|x: f32| (-x).ln_1p()); //we need ln(a-output) but
                                                              //as ln(1+a) is directly available, we're using (-x).ln_1p()
    let temp = -output.clone() * log_a - (1.0 - output) * log_one_minus_a;
    let corrected = temp.mapv_into(|x: f32| {
        if f32::is_nan(x) {
            0.0
        } else if f32::is_infinite(x) {
            std::f32::MAX
        } else {
            x
        }
    });
    corrected.into_iter().fold(0.0, |sum, val| sum + val)
}
//(1.0 - eta * (lambda / n)) *
fn cec_delta(_z: Array1<f32>, output: Array1<f32>, y: Array1<f32>) -> Array1<f32> {
    output - y
}
pub fn get_cec() -> Cost {
    Cost {
        cost: Box::new(cec_cost),
        delta: Box::new(cec_delta),
    }
}

fn q_cost(output: Array1<f32>, y: Array1<f32>) -> f32 {
    let temp = output - y;
    0.5 * (temp.dot(&temp))
}
fn q_delta(z: Array1<f32>, output: Array1<f32>, y: Array1<f32>) -> Array1<f32> {
    (output - y) * sigmoid_prime(z)
}

pub fn get_qcost() -> Cost {
    Cost {
        cost: Box::new(q_cost),
        delta: Box::new(q_delta),
    }
}

fn sigmoid(x: Array1<f32>) -> Array1<f32> {
    x.map(|x: &f32| 1.0 / (1.0 + std::f64::consts::E.powf(*x as f64)) as f32)
}

fn sigmoid_prime(x: Array1<f32>) -> Array1<f32> {
    let temp = Array1::<f32>::ones(x.raw_dim()) - sigmoid(x.clone());
    sigmoid(x) * temp
}
