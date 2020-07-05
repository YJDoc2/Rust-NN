#![allow(dead_code)]
mod NN;
mod util;

use ndarray::{Array1, Array2};
use NN::c1::Network;
fn main() {
    // load all the data
    let ((tr_data, tr_res), (val_data, val_res), (ts_data, test_res)) = util::data::load_data();
    let mut net = Network::new(vec![784, 30, 10]);

    let training_data: Vec<(Array1<f32>, Array1<f32>)> =
        tr_data.into_iter().zip(tr_res.into_iter()).collect();

    let test_data: Vec<(Array1<f32>, f32)> =
        ts_data.into_iter().zip(test_res.into_iter()).collect();
    net.SGD(training_data, 30, 10, 3.0, Some(&test_data));
}
