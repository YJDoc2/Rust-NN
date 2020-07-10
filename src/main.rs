#![allow(dead_code)]
mod NN;
mod util;
use ndarray::Array1;
use std::fs::File;
use std::io::prelude::*;

fn main() {
    // load all the data
    let ((tr_data, tr_res), (val_data, val_res), (ts_data, test_res)) = util::data_ext::load_data();
    // let mut net = NN::c1::Network::new(vec![784, 30, 10]);
    let mut net = NN::c2_mod::Network::new(
        vec![784, 30, 10],
        NN::c2_mod::Weight_Init::Default,
        util::c2::get_cec(),
    );
    // let s = std::fs::read_to_string("./network.json").unwrap();
    // let net = NN::c2::load_from_string(vec![784, 30, 10], util::c2::get_cec(), s);
    let training_data: Vec<(Array1<f32>, Array1<f32>)> =
        tr_data.into_iter().zip(tr_res.into_iter()).collect();

    let test_data: Vec<(Array1<f32>, f32)> =
        ts_data.into_iter().zip(test_res.into_iter()).collect();

    let val_dt: Vec<(Array1<f32>, f32)> = val_data.into_iter().zip(val_res.into_iter()).collect();
    // println!(
    //     "Accuracy on evaluation data : {} / {} ",
    //     net.eval_accuracy(&val_dt),
    //     val_dt.len()
    // );
    // println!(
    //     "Accuracy on train data : {} / {} ",
    //     net.train_accuracy(&training_data),
    //     training_data.len()
    // );
    net.SGD(
        training_data,
        30,
        10,
        1.0,
        5.0,
        Some(&val_dt),
        true,
        true,
        true,
        true,
        true,
    );
    //println!("saving....");
    //net.save();
}
