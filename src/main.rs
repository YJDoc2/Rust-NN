#![allow(dead_code)]
mod NN;
mod util;
use ndarray::Array1;

fn main() {
    // load all the data
    let ((tr_data, tr_res), (val_data, val_res), (ts_data, test_res)) = util::data_all::load_data();
    let mut net = NN::c2_mod2::Network::new(
        vec![784, 50, 30, 10],
        NN::c2_mod2::Weight_Init::Default,
        util::c2::get_cec(),
    );
    // let s = std::fs::read_to_string("./network.json").unwrap();
    // let mut net = NN::c2_mod::load_from_string(vec![784, 30, 10], util::c2::get_cec(), s);
    let mut training_data: Vec<(Array1<f32>, Array1<f32>)> =
        tr_data.into_iter().zip(tr_res.into_iter()).collect();

    let mut test_data: Vec<(Array1<f32>, Array1<f32>)> =
        ts_data.into_iter().zip(test_res.into_iter()).collect();
    let mut val_dt: Vec<(Array1<f32>, Array1<f32>)> =
        val_data.into_iter().zip(val_res.into_iter()).collect();
    training_data.append(&mut test_data);
    training_data.append(&mut val_dt);

    net.SGD(
        training_data,
        50,
        10,
        0.5,
        5.0,
        None,
        true,
        true,
        true,
        true,
        true,
    );
    //println!("saving....");
    //net.save();
}
