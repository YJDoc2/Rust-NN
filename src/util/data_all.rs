use super::manipulation;
use super::manipulation::Direction;
pub use ndarray::{arr1, Array1, Array2, Zip};
pub use ndarray_npy::read_npy;
use rand::Rng;

const TEST_IMG_LOC: &str = "./data/test_data_img.npy";
const TEST_RES_LOC: &str = "./data/test_data_res.npy";

const TRAIN_IMG_LOC: &str = "./data/train_data_img.npy";
const TRAIN_RES_LOC: &str = "./data/train_data_res.npy";

const VAL_IMG_LOC: &str = "./data/val_data_img.npy";
const VAL_RES_LOC: &str = "./data/val_data_res.npy";

// constants for shifting and noise addition
const MIN_SHIFT: usize = 3;
const MAX_SHIFT: usize = 5; // note actual max shift is 8-1 = 7

const MIN_NOISE: usize = 50;
const MAX_NOISE: usize = 251; // note actual max noise is 251-1 = 250

fn load_test_data() -> (Array2<f32>, Array1<i64>) {
    let img = read_npy(TEST_IMG_LOC).unwrap();
    let res = read_npy(TEST_RES_LOC).unwrap();
    (img, res)
}

fn load_train_data() -> (Array2<f32>, Array1<i64>) {
    let img = read_npy(TRAIN_IMG_LOC).unwrap();
    let res = read_npy(TRAIN_RES_LOC).unwrap();
    (img, res)
}

fn load_val_data() -> (Array2<f32>, Array1<i64>) {
    let img = read_npy(VAL_IMG_LOC).unwrap();
    let res = read_npy(VAL_RES_LOC).unwrap();
    (img, res)
}

pub fn load_data() -> (
    (Vec<Array1<f32>>, Vec<Array1<f32>>),
    (Vec<Array1<f32>>, Vec<Array1<f32>>),
    (Vec<Array1<f32>>, Vec<Array1<f32>>),
) {
    println!("Starting loading data...");
    let train_data = load_train_data();
    let test_data = load_test_data();
    let val_data = load_val_data();
    println!("Data loading complete...");
    println!("Starting Data shaping...");

    let tr_img_reshape = train_data.0.reversed_axes();
    let mut tr_ret: Vec<Array1<f32>> = Vec::with_capacity(tr_img_reshape.shape()[1]);

    for ip in tr_img_reshape.gencolumns() {
        tr_ret.push(ip.to_owned());
    }

    let mut tr_res = vectorise_results(&train_data.1);

    let val_img_reshape = val_data.0.reversed_axes();
    let mut val_ret: Vec<Array1<f32>> = Vec::with_capacity(val_img_reshape.shape()[1]);
    for ip in val_img_reshape.gencolumns() {
        val_ret.push(ip.to_owned());
    }
    let mut val_res = split_res(val_data.1.reversed_axes());

    let test_img_reshape = test_data.0.reversed_axes();
    let mut test_ret: Vec<Array1<f32>> = Vec::with_capacity(test_img_reshape.shape()[1]);
    for ip in test_img_reshape.gencolumns() {
        test_ret.push(ip.to_owned());
    }
    let mut test_res = split_res(test_data.1);

    println!("Data shaping complete...");

    println!("Starting extending data...");
    // Now extend the data artificially
    let mut temp_img_vec = Vec::with_capacity(50000);
    let mut temp_res_vec = Vec::with_capacity(50000);

    // shift the image around
    for (i, (arr, res)) in tr_ret[0..15000]
        .iter()
        .zip(tr_res[0..15000].iter())
        .enumerate()
    {
        let dir = match i % 4 {
            0 => Direction::Left,
            1 => Direction::Down,
            2 => Direction::Right,
            3 => Direction::Up,
            _ => panic!("Mod is not supposed to be greater than 3..."),
        };
        let shift: usize = rand::thread_rng().gen_range(MIN_SHIFT, MAX_SHIFT); // at least shift 3 row/col or at most shift 7 row/cols
        temp_img_vec.push(manipulation::shift(arr, dir, shift));
        temp_res_vec.push(res.clone());
    }

    // add random noise to image
    for (arr, res) in tr_ret[15000..30000].iter().zip(tr_res[15000..30000].iter()) {
        let noisy: usize = rand::thread_rng().gen_range(MIN_NOISE, MAX_NOISE); // at least add 50 randomized pixels at max add 251
        temp_img_vec.push(manipulation::add_noise(arr, noisy));
        temp_res_vec.push(res.clone());
    }

    // now add shift as well as add noise
    for (i, (arr, res)) in tr_ret[30000..50000]
        .iter()
        .zip(tr_res[30000..50000].iter())
        .enumerate()
    {
        let dir = match i % 4 {
            0 => Direction::Left,
            1 => Direction::Down,
            2 => Direction::Right,
            3 => Direction::Up,
            _ => panic!("Mod is not supposed to be greater than 3..."),
        };
        let shift: usize = rand::thread_rng().gen_range(MIN_SHIFT, MAX_SHIFT); // at least shift 3 row/col or at most shift 7 row/cols
        let temp = manipulation::shift(arr, dir, shift);
        let noisy: usize = rand::thread_rng().gen_range(MIN_NOISE, MAX_NOISE); // at least add 50 randomized pixels at max add 251
        temp_img_vec.push(manipulation::add_noise(&temp, noisy));
        temp_res_vec.push(res.clone());
    }

    tr_ret.append(&mut temp_img_vec);
    tr_res.append(&mut temp_res_vec);

    temp_img_vec = Vec::with_capacity(10000);
    let mut temp_res_vec = Vec::with_capacity(10000);
    for (i, (arr, res)) in val_ret.iter().zip(val_res.iter()).enumerate() {
        let dir = match i % 4 {
            0 => Direction::Left,
            1 => Direction::Down,
            2 => Direction::Right,
            3 => Direction::Up,
            _ => panic!("Mod is not supposed to be greater than 3..."),
        };
        let shift: usize = rand::thread_rng().gen_range(MIN_SHIFT, MAX_SHIFT); // at least shift 3 row/col or at most shift 7 row/cols
        let temp = manipulation::shift(arr, dir, shift);
        let noisy: usize = rand::thread_rng().gen_range(MIN_NOISE, MAX_NOISE); // at least add 50 randomized pixels at max add 251
        temp_img_vec.push(manipulation::add_noise(&temp, noisy));
        temp_res_vec.push(*res);
    }

    val_ret.append(&mut temp_img_vec);
    val_res.append(&mut temp_res_vec);

    temp_img_vec = Vec::with_capacity(10000);
    let mut temp_res_vec = Vec::with_capacity(10000);
    for (i, (arr, res)) in test_ret.iter().zip(test_res.iter()).enumerate() {
        let dir = match i % 4 {
            0 => Direction::Left,
            1 => Direction::Down,
            2 => Direction::Right,
            3 => Direction::Up,
            _ => panic!("Mod is not supposed to be greater than 3..."),
        };
        let shift: usize = rand::thread_rng().gen_range(MIN_SHIFT, MAX_SHIFT); // at least shift 3 row/col or at most shift 7 row/cols
        let temp = manipulation::shift(arr, dir, shift);
        let noisy: usize = rand::thread_rng().gen_range(MIN_NOISE, MAX_NOISE); // at least add 50 randomized pixels at max add 251
        temp_img_vec.push(manipulation::add_noise(&temp, noisy));
        temp_res_vec.push(*res);
    }
    test_ret.append(&mut temp_img_vec);
    test_res.append(&mut temp_res_vec);

    println!("Data extending complete...");

    let val_res: Vec<_> = val_res.iter().map(|x| vectorize_res(*x as usize)).collect();
    let test_res: Vec<_> = test_res
        .iter()
        .map(|x| vectorize_res(*x as usize))
        .collect();

    return ((tr_ret, tr_res), (val_ret, val_res), (test_ret, test_res));
}

fn vectorise_results(res: &Array1<i64>) -> Vec<Array1<f32>> {
    let mut temp = Array2::<f32>::zeros([10, res.shape()[0]]);
    Zip::from(temp.gencolumns_mut())
        .and(res)
        .apply(|mut col, &idx| col[idx as usize] = 1.0);
    let mut ret = Vec::with_capacity(temp.shape()[1]);
    for ip in temp.gencolumns() {
        ret.push(ip.to_owned());
    }
    ret
}

fn split_res(input: Array1<i64>) -> Vec<f32> {
    let mut ret = Vec::with_capacity(input.shape()[0]);
    for col in input.iter() {
        ret.push(*col as f32);
    }
    ret
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
