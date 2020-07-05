pub use ndarray::{arr1, Array1, Array2, Zip};
pub use ndarray_npy::read_npy;

const TEST_IMG_LOC: &str = "./data/test_data_img.npy";
const TEST_RES_LOC: &str = "./data/test_data_res.npy";

const TRAIN_IMG_LOC: &str = "./data/train_data_img.npy";
const TRAIN_RES_LOC: &str = "./data/train_data_res.npy";

const VAL_IMG_LOC: &str = "./data/val_data_img.npy";
const VAL_RES_LOC: &str = "./data/val_data_res.npy";
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
    (Vec<Array1<f32>>, Vec<f32>),
    (Vec<Array1<f32>>, Vec<f32>),
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

    let tr_res = vectorise_results(&train_data.1);

    let val_img_reshape = val_data.0.reversed_axes();
    let mut val_ret: Vec<Array1<f32>> = Vec::with_capacity(val_img_reshape.shape()[1]);
    for ip in val_img_reshape.gencolumns() {
        val_ret.push(ip.to_owned());
    }
    let val_res = split_res(val_data.1.reversed_axes());

    let test_img_reshape = test_data.0.reversed_axes();
    let mut test_ret: Vec<Array1<f32>> = Vec::with_capacity(test_img_reshape.shape()[1]);
    for ip in test_img_reshape.gencolumns() {
        test_ret.push(ip.to_owned());
    }
    let test_res = split_res(test_data.1);

    println!("Data shaping complete...");
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
