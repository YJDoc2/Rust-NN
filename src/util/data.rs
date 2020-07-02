
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
    (Array2<f32>, Array2<f32>),
    (Array2<f32>, Array1<i64>),
    (Array2<f32>, Array1<i64>),
) {
    let train_data = load_train_data();
    let test_data = load_train_data();
    let val_data = load_val_data();

    let tr_img_reshape = train_data.0.reversed_axes();
    let tr_res = vectorise_results(&train_data.1);
    let val_img_reshape = val_data.0.reversed_axes();
    let test_img_reshape = test_data.0.reversed_axes();
    return (
        (tr_img_reshape, tr_res),
        (val_img_reshape, val_data.1),
        (test_img_reshape, test_data.1),
    );
}

fn vectorise_results(res: &Array1<i64>) -> Array2<f32> {
    let mut ret = Array2::<f32>::zeros([10, res.shape()[0]]);
    Zip::from(ret.gencolumns_mut())
        .and(res)
        .apply(|mut col, &idx| col[idx as usize] = 1.0);
    ret
}

#[test]
fn test_vec_res() {
    let arr: Array1<i64> = arr1(&[1, 2, 3]);
    let vec_res = vectorise_results(&arr);
    assert_eq!(vec_res.shape(), &[10, 3]);
    assert_eq!(vec_res[[1, 0]], 1.0);
    assert_eq!(vec_res[[2, 1]], 1.0);
    assert_eq!(vec_res[[3, 2]], 1.0);
}
