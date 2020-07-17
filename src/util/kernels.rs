use ndarray::{arr1, Array1};

fn get_vertical() -> Vec<[f32; 4]> {
    vec![
        [-1.0, 1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0, -1.0],
    ]
}
fn get_horizontal() -> Vec<[f32; 4]> {
    vec![
        [-1.0, -1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0],
    ]
}
fn get_diag1() -> Vec<[f32; 4]> {
    vec![
        [-1.0, 0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0, 0.5],
        [0.5, 1.0, 0.5, 0.0],
        [1.0, 0.5, 0.0, -1.0],
    ]
}
fn get_diag2() -> Vec<[f32; 4]> {
    vec![
        [1.0, 0.5, 0.0, -1.0],
        [0.5, 1.0, 0.5, 0.0],
        [0.0, 0.5, 1.0, 0.5],
        [-1.0, 0.0, 0.5, 1.0],
    ]
}
fn get_circle1() -> Vec<[f32; 4]> {
    vec![
        [-1.0, 1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0, -1.0],
    ]
}
fn get_circle2() -> Vec<[f32; 4]> {
    vec![
        [0.5, 1.0, 1.0, 0.5],
        [1.0, 0.5, 0.5, 1.0],
        [1.0, 0.5, 0.5, 1.0],
        [0.5, 1.0, 1.0, 0.5],
    ]
}

pub fn get_kernels() -> Vec<Vec<[f32; 4]>> {
    vec![
        get_vertical(),
        get_horizontal(),
        get_diag1(),
        get_diag2(),
        get_circle1(),
        get_circle2(),
    ]
}

pub fn apply_kernel(matrix: &Array1<f32>, side: usize, kernel: Vec<[f32; 4]>) -> Array1<f32> {
    let mut ret = Array1::zeros(matrix.dim() / 16);
    if matrix.dim() != 784 {
        panic!(
            "input array size must be 784,got {} ...kernel size",
            matrix.dim()
        );
    }
    let mut k = 0;
    for i in (0..side).step_by(4) {
        for j in (0..side).step_by(4) {
            let mut sum = 0.0;
            for l1 in 0..4 {
                for l2 in 0..4 {
                    sum += kernel[l1][l2] * matrix[side * (i + l1) + j + l2];
                }
            }
            ret[k] = sum;
            k += 1;
        }
    }

    ret
}

#[test]
fn test_kernel_apply() {
    let arr = arr1(&[
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    ]);
    let val = apply_kernel(&arr, 4, get_horizontal());
    assert_eq!(val, arr1(&[8.0]));
    let arr = arr1(&[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
    ]);
    let val = apply_kernel(&arr, 8, get_horizontal());
    assert_eq!(val, arr1(&[8.0, 8.0, 8.0, 8.0]));
}
