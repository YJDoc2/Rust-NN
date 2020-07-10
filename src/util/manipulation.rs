use ndarray::Array1;
use rand::Rng;
pub enum Direction {
    Up,
    Down,
    Right,
    Left,
}
pub fn shift(input: &Array1<f32>, direction: Direction, offset: usize) -> Array1<f32> {
    let mut newInput = Array1::zeros(input.len());

    match direction {
        Direction::Left => {
            for i in 0..input.len() {
                if i % 28 >= offset {
                    newInput[i - offset] = input[i] as f32;
                }
            }
        }
        Direction::Right => {
            for i in 0..input.len() {
                if i % 28 >= offset {
                    newInput[i] = input[i - offset] as f32;
                }
            }
        }
        Direction::Up => {
            for i in 0..input.len() {
                if i >= 28 * offset {
                    newInput[i - (28 * offset)] = input[i] as f32;
                }
            }
        }
        Direction::Down => {
            for i in 0..input.len() {
                if i >= 28 * offset {
                    newInput[i] = input[i - (28 * offset)] as f32;
                }
            }
        }
    }
    newInput
}

pub fn add_noise(input: &Array1<f32>, no_of_noises: usize) -> Array1<f32> {
    let mut modified: Array1<f32> = input.clone();
    let mut range = rand::thread_rng();
    for _ in 0..no_of_noises {
        let ind = range.gen_range(0, 784);
        modified[ind] = if modified[ind] > 0.5 { 0.0 } else { 1.0 };
    }
    modified
}
