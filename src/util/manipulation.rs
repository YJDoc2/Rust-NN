
use ndarray::Array1;
use rand::Rng;
pub fn shift(input: &Array1<f32>,direction: i32,offset:usize) -> Array1<f32> 
{
    let mut newInput = Array1::zeros(input.len());
    if(direction==1 || direction==2)//left & right shift
    {

        for i in 0..input.len()
        {
           if(i%28>=(offset))
           {
            if(direction==1)//left shift
            {
                newInput[i-offset]=input[i] as f32;
            }
            else
            {
                newInput[i]=input[i-offset] as f32;//right shift
            }              
           }
        }
    }
    if(direction==3 || direction==4)//up & down shifts
    {
        let x:usize=28*offset;
        for i in 0..784
        {
           if(i>=28*offset)
           {
             if(direction==3)  //up shift
            {
                newInput[i-x]=input[i];
            }
            else
            {
                newInput[i]=input[i-x];//down shift
            }
           }
        }

    }
    newInput
}

pub fn add_noise(input: &Array1<f32>,no_of_noises:usize)->Array1<f32>
{
    let mut modified: Array1<f32> = input.clone();
    let mut range = rand::thread_rng();
    for i in 0..no_of_noises{
        let ind=range.gen_range(0, 784);
        modified[ind]=0.0;
    }
    modified
}