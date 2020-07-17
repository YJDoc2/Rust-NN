# Recognize handwritten digits Neural Net-Using Rust

## About

This project is contributed by : Yashodhan Joshi , Vatsal Soni, Yatharth Vyas & Tejas Ghone.
Neural networks currently provide the best solutions to many problems in image recognition, speech recognition, and natural language processing etc. Here is an example of our effort of
implementing the traditional Neural Network of Recognizing handwritten digits in Rust Programming Language.
You can read more about it from [here](http://neuralnetworksanddeeplearning.com/index.html)

This is also used with WASM to create a web digit recognizer , check that repository [here](https://github.com/YJDoc2/Rust-NN-Web).

## Why this ?

Initially this started as a way of learning about NN and Rust as well, then we tried to test the speed of this against the python implementation of the same, and finally we used this to implement web digit predictor using Rust and WASM which predicts digits completely in frontend, see the github repository [here](https://github.com/YJDoc2/Rust-NN-Web).
As far as the speed go, we tested this and the python code from the website form which we implemented this, updating it for python3. Without we doing any manual optimizations, and using the optimizations that are by default in the libraries only in both Rust as well as Python code, we saw that the Production build of Rust would Run in 1 minute and some time for 30 epochs where as Python3 would take 6 minutes and some time for the same. ONe might also check this [blog](https://ngoldbaum.github.io/posts/python-vs-rust-nn/) where the person didn't find much difference between both code speed, but when we used code from his repository and compiled production build, that ran faster than ours , and finished almost withing 1 minute, which we think is because of using OpenBLAS in the Ndarray crate as well as using matrix for input instead of vectors optimization, which we haven't used in ours.

That said, In our opinion it may not be very practical to use Rust for direct interface of NN libraries, as a very well coded, tested, and widely used ones are available in Python. Instead maybe it would be very good option, seeing the promises Rust makes and given that it can work very well with other languages through bindings, to use it for the implementation of libraries that are used by Python code to gain speed, along with C or C++.

## About the Neural Net

We have implemented two versions of the Neural Net whose code you can view in files c1 and c2 of the NN folder. We have used the MNIST data set for training and testing of our model.
The Net has about 784 input + 10 output + 30 hidden which equals 824 neurons.

<h4>Version 1 /src/NN/c1</h4>
As we have tried to code in Rust we have used the inbuilt Rust libraries which are provided by cargo for corresponding handling of data and its manipulation.
We have implemented the stochastic gradient descent method for making our model learn along with back propagation in order to get the optimum weights and biases required for predicting the given input digit.
We reduce the quadratic cost function to gain higher accuracy and the model trains accordingly. This trained network gives us a classification rate of about 95 percent - 95.42 percent at its peak .
Code for this version can be found in the file C1 of NN folder.
You can read about it from [here](http://neuralnetworksanddeeplearning.com/chap1.html)

<h4>Version 2.0 /src/NN/c2</h4>
In the first version we minimised the quadratic cost function but the problem with it is that as the sigmoid function gets smaller and smaller as it approaches 0 and thus the change in cost function wrt weights and biases reduces which in turn reduces the learning of our model. To improve the learning we use the Cross-Entropy-Cost function.It tells us that the rate at which the weight learns is controlled by by the error in the output.The larger the error, the faster the neuron will learn.Hence our  model learns faster in this case. Code for this version can be found in the file C2 of NN folder.
This trained network gives us a classification rate of about 96-97 percent at its peak (ideal data)
You can read about it from [here](http://neuralnetworksanddeeplearning.com/chap3.html)

<h4>Version 2.1 /src/NN/c2_mod</h4>
This is Version 2 with some additional updates such as saving the model and calculating the confidence with which the model predicts the given input. The confidence is calculated by
subtracting the mean of the weights of the non-desiable output neurons from the weight of desirable output neuron.
Code for the same can be found in c2.mod file of NN folder

<h4>Now our model works almost perfectly for ideal data which is basically centralised digit input but fails if the data is randomised . To overcome this we have added functions for adding randomization to the training data which improves its predicitbility for randomised data as well</h4>

<h4>/src/NN/c2_conv</h4>
This is a try at implementing convolutional network in rust, but not fully convolutional but partly. This applies a 4 x 4 kernel on disjoint groups of pixels, instead of applying kernel shifting by 1 pixel. this creates the initial inputs for the Fully connected NN which is same as c2_mod. We did not find any significant increase in accuracy, in fact some times it showed decreased accuracy, and performed worse in the web implementation of this for recognizing real world input.

<h4>/src/NN/c2_mod2</h4>
This is almost exactly same as c2_mod, except saving is done on accuracy of training data instead of test/evaluation data. This 
was implemented as we were trying to use all MNIST data for training instead of train-test split, in hope of increasing accuracy. This did increase accuracy in the training than c2_mod, but did not show any significant difference in real world input. THis might be due to overfitting of model.
