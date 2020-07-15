# Recognize handwritten digits Neural Net-Using Rust

## About

This project is contributed by : Yashodhan Joshi , Vatsal Soni, Yatharth Vyas & Tejas Ghone.
Neural networks currently provide the best solutions to many problems in image recognition, speech recognition, and natural language processing etc. Here is an example of our effort of 
implementing the traditional Neural Network of Recognizing handwritten digits in Rust Programming Language.
You can read more about it from [here](http://neuralnetworksanddeeplearning.com/index.html)

## About the Neural Net

We have implemented two versions of the Neural Net whose code you can view in files c1 and c2 of the NN folder. We have used the MNIST data set for training and testing of our model.
The Net has about 784 input + 10 output + 30 hidden which equals 824 neurons.
<h4>Version 1</h4>
As we have tried to code in Rust we have used the inbuilt Rust libraries which are provided by cargo for corresponding handling of data and its manipulation.
We have implemented the stochastic gradient descent method for making our model learn along with back propagation in order to get the optimum weights and biases required for predicting the given input digit.
We reduce the quadratic cost function to gain higher accuracy and the model trains accordingly. This trained network gives us a classification rate of about 95 percent - 95.42 percent at its peak .
Code for this version can be found in the file C1 of NN folder.
You can read about it from [here](http://neuralnetworksanddeeplearning.com/chap1.html)
<h4>Version 2.0</h4>
In the first version we minimised the quadratic cost function but the problem with it is that as the sigmoid function gets smaller and smaller as it approaches 0 and thus the change in cost function wrt weights and biases reduces which in turn reduces the learning of our model. To improve the learning we use the Cross-Entropy-Cost function.It tells us that the rate at which the weight learns is controlled by by the error in the output.The larger the error, the faster the neuron will learn.Hence our  model learns faster in this case. Code for this version can be found in the file C2 of NN folder.
This trained network gives us a classification rate of about 96-97 percent at its peak (ideal data)
You can read about it from [here](http://neuralnetworksanddeeplearning.com/chap3.html)
<h4>Version 2.1</h4>
This is Version 2 with some additional updates such as saving the model and calculating the confidence with which the model predicts the given input. The confidence is calculated by
subtracting the mean of the weights of the non-desiable output neurons from the weight of desirable output neuron.
Code for the same can be found in c2.mod file of NN folder
<h4>Now our model works almost perfectly for ideal data which is basically centralised digit input but fails if the data is randomised . To overcome this we have added functions for adding randomization to the training data which improves its predicitbility for randomised data as well</h4>


