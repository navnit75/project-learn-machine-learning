# INTRODUCTION

Deep Learning is the most exciting and powerful branch of Machine Learning. Deep Learning models can be used for a variety of complex tasks:

- Artificial Neural Networks for Regression and Classification
- Convolutional Neural Networks for Computer Vision
- Recurrent Neural Networks for Time Series Analysis
- Self Organizing Maps for Feature Extraction
- Deep Boltzmann Machines for Recommendation Systems
- Auto Encoders for Recommendation Systems

#### NEURON: Are the building blocks of an NEURAL NETWORK

#### SYNAPSE : Are the lines which join a node to another node

- Each synapse has weights
- which gets adjusted while training happens.

- INPUT LAYER: Each node in input layer denote a single attribute of a row in dataset ( Independent variable )
- OUPUT LAYER :
  - Can contain muliple neuron ( if the output has to be categorical)
  - Can contain continuous value
  - Can contain binary value (classification)

## THE ACTIVATION FUNCTION

- A neuron is trained by two step

  1. It sums xi\*wi
  2. Activation function is applied on the sum

  - Various activation function includes:
    - Threshold function
    - Sigmoid function
    - Rectifier
    - Hyperbolic tanh

## WORKING OF NEURAL NETWORK :

- Every hidden layer tries to establish relation between the attributes provided.
- The more the layers the more the information extracted from the data taken from the dataset .
- A neuron fires up only when a certain criteria is met.
- So a neuron will try to find out the relation between the two attribute, and will change the weights according to that.
- Meanwhile another neuron in the same layer , will try to find the relation between 3 attributes.

## LEARNINGs FROM NEURAL NETWORK:

- Consitsts of neurons, with input layer , hidden layer and output layer.
- After giving input value, a output value will be generated.
- It is compared to the actual value
- We are going to be calculating a function called cost function. ( 1/2 (output_value - actual_value)^ 2 )
- It describes the error in the output obtained.
- Our goal is to MINIMIZE the output of the cost function.
- This error goes again into the neural network and the weights gets updated.

### Lets take an example we have 8 rows in dataset

- 1 epoch
- The model will take all the rows one by one and caluclate the output
- The cost function will calculate the cost
- according to the cost or error experienced the weights would be updated.
- Weight would be same for all the rows.
- 2 epoch
- Same thing will get repeated.
- Until an optimal point is reached where the output value simulates the actual value

## GRADIENT DESCENT:

- The cost function is quadratic function
- if we use brute force , there are calculations in terms of 10^75.
- With best supercomputer it will take around 3 x 10^50 years. ( Also called curse of dimensionality )
- So to minmize the cost function , we will look for the any arbitary point and check the slope of that point, if its either positive or negative , we will
  such that the slope is nearest to 0.
- That way it minimizes the number of calculations needed for minimizing the cost function.

## STOCHASTIC GRADIENT DESCENT:

- One of the issue is gradient descent works when the chosen cost function denotes a CONVEX curve in the graph.
- So what we do in here , earlier after reading whole dataset we were adjusting the weights.
- But in here, after every row given as input the weights of the neural network are adjusted.
- These rows are selected at random , which makes it very non - deterministic.
- Gradient descent is called Deterministic.
- There is also a model known as MINI BATCH GRADIENT DESCENT.
- Uses best of both , creates small batches of rows , give it to the model , then update the weights.

## BACKPROPAGATION:

- The algorithm for backprogagation is complex. The weights of all the synapse in the neural network are updated at oncce.
- Unlike the sequential or individual updation.

## DATASET DESCRIPTION:

- We would be working on the Bank data set which would be using the data to indicate if the custormer is going to leave the bank or not.
- Meanwhile we can also predict if the customer is reliable or not.
- We can also predict, we can predict if a transaction is fradulent or not.
