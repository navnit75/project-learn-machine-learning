# INTRODUCTION

- Usually the understanding of the Deep learning algorithm revolves around brain like structure.
- Input layer then ----> Hidden Layer -----> Output Layer
- In RNN its bit different

```
      output layer
          ^
          |
          |
      Hidden layers---------+
           ^  ^             |
           |  |             |
           |  +-------------+
      Input Layer
```

```
      output layer
          ^
          |
          |
     Hidden layers---------> Hidden Layers-------> Hidden Layers
           ^                       ^                      ^
           |                       |                      |
           |                       |                      |
      Input Layer             Input Layer              Input Layer
            t = 0                 t = 1                     t = 2
```

- So every hidden layer provides the output to its future layer, which generates a Short term memory of the Network.
- So every time a neuron gets trained in hidden it generates one extra output to the future of the Neuron.

## THE VANISHING GRADIENT PROBLEM:

- When the model is trained for imagine 5s , it needs to find the loss , and backpropagate the loss back to model at t = 0 seconds.
- Then the model again trains the neurons.
- But issue with RNN was it was using Same weight to BACK PROPAGATE ( W res ).
- But as the weights propagate back in time, it keeps getting lower, because of the multiplication with itself.
- During a period of time it was seen the neurons at the t = 0s will learn slower and the neurons near the t = 5s will be quicker.
- And as the neurons at t = 0s , we are providing the inputs to the t= 1s neuron, which in turn provides the output -- > next layer.
- It creates a DOMINO effect.
- This is called vanishing gradient problem.
- So now , W rec is SMALL (< 1) ---> Then we have VANISHING GRADIENT.
- So now, W rec is LARGE ( > 1)-----> Then we have EXPLODING GRADIENT problem.

## SOLUTION:

### Exploding Gradient:

- Truncated Backpropagation
- Penalties (Gradient being penalised )
- Gradient Clipping ( Don't let gradient cross a maximum value )

### Vanishing Gradient :

- Weight Intialization ( Smart about intialising the intial weights so that the issue doesn't come )
- Echo State Networks
- Long Short Term Memory Networks ( LSTMs )
