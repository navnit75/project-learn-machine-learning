# INTRODUCTION

- Basically deals with the images classification.
- Application varies from handwritting recogniztion to face detection.
- FEATURE DETECTOR :
- Called kernel, filter , is a 3 x 3 matrix can be more
- Convolution operation is signified by x inside a circle.
- Feature matrix compares the image pixels and gives out a resultant matrix
- if a feature matrix is 3 x 3 start with upper left 3 X 3
- We need to sum up all the matching pixel values and create an resultant matrix.
- move the next 3 x 3 by shifting one coloumn to the right
- once the coloumn are complete , we will shift one row down.
- This single stepping and comparison when happens in gap of 1 PIXEL is called to have STRIDE of 1.
- The resultant matrix obtained is called FEATURE MAP
- more the STRIDE smaller the dimension of feature map.
- There could be multiple feature maps for different different features.
- And during the training the CNN will recognize which of the features are important.

## ReLU Layer:

- After applying feature matrix --> The obtained feature map may have some negative values and positive values.
- To keep all the positive values, or to remove all the black spots
- We apply rectifier.
- Mathematically this breaks the linearity

## Max Pooling: ( also called down sampling )

- Features has to be recognized in the image in any orientation. Images may be rotated , squeezed etc. But we want our convolutional network to
  recognize the object.
- So we will create a 2 X 2 matrix, we will start at the left corner of the FEATURE MAP .
- and out of left cornered 2 X 2 matrix we will find the --> Max pixel
- Then we move right, by length of the fixed STRIDE i.e 2 or 1
- So a rotated image and normal image will have the same MAX POOLED FEATURE MAP
- So the application is we are able to preserve the features and moreover account for possible spatial or textural or other kind of distortions.
- In addittion to all that we are reducing the size. 75% and we are reducing the parameters.
- Which helps in avoiding OVERFITTING
- Sub sampling --> Average pooling

## Flattening:

- Every image will be changed into 1d array , all rows cocatenated together and passed to the next ANN.
- Now the output of flattening would be provided to the ANN

## Full Connection:

- this ANN will have fully connected layers , usually ANN have choices of not being full connected .
- But here it needs to be fully connected.
- So that vector of layers after flattening is passed to input layer of the ANN.
- ANN being best in classification , provided addittional hands in the correct classification of object.

### Process followed:

- Input
- Convolution
- Max pooled
- Flattened
- ANN
- Classification
- Backpropagation of errors
- modification of maps as well as feature detectors
- repeat

## Softmax and Cross entropy:

- Question needs to be asked:
- If there are two neurons in the output layer , how come the the result obtained from both neurons adds up to be one.
- Because the way ann works ? How the values result in being 1 as total?
- Usually classic ann doesn 't give total as zero.
- There is an extra function called softmax applied so that the resulted values are sums up to one,
- soft max is also called normalised exponential function in generalisation.
- Squashes the arbitary real values in k dimensions to values between 0 and 1 for k values.
- Usually in ANN to optimize the weights we use mean squared error as cost function.
- Similarly after applying sofmax function at the output layer, cross entropy function is recommended as cost function in CNN its called loss function.
- The basic reason is to cross entropy is , it can recognize even the smallest of changes made the model , even if it 0.000001 (if you jump from millionth to thosand).
- Where as mean squared error doesn't reward model so much. Eventually it will reward but it takes time.
- Meanwhile cross entropy understands even smaller improvement.
- Cross entropy is the preferred method only for classification
- Regression --> MSE
- Google video by softmax output function by Geoffrey Hinton
