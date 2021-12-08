# INTITUTION

- The main difference with SVM is, SVM is applicable only if the data is linearly separable
- Basically we introduce a higher dimension
- Introduction of higher dimension leads to , arrange data in such a way that the, data can be classified. (In Images)
- Then again , we transform the separating boundary back to the normal state.
- Which gives us a non linear boundary , in the basic dimension.
- Later we can use that to classify the data
- But this operation is very COMPUTE INTENSIVE.
- In a formula double working line means || x - l|| , distance between two points

# NON LINEAR SVR

- We have seen in SVR basically deciding of the curve happens through a Linear Line (of some width) provided to analyze various data points.
- But when we , use it with Gausiaan RBF kernel , there is a chance we can find the curve at a better rate.
- How the step proceeds I have put in images.
- Now the question arises, what about the two boundary margins which we have seen in the SVR.
- They are also made up in the upgraded dimension space and provided to lower dimension space.

# TRAINING SNIPPET

```python
# in sklearn libraries there are two varities available
# one being --> linearSVC , another being ---> SVC
# We can choose SVC and we can mention linear as kernel type
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state = 0)
classifier.fit(X_train,y_train)
```
