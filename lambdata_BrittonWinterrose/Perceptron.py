
import numpy as np


class Perceptron (object):

    """Perceptron classifier.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight
        initialization.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of missclassifications (updates) in each epoch (cycle).

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples
        and n_features is the # of features.

        y : array-like, shape = [n_samples]
        Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)  # Sets the seed.
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        self.errors_ = []

        # It loops over and updates weights according to perceptron learning.
        for _ in range(self.n_iter):
            errors = 0
        for xi, target in zip(X, y):
            update = self.eta * (target - self.predict(xi))
            self.w_[1:] += update * xi
            self.w_[0] += update
            errors += int(update != 0.0)
        self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        # np.dot calculates the vector dot product of W T * X.
        return np.dot(X, self.w_[1:]) + self.w_[0]  # self.w_[0] == Xsub0*Wsub0

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

"""
USE
----

I can use this to initialize new 'perceptron' objects with a given
learning rate 'eta', and 'n_iter' number of epochs.

The 'fit' method initializes the weights in 'self.w_' to a vector R m+1, where
m is the number of dimensions (features) in the dataset, and we add 1 as the
first element of this data set, 'w_[0]', as our bias unit.

The vector also contains small random numbers drawn from a normal distribution
with standard deviation 0.01, via the 'rgen.normal' function. We don't
initialize to zero because 'eta', our learning rate, only has an effect on the
classification outcome if the weights are non-zero values. If weights are set
to  zero the learning rate effects the scale of the vector, not the direction.
The choice for normal vs uniform distribution is irrelevant, as the end goal
is just to get small random variables and avoid the all-zero vector properties.

Once initialized, 'fit' loops through all the samples in the training set and
updates the weights according to the perceptron learning rule:
change in weight = (learning rate * (Actual class  - Predicted class ) *sample)

the 'predict' method predicts the class label for that weight update, but after
successfully fitting the model to the test data the 'predict' method can be
used to predict the class labels of new data!!

While running test data we collect the number of misclassifications from each
epoch in the 'self.errors_' list so we can see how well it performed.
"""
