"""
Provides a basic gradient descent algorithm for learning.
"""

def gradient_descent(parameters, error_function, num_iters, learning_rate, verbose=False):
    """
    Uses gradient descent to find parameters that minimize training error.
    Parameters should be a list of Variables and/or VariableArrays.
    error_function should be a function handle for computing error.
    error_function(parameters) should return a single Variable e.
    e is a variable dependent on the parameters representing their error.
    num_iters is the number of gradient descent iterations to perform.
    learning_rate is a fixed learning rate for the gradient descent.
    If verbose is true, prints the current error at each iteration.
    Returns a list errors, where error[i] is e at the start of the i^{th} iteration.
    """
    errors = []
    for i in range(0,num_iters):
        e = error_function(parameters)
        errors.append(e)
        for p in range(0,len(parameters)):
            parameters[p] = parameters[p] - learning_rate * parameters[p].gradient(e)
        if verbose:
            print (i,e)
    return errors
    #raise(NotImplementedError)

if __name__ == "__main__":

    """
    Scratch pad for informal testing.
    You can edit the following without affecting the tests.
    Shows an example using gradient descent to train a linear regression model.
    An example with a neural network model is given in tests.py.
    """

    import numpy as np
    from variable_array import VariableArray

    # Random training data
    X = np.random.randn(2,4)
    Y = np.random.randn(2,4)
    print("X, Y")
    print(X)
    print(Y)

    # Initialize the regression parameters to zero
    # Trying to learn Y ~ W.dot(X)
    W = VariableArray(np.zeros((2,2)))
    print("initial W")
    print(W)
    parameters = [W]

    def error_function(params):
        e = np.sum((params[0].dot(X) - Y)**2) # sum of squared errors
        return e

    # Use gradient descent to find best parameters
    gradient_descent(parameters, error_function, num_iters=10, learning_rate=0.01, verbose=True)

    # W after learning
    print("learned W")
    print(W)
