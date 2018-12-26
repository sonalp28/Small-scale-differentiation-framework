"""
Provides VariableArray, a sub-class of numpy.ndarray for manipulating arrays of Variables.
VariableArray is essentially a numpy.ndarray with dtype=Variable.
The Variable data type overloads most Python arithmetic operators.
Arithmetic on Numpy.ndarrays usually invokes those operators on respective array elements.
So most numpy operations work on VariableArrays as expected.
For example, if x and y are 1D VariableArrays with the same shape,
(x+y)[i] = x[i] + y[i].

In addition, VariableArrays support several custom functions not available for numpy.ndarrays.
An ndarray of float values can be assigned to an VariableArray of independent Variables.
The gradient of a single Variable with respect to an entire VariableArray can also be computed.

A VariableArray V can be initialized from a numpy.ndarray A of values with
V = VariableArray(A).
"""

import numpy as np
from variable import Variable, promote

class VariableArray(np.ndarray):

    def __new__(cls, value_array):
        """
        Constructs a new VariableArray from a numpy.ndarray of values.
        """
        variable_array = np.empty(value_array.shape, dtype=object)
        for a in range(value_array.size):
            variable_array.flat[a] = promote(value_array.flat[a])
        return variable_array.view(cls)

    def __array_wrap__(self, out_arr, context=None):
        """
        Handles numpy operations like sum() that produce a dimensionless VariableArray.
        Converts the dimensionless array to a single "scalar" Variable object.
        """
        if out_arr.ndim:
            return np.ndarray.__array_wrap__(self, out_arr, context)

    def assign(self, value_array):
        """
        Assigns each float value in a numpy.ndarray to the corresponding Variable in self.
        Raises an error if value_array has a different shape than self.
        """
        if self.shape != value_array.shape:
            raise(Exception("Assigning values of different shape"))
        for a in range(value_array.size):
            self.flat[a].assign(value_array.flat[a])

    def evaluate(self):
        """
        Returns the current Variable values in self as a numpy.ndarray of floats.
        """
        value_array = np.empty(self.shape)
        for a in range(self.size):
            value_array.flat[a] = self.flat[a].value
        return value_array

    def gradient(self, v):
        """
        Evaluates the gradient of v with respect to self.
        Returns the gradient as a numpy.ndarray of floats with the same shape as self.
        For example, if self is a 1D VariableArray,
        self.gradient(v)[i] = self[i].gradient(v).
        """
        gradient_array = np.empty(self.shape)
        for a in range(self.size):
            gradient_array.flat[a] = v.derivative(self.flat[a])
        return gradient_array

if __name__ == "__main__":

    """
    Scratch pad for informal testing.
    You can edit the following without affecting the tests.
    """

    x = VariableArray(np.arange(4).reshape((2,2)))
    print("*****",x.size)
    y = x.sum()
    print(x.gradient(y))

    a = VariableArray(np.arange(4).reshape((2,2)))
    b = VariableArray(np.ones((2,2)))
    print("a,b")
    print(a)
    print(b)

    b *= 2
    print("new b")
    print(b)

    x = a + b
    y = a * b
    z = y.sum()
    print("x,y,z")
    print(x)
    print(y)
    print(z)

    print("dy[0,0]/da")
    print(a.gradient(y[0,0]))

    print("dz/db")
    print(b.gradient(z))

    a.assign(np.ones((2,2)) * 3)
    # Need to re-construct dependent variables after assignment
    y = a * b
    z = y.sum()
    print("new dz/db")
    print(b.gradient(z))
