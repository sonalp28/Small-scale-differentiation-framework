"""
Provides a Variable class that supports automatic differentiation.
Variables can be combined with arithmetic operators to produce new variables.
For example, if x and y are variables, x + y returns a new variable representing their sum.
The new variable is a "dependent" variable because it is a function of x and y.
x and y are called the "operands" of the new variable.
Variables can be automatically differentiated with respect to each other.
Independent variables can also be assigned new values.
This can change the derivative values of dependent variables.

This class overloads the Python arithmetic operators.
For example the Python expression x + y is implemented by x.__add__(y).
One of x or y can be an int or float, in which case it is promoted to a Variable object.
"""
import math
from chain import *

def promote(operand):
    """
    Helper function that converts int/float operands into Variables.
    """
    if not isinstance(operand, Variable):
        operand = Variable(value=operand)
    return operand

class Variable(object):

    def __init__(self, value, d_op=None, operands=None):
        """
        Initialize a new variable self with a given value.
        If operand and d_op are None, self is an independent variable.
        Otherwise, self depends on the operands and their dependencies.
        d_op is a function handle for automatic differentiation.
        d_op(operands, v) differentiates self with respect to v.
        It does this by recursively applying the chain rule to the operands.
        If operands is not None, it should be a list of Variable objects.
        """
        self.value = value
        self.d_op = d_op
        self.operands = operands
    def __str__(self):
        """
        Produces a string representation of self.
        """
        return "<var = %s>" % self.value
    def __repr__(self):
        """
        Produces a string representation of self.
        """
        return str(self)
    def tree_string(self, depth=0):
        """
        Produces a string representation of self's dependency tree.
        """
        s = " "*depth + "%s" % self.value
        if self.operands is not None:
            s += " = %s:" % self.d_op
            for operand in self.operands:
                s += "\n" + operand.tree_string(depth = depth+1)
        return s

    def evaluate(self):
        """
        Returns the current value assigned to self.
        """
        return self.value

    def assign(self, value):
        """
        Assigns a new value to self.
        Raises an error if self is a dependent variable.
        May invalidate other Variables dependent on self;
        any such Variables should be constructed again.
        """
        if self.operands is not None:
            raise(Exception("Cannot assign to dependent variable"))
        self.value = value

    def derivative(self, v):
        """
        Evaluate the derivative of self with respect to Variable v.
        The derivative is evaluated at the current value of v.
        Returns the derivative as a float, not another Variable.
        If self is the same variable as v, the derivative is 1.
        Else if self is an independent variable, the derivative is 0.
        Otherwise, the derivative is computed recursively,
        by calling self.d_op on self.operands and v.
        Returns the value of the derivative as a float.
        """
        if self == v:
            if self.d_op == d_neg:
                return -1
            else:
                return 1
        elif self.d_op == None and self.operands == None:
            return 0
        elif self.operands is not None:
            return self.d_op(self.operands,v)
        #raise(NotImplementedError)

    def gradient(self, other):
        """
        Evaluate the derivative of other with respect to self.
        """
        return other.derivative(self)

    def __neg__(self):
        """
        Returns a new variable that represents the negative of self.
        The value is the negative of self's value.
        The d_op of the new variable is chain.d_neg
        The operands of the new variable is a singleton list [self].
        """
        return Variable(
            value = -self.value,
            d_op = d_neg,
            operands = [self])
    def __add__(self, other):
        """
        Returns a new variable that represents self + other.
        Promotes other in case it is an int or float.
        The value is the sum of self's and other's values.
        The d_op of the new variable is chain.d_add
        The operands of the new variable is the list [self, other].
        """
        return Variable(
            value = self.value + promote(other).value,
            d_op = d_add,
            operands = [self, promote(other)])
    def __radd__(self, other):
        """
        Returns a new variable that represents other + self.
        Promotes other in case it is an int or float.
        """
        return promote(other).__add__(self)
    def __sub__(self, other):
        """
        Returns a new variable that represents self - other.
        Promotes other in case it is an int or float.
        The value is the difference of self's and other's values.
        The d_op of the new variable is chain.d_sub
        The operands of the new variable is the list [self, other].
        """
        return Variable(
            value = self.value - promote(other).value,
            d_op = d_sub,
            operands = [self, promote(other)])
        #raise(NotImplementedError)
    def __rsub__(self, other):
        """
        Returns a new variable that represents other - self.
        Promotes other in case it is an int or float.
        """
        return promote(other).__sub__(self)
    def __mul__(self, other):
        """
        Returns a new variable that represents self * other.
        Promotes other in case it is an int or float.
        The value is the product of self's and other's values.
        The d_op of the new variable is chain.d_mul
        The operands of the new variable is the list [self, other].
        """
        return Variable(
            value = self.value * promote(other).value,
            d_op = d_mul,
            operands = [self, promote(other)])
    def __rmul__(self, other):
        """
        Returns a new variable that represents other * self.
        Promotes other in case it is an int or float.
        """
        return promote(other).__mul__(self)
    def __div__(self, other):
        """
        Returns a new variable that represents self / other.
        Promotes other in case it is an int or float.
        This uses "true" floating point division, not integer division.
        The value is the quotient of self's and other's values.
        The d_op of the new variable is chain.d_truediv
        The operands of the new variable is the list [self, other].
        """
        return self.__truediv__(other)
    def __rdiv__(self, other):
        """
        Returns a new variable that represents other / self.
        Promotes other in case it is an int or float.
        """
        return promote(other).__div__(self)
    def __truediv__(self, other):
        """
        Returns a new variable that represents self / other.
        Promotes other in case it is an int or float.
        This uses "true" floating point division, not integer division.
        The value is the quotient of self's and other's values.
        The d_op of the new variable is chain.d_truediv
        The operands of the new variable is the list [self, other].
        """
        return Variable(
            value = self.value / promote(other).value,
            d_op = d_truediv,
            operands = [self, promote(other)])
        #raise(NotImplementedError)
    def __rtruediv__(self, other):
        """
        Returns a new variable that represents other / self.
        Promotes other in case it is an int or float.
        """
        return promote(other).__truediv__(self)
    def __pow__(self, other):
        """
        Returns a new variable that represents self ** other.
        Promotes other in case it is an int or float.
        The value is self's value to the power of other's value.
        The d_op of the new variable is chain.d_pow
        The operands of the new variable is the list [self, other].
        """
        return Variable(
            value = self.value ** promote(other).value,
            d_op = d_pow,
            operands = [self, promote(other)])
        #raise(NotImplementedError)
    def __rpow__(self, other):
        """
        Returns a new variable that represents other ** self.
        Promotes other in case it is an int or float.
        """
        return promote(other).__pow__(self)
    def tanh(self):
        """
        Returns a new variable that represents tanh(self).
        The value is the hyperbolic tangent of self's value.
        The d_op of the new variable is chain.d_tanh
        The operands of the new variable is the singleton list [self].
        """
        return Variable(
            value = math.tanh(self.value),
            d_op = d_tanh,
            operands = [self])

if __name__ == "__main__":

    """
    Scratch pad for informal testing.
    You can edit the following without affecting the tests.
    """

    '''x = Variable(2)
    y = Variable(5)
    z = x * y

    w = 6 - 2*z
    v = x + 2
    print(x, " " , y, " ", z, " ", v, " ", w)
    print("v dependency tree:")
    print(v.tree_string())

    print("v, dv/dz, dv/dy, dv/dx")
    print(v)'''
    x = Variable(2.)
    y = x
    print((y**2).derivative(x))
    '''print(v.derivative(z))
    print(v.derivative(y))
    print(v.derivative(x))'''
