import math
import unittest as ut
import numpy as np
from variable import Variable
from variable_array import VariableArray
from learn import gradient_descent

TOL = 0.0001

class ADTestCase(ut.TestCase):
    def assertRoughlyEqual(self, a, b):
        self.assertTrue(abs(a - b) < TOL)
    def assertArraysRoughlyEqual(self, a, b):
        self.assertTrue(np.allclose(a, b, rtol=0, atol=TOL))

class NegTestCase(ADTestCase):

    def test_0(self):
        x = Variable(0.)
        self.assertRoughlyEqual((-x).derivative(x), -1.)
    def test_1(self):
        x = Variable(2.5)
        self.assertRoughlyEqual((-x).derivative(x), -1.)
    def test_2(self):
        x = Variable(-35.)
        self.assertRoughlyEqual((-x).derivative(x), -1.)

class AddTestCase(ADTestCase):

    def test_0(self):
        x, y = Variable(1.), Variable(2.)
        self.assertRoughlyEqual((x+y).derivative(x), 1.)
        self.assertRoughlyEqual((x+y).derivative(y), 1.)
    def test_1(self):
        x, y = Variable(1.), Variable(-2.)
        self.assertRoughlyEqual((x+y).derivative(x), 1.)
        self.assertRoughlyEqual((x+y).derivative(y), 1.)
    def test_2(self):
        x = Variable(3.)
        self.assertRoughlyEqual((x+x).derivative(x), 2.)

class SubTestCase(ADTestCase):

    def test_0(self):
        x, y = Variable(1.), Variable(2.)
        self.assertRoughlyEqual((x-y).derivative(x), 1.)
        self.assertRoughlyEqual((x-y).derivative(y), -1.)
    def test_1(self):
        x, y = Variable(1.), Variable(-2.)
        self.assertRoughlyEqual((x-y).derivative(x), 1.)
        self.assertRoughlyEqual((x-y).derivative(y), -1.)

class MulTestCase(ADTestCase):

    def test_0(self):
        x, y = Variable(1.), Variable(2.)
        self.assertRoughlyEqual((x*y).derivative(x), 2.)
        self.assertRoughlyEqual((x*y).derivative(y), 1.)
    def test_1(self):
        x, y = Variable(1.), Variable(-2.)
        self.assertRoughlyEqual((x*y).derivative(x), -2.)
        self.assertRoughlyEqual((x*y).derivative(y), 1.)
    def test_2(self):
        x = Variable(3.)
        self.assertRoughlyEqual((x*x).derivative(x), 3. + 3.)

class DivTestCase(ADTestCase):

    def test_0(self):
        x, y = Variable(2.), Variable(4.)
        self.assertRoughlyEqual((x/y).derivative(x), 1./4.)
        self.assertRoughlyEqual((x/y).derivative(y), -2 * 4.**-2.)
    def test_1(self):
        x, y = Variable(2.), Variable(-4.)
        self.assertRoughlyEqual((x/y).derivative(x), -1./4.)
        self.assertRoughlyEqual((x/y).derivative(y), -2. * (-4.) ** -2.)

class PowTestCase(ADTestCase):

    def test_0(self):
        x, y = Variable(2.), Variable(4.)
        self.assertRoughlyEqual((x**y).derivative(x), 4.*(2.**3.))
        self.assertRoughlyEqual((x**y).derivative(y), 2.**4. * math.log(2.))
    def test_1(self):
        x, y = Variable(2.), Variable(-4.)
        self.assertRoughlyEqual((x**y).derivative(x), -4.*(2.**-5.))
        self.assertRoughlyEqual((x**y).derivative(y), 2.**-4. * math.log(2.))

class TanhTestCase(ADTestCase):

    def test_0(self):
        x = Variable(0.)
        self.assertRoughlyEqual(x.tanh().derivative(x), 1.)
    def test_1(self):
        x = Variable(100.)
        self.assertRoughlyEqual(x.tanh().derivative(x), 0.)

class ChainTestCase(ADTestCase):

    def test_0(self):
        x = Variable(2.)
        y = -x
        self.assertRoughlyEqual((y**2).derivative(x), 2. * 2.)

    def test_1(self):
        x, y = Variable(2.), Variable(3.)
        z = 2. * x + 4. * y
        self.assertRoughlyEqual(z.derivative(x), 2.)
        self.assertRoughlyEqual(z.derivative(y), 4.)

    def test_2(self):
        x, y = Variable(2.), Variable(3.)
        z = -x * 2. - 4. * y
        self.assertRoughlyEqual(z.derivative(x), -2.)
        self.assertRoughlyEqual(z.derivative(y), -4.)

class GradientTestCase(ADTestCase):

    def test_0(self):
        x, y = Variable(2.), Variable(3.)
        v = VariableArray(np.array([x,y]))
        self.assertArraysRoughlyEqual(v.gradient(x), np.array([1., 0.]))
        self.assertArraysRoughlyEqual(v.gradient(y), np.array([0., 1.]))

    def test_1(self):
        x, y = Variable(2.), Variable(3.)
        v = VariableArray(np.array([x,y]))
        z = v.sum()
        self.assertArraysRoughlyEqual(v.gradient(z), np.array([1., 1.]))
        self.assertRoughlyEqual(z.derivative(x), 1.)
        self.assertRoughlyEqual(z.derivative(y), 1.)

    def test_2(self):
        x, y = Variable(2.), Variable(3.)
        v = VariableArray(np.array([x,y]))
        z = (2*v).sum()
        self.assertArraysRoughlyEqual(v.gradient(z), np.array([2., 2.]))

    def test_3(self):
        x, y = Variable(2.), Variable(3.)
        v = VariableArray(np.array([x,y]))
        z = (v*v).sum()
        self.assertArraysRoughlyEqual(v.gradient(z), np.array([4., 6.]))

class LearnTestCase(ADTestCase):

    def test_0(self):
        X = np.array([[-0.55427249,  0.40034063, -1.40994713,  0.51925678],
                      [ 0.34043718,  0.02484774,  1.02835799,  0.50503202]])
        Y = np.array([[ 1.13055928, -0.90340322,  1.90165584, -1.09158475],
                      [ 0.29670035, -0.25619711,  0.46959747, -0.33156514]])
        W = [VariableArray(np.array([[-0.50043522, -0.15420026],
                                     [ 0.34272670,  0.24172611]])),
             VariableArray(np.array([[ 1.17618063,  1.2767736 ],
                                     [ 0.96057281, -0.66617526]]))]

        E = VariableArray(np.array([
                5.551991873935021,
                4.025973161047319,
                2.981961622205049,
                2.2746022766135265,
                1.787089056993292,
                1.44064870266597,
                1.1863588179760975,
                0.9942344318783873,
                0.8455163235626736,
                0.7280748922416285]))

        def error_function(parameters):
            return np.sum((parameters[1].dot(np.tanh(parameters[0].dot(X))) - Y)**2)

        errors = gradient_descent(W, error_function, num_iters=10, learning_rate=0.01)

        for e in range(len(errors)):
            self.assertRoughlyEqual(E[e].evaluate(), errors[e].evaluate())

if __name__ == "__main__":

    test_suite = ut.TestLoader().loadTestsFromTestCase(NegTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)

    test_suite = ut.TestLoader().loadTestsFromTestCase(AddTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)

    test_suite = ut.TestLoader().loadTestsFromTestCase(SubTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)

    test_suite = ut.TestLoader().loadTestsFromTestCase(MulTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)

    test_suite = ut.TestLoader().loadTestsFromTestCase(DivTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)

    test_suite = ut.TestLoader().loadTestsFromTestCase(PowTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)

    test_suite = ut.TestLoader().loadTestsFromTestCase(TanhTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)

    test_suite = ut.TestLoader().loadTestsFromTestCase(ChainTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)

    test_suite = ut.TestLoader().loadTestsFromTestCase(GradientTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)

    test_suite = ut.TestLoader().loadTestsFromTestCase(LearnTestCase)
    ut.TextTestRunner(verbosity=2).run(test_suite)
