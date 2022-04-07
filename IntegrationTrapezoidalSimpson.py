''' # This code establishes how to approximate the integral of a function using either the
    # trapezoidal or Simpson's method.
    #
    # @author Rishab Parthasarathy
    # @version 04.05.2022
    #'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy

''' # Method func1 defines a quadratic function of the form 
    #       f(x) = 1 + x^2.
    #
    # This method also processes a matrix of inputs, allowing
    # multiple samples to be performed simultaneously.
    # 
    # @param inp    the matrix of input values to test
    # @param type   whether to use numpy ('np') or sympy ('sp')
    # @return       the matrix w/ values of f(x) for all sets of inputs
    # '''
def func1(inp, type = 'np'):
    return 1 + inp**2
    # End of method func1

''' # Method func2 defines a quadratic function of the form 
    #       f(x) = x * e^(-x^2)
    #
    # This method also processes a matrix of inputs, allowing
    # multiple samples to be performed simultaneously.
    # 
    # @param inp    the matrix of input values to test
    # @param type   whether to use numpy ('np') or sympy ('sp')
    # @return       the matrix w/ values of f(x) for all sets of inputs
    # '''
def func2(inp, type = 'np'):
    if type == 'sp':
        return inp * sy.exp(-inp**2)
    return inp*np.exp(-inp**2)
    # End of method func2

''' # Method func3 defines a quadratic function of the form 
    #       f(x) = x * e^(-x)
    #
    # This method also processes a matrix of inputs, allowing
    # multiple samples to be performed simultaneously.
    # 
    # @param inp    the matrix of input values to test
    # @param type   whether to use numpy ('np') or sympy ('sp')
    # @return       the matrix w/ values of f(x) for all sets of inputs
    # '''
def func3(inp, type = 'np'):
    if type == 'sp':
        return inp*sy.exp(-inp)
    return inp*np.exp(-inp)
    # End of method func3

''' # Method func4 defines a quadratic function of the form 
    #       f(x) = sin(x)
    #
    # This method also processes a matrix of inputs, allowing
    # multiple samples to be performed simultaneously.
    # 
    # @param inp    the matrix of input values to test
    # @param type   whether to use numpy ('np') or sympy ('sp')
    # @return       the matrix w/ values of f(x) for all sets of inputs
    # '''
def func4(inp, type = 'np'):
    if type == 'sp':
        return sy.sin(inp)
    return np.sin(inp)
    # End of method func4

''' # Method trapezoidalCurve approximates the values of the function calculated by the trapezoidal
    # integral by approximating the function as a series of lines. 
    # 
    # @param x      the x-values used by the trapezoidal integral
    # @param y      the y-values used by the trapezoidal integral
    # @param sample the points at which to approximate the function
    # @precondition the sample points are in the x-range
    # @return       the approximated values of the function at the samples
    # '''
def trapezoidalCurve(x, y, sample):
    ind = np.searchsorted(x, sample, side='left')
    out = np.zeros(sample.shape[0])
    for i in range(sample.shape[0]):
        if ind[i] == 0:
            out[i] = y[0]
        else:
            out[i] = (y[ind[i]] - y[ind[i] - 1])*(sample[i] - x[ind[i] - 1]) / (
                x[ind[i]] - x[ind[i] - 1]) + y[ind[i] - 1]
    return out
    # End of method trapezoidalCurve

''' # Method simpsonCurve approximates the values of the function calculated by Simpson's method
    # integrals by approximating the function as a series of parabolas. 
    # 
    # @param x      the x-values used by the Simpson's integral
    # @param y      the y-values used by the Simpson's integral
    # @param sample the points at which to approximate the function
    # @precondition the sample points are in the x-range and there are an odd # of data points
    #               in x to complete the Simpson's rule.
    # @return       the approximated values of the function at the samples or None if the # of
    #               data points is not odd
    # '''
def simpsonCurve(x, y, sample):
    if (x.shape[0] % 2 == 0):
        print("Simpson's needs odd # of data points!")
        return None
    ind = np.searchsorted(x[::2], sample, side = 'left')
    out = np.zeros(sample.shape[0])
    for i in range(sample.shape[0]):
        if ind[i] == 0:
            out[i] = y[0]
        else:
            s = sample[i]
            a = x[2*ind[i] - 2]
            m = x[2*ind[i] - 1]
            b = x[2*ind[i]]
            out[i] = y[2*ind[i] - 2] * (s - m)*(s-b)/((a - m)*(a-b)) + y[2*ind[i] - 1] * (s-a)*(s
                -b)/((m-a)*(m-b)) + y[2*ind[i]] * (s-a)*(s-m)/((b-a)*(b-m))
    return out
    # End of method simpsonCurve

''' # Method calcRMS calculates the RMS error of a given function approximation at
    # a given sampling of points.
    # 
    # @param x          the x-values used by the approximation
    # @param y          the y-values used by the approximation
    # @param sample     the points at which to sample
    # @param curveFunc  the function to use to approximate the curve
    # @param calcFunc   the function to use to calculate the empirical values
    # @precondition     the sample values are in the range of x, and the # of
    #                   datapoints in x satisfies the precondition of the
    #                   approximation used
    # @return           the RMS Error between the function and its approximation
    # '''
def calcRMS(x, y, sample, curveFunc, calcFunc):
    pred = curveFunc(x, y, sample)
    act = calcFunc(sample)
    if pred is None:
        return None
    return np.sqrt(np.mean((pred - act)**2))
    # End of method calcRMS

''' # Method calcError calculates the percent error between a predicted and actual value.
    # 
    # @param pred       the predicted value
    # @param act        the actual value
    # @return           the percent error, and if the actual value is 0, return None
    # '''
def calcError(pred, act):
    if (act == 0):
        print("Cannot calculate error: Actual value 0")
        return None
    return abs((act-pred)/act) * 100
    # End of method calcError

''' # Method trapezoidal approximates a definite integral for a sorted list of coordinates
    # by approximating the area between each pair of points as a trapezoid between those points
    # and the x-axis.
    # 
    # @param x      the x-values used by the integral
    # @param y      the y-values used by the integral
    # @precondition the x-values are sorted
    # @return       the value of the trapezoidal integral
    # '''
def trapezoidal(x, y):
    return np.sum(np.diff(x)*((y + np.roll(y, -1))[:-1]))/2
    # End of method trapezoidal

''' # Method simpson approximates a definite integral for a uniformly distributed coordinate list
    # by approximating the area between each trio of points as the area between a parabola through
    # all three points and the x-axis.
    # 
    # @param x      the x-values used by the integral
    # @param y      the y-values used by the integral
    # @precondition there is an odd # of data points which are uniformly distributed
    # @return       the value of the Simpson's integral
    # '''
def simpson(x, y):
    if (x.shape[0] % 2 == 0):
        print("Simpson's needs odd # of data points!")
        return None
    interval = x[2]-x[0]
    temp = y.copy()
    temp[2::2] *= 2
    temp[temp.shape[0] - 1] /= 2
    temp[1::2] *= 4
    temp *= interval/6
    return np.sum(temp)
    # End of method simpson

''' # Method test tests the trapezoidal and Simpson's rule integrals for a function in the range
    # [-2, 2], comparing the methods to empirical results using the calculated integral values,
    # the RMS error of the approximation, and the integral % error. Finally, the function and
    # the approximation are graphed.
    #
    # @param func           the function to integrate
    # @param numPoints      the number of points at which to approximate
    # @param sample         the sample used to calculate RMS error
    # '''
def test(func, numPoints, sample):
    data = np.linspace(-2, 2, numPoints)
    test = func(sample)
    actFunc = func(data)
    print("\n ----------- Next Test -----------\n")
    x = sy.Symbol("x")
    act = sy.integrate(func(x, 'sp'), (x, -2, 2)).evalf()
    print("Empirical value: {}".format(act))
    trap = trapezoidal(data, actFunc)
    print("Trapezoidal value: {}".format(trap))
    simp = simpson(data, actFunc)
    print("Simpson value: {}\n".format(simp))
    trapRMS = calcRMS(data, actFunc, sample, trapezoidalCurve, func)
    print("Trapezoidal RMS: {}".format(trapRMS))
    simpRMS = calcRMS(data, actFunc, sample, simpsonCurve, func)
    print("Simpson RMS: {}\n".format(simpRMS))
    trapError = calcError(trap, act)
    print("Trapezoidal % Error: {}".format(trapError))
    simpError = None
    if not simp is None:
        simpError = calcError(simp, act)
    print("Simpson % Error: {}".format(simpError))
    
    plt.figure()
    plt.plot(sample, test, 'black')
    plt.plot(sample, trapezoidalCurve(data, actFunc, sample), 'g')
    plt.figure()
    plt.plot(sample, test, 'black')
    if not simp is None:
        plt.plot(sample, simpsonCurve(data, actFunc, sample), 'r')

''' # Method main tests the integration techniques for all four functions described in the lab
    # document, establishing the sample set as a uniformly distributed set of 10000 points in
    # the range [-2, 2].
    # 
    # @param numPoints      the number of points at which to integrate
    # '''
def main(numPoints):
    sample = np.linspace(-2, 2, 10000)
    test(func1, numPoints, sample)
    test(func2, numPoints, sample)
    test(func3, numPoints, sample)
    test(func4, numPoints, sample)
    plt.show()

'''#
   # This statement functions to run the main function by default. This statement also encodes
   # a command line argument to accept the number of points to sample.
   #'''
if (__name__ == "__main__"):
    commandLineArgs = argparse.ArgumentParser()
    commandLineArgs.add_argument("--num_points", type = int, default = 100)
    arguments = commandLineArgs.parse_args() 
    main(arguments.num_points)