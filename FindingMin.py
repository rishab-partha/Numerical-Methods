import argparse
import math
from turtle import left
from cv2 import rectangle
import matplotlib.pyplot as plt
import numpy as np

''' # Class CalcFunc defines a functionality for calculating the values of the function
    # (t, f(t)), its derivative (t, f'(t)), and (t, f''(t)). This class also provides functions
    # for saving the values and derivatives to files as a line-delineated sequence of ordered
    # pairs.
    #
    # @author Rishab Parthasarathy
    # @version 03.25.2022
    #'''
class CalcFunc:
    ''' # Method __init__ defines the constructor for the range in which the function is calculated
        # along with the number of points to sample in the range. Then, the method calculates
        # the values of a given f(t) and its derivative at a uniform sampling of points in
        # the range provided. The derivative is calculated using the five-point stencil, with right
        # and left sided derivatives used for the edges and three-point derivatives used for the 
        # second points from edges.
        # 
        # @param self       the instance of the CalcFunc class created
        # @param rangeL     the left boundary of the range
        # @param rangeR     the right boundary of the range
        # @param numPoints  the number of points to sample in the range
        # @postcondition    the instance of CalcFunc has been instantiated with appropriate values 
        #                   of f(t) and its derivative
        #'''
    def __init__(self, rangeL, rangeR, numPoints):
        self.numPoints = numPoints
        self.rangeR = rangeR
        self.rangeL = rangeL
        intervalSize = (self.rangeR - self.rangeL) / (self.numPoints - 1)

        self.funcVals = np.zeros((2, self.numPoints))
        for i in range(self.numPoints):
            evalPoint = self.rangeL + i*intervalSize
            self.funcVals[0][i] = evalPoint
            self.funcVals[1][i] = evalPoint*math.e**(evalPoint)

        self.derivVals = np.zeros((2, self.numPoints))
        self.secDeriv = np.zeros((2, self.numPoints))
        self.derivVals[0][0] = self.rangeL
        self.derivVals[1][0] = (self.funcVals[1][1] - self.funcVals[1][0])/intervalSize
        self.derivVals[0][1] = self.rangeL + intervalSize
        self.derivVals[1][1] = (self.funcVals[1][2] - self.funcVals[1][0])/(2*intervalSize)
        for i in range(2, self.numPoints - 2):
            self.derivVals[0][i] = self.rangeL + (i)*intervalSize
            self.derivVals[1][i] = self.funcVals[1][i-2] + -8*self.funcVals[1][i - 1]
            self.derivVals[1][i] += 8*self.funcVals[1][i + 1] + -1*self.funcVals[1][i + 2]
            self.derivVals[1][i] /= 12*intervalSize

        self.derivVals[0][self.numPoints - 2] = self.rangeR - intervalSize
        self.derivVals[1][self.numPoints - 2] = (self.funcVals[1][self.numPoints - 1] - 
            self.funcVals[1][self.numPoints - 3])/(2*intervalSize)
        self.derivVals[0][self.numPoints - 1] = self.rangeR
        self.derivVals[1][self.numPoints - 1] = (self.funcVals[1][self.numPoints - 1] - 
            self.funcVals[1][self.numPoints - 2])/intervalSize

        self.secDeriv[0][0] = self.rangeL
        self.secDeriv[1][0] = (self.derivVals[1][1] - self.derivVals[1][0])/intervalSize
        self.secDeriv[0][1] = self.rangeL + intervalSize
        self.secDeriv[1][1] = (self.derivVals[1][2] - self.derivVals[1][0])/(2*intervalSize)
        for i in range(2, self.numPoints - 2):
            self.secDeriv[0][i] = self.rangeL + (i)*intervalSize
            self.secDeriv[1][i] = self.derivVals[1][i-2] + -8*self.derivVals[1][i - 1]
            self.secDeriv[1][i] += 8*self.derivVals[1][i + 1] + -1*self.derivVals[1][i + 2]
            self.secDeriv[1][i] /= 12*intervalSize

        self.secDeriv[0][self.numPoints - 2] = self.rangeR - intervalSize
        self.secDeriv[1][self.numPoints - 2] = (self.derivVals[1][self.numPoints - 1] - 
            self.derivVals[1][self.numPoints - 3])/(2*intervalSize)
        self.secDeriv[0][self.numPoints - 1] = self.rangeR
        self.secDeriv[1][self.numPoints - 1] = (self.derivVals[1][self.numPoints - 1] - 
            self.derivVals[1][self.numPoints - 2])/intervalSize
        # End of method __init__

    ''' # Method calcFuncVals returns the values of (t, f(t)) in the given range.
        #
        # @param self   the instance of CalcFunc storing the values
        # @return       the values of (t, f(t)) in the given range 
        #'''
    def calcFuncVals(self):
        return self.funcVals
        # End of method calcFuncVals

    ''' # Method withinEps checks whether a value t is within epsilon (defined as 1e-6 * n) of a
        # value n.
        #
        # @param self   the calcFunc performing the computation
        # @param t      the value to compare
        # @param n      the value of n to compare
        # @return       whether t is within epsilon of n
        #'''
    def withinEps(self, t, n):
        return t < n + max(n*1e-6,1e-6) and t > n- max(n*1e-6,1e-6)
    
    ''' # Method calcDerivativeVals returns the values of the derivative (t, f'(t)) 
        # in the given range.
        #
        # @param self   the instance of CalcFunc storing the values
        # @return       the values of (t, f'(t)) in the given range 
        #'''
    def calcDerivativeVals(self):
        return self.derivVals
        # End of method calcDerivativeVals

    ''' # Method calcSecDerivativeVals returns the values of the 2nd derivative (t, f''(t)) 
        # in the given range.
        #
        # @param self   the instance of CalcFunc storing the values
        # @return       the values of (t, f''(t)) in the given range 
        #'''
    def calcSecDerivativeVals(self):
        return self.secDeriv
        # End of method calcSecDerivativeVals

    ''' # Method saveFuncVals saves the values of (t, f(t)) in the given range to
        # 'function.txt', storing the values as a line separated file of ordered pairs.
        #
        # @param self       the instance of CalcFunc storing the values
        # @postcondition    the values of (t, f(t)) in the given range have been saved to
        #                   'function.txt' as ordered pairs
        #'''
    def saveFuncVals(self):
        np.savetxt('function.txt', np.column_stack(self.funcVals), fmt='(%f, %f)')
        # End of method saveFuncVals

    ''' # Method saveDerivVals saves the values of (t, f'(t)) in the given range to
        # 'deriv.txt', storing the values as a line separated file of ordered pairs.
        #
        # @param self       the instance of CalcFunc storing the values
        # @postcondition    the values of (t, f'(t)) in the given range have been saved to
        #                   'deriv.txt' as ordered pairs
        #'''
    def saveDerivVals(self):
        np.savetxt('deriv.txt', np.column_stack(self.derivVals), fmt='(%f, %f)')
        # End of method saveDerivVals
    # End of class CalcFunc

''' # Class FindMin calculates one local min of the function f(t) by using the bisection method,
    # which progressively halves the interval in which the min can be found. This class also
    # accounts for edge cases like asymptotic min and unconfirmed min.
    #'''
class FindMin:
    ''' # Method __init__ defines the constructor for the range in which the function is calculated
        # along with the number of points to sample in the range. Then, the method calculates
        # the values of the derivative of f(t), its derivative f'(t), and f''(t) at a 
        # uniform sampling of points in the range provided using the CalcFunc class.
        # 
        # @param self       the instance of the FindMin class created
        # @param rangeL     the left boundary of the range
        # @param rangeR     the right boundary of the range
        # @param numPoints  the number of points to sample in the range
        # @postcondition    the instance of FindMin has been instantiated with proper
        #                   boundaries and values of f(t), f'(t), and f''(t)
        #'''
    def __init__(self, rangeL, rangeR, numPoints):
        self.func = CalcFunc(rangeL, rangeR, numPoints)
        self.rangeL = rangeL
        self.rangeR = rangeR
        self.numPoints = numPoints
        self.funcVals = self.func.calcFuncVals()
        self.derivVals = self.func.calcDerivativeVals()
        self.secDeriv = self.func.calcSecDerivativeVals()
        # End of method __init__

    ''' # Method findMin uses the bisection method to find a local min of the function f(t) in the
        # interval. First, the method checks the endpoints for asymptotic min, and then, the
        # max and min values are checked for either no mins or unconfirmed mins. Finally,
        # the bisection method is used with the min and max values to progressively find the
        # interval in which the possible min lies, using f'(t).
        #
        # @param self       the instance of FindMin performing the computation
        # @postcondition    the method has identified possible asymptotic min, possible
        #                   unconfirmed min, and a min calculated from the bisection method
        #                   if possible
        #'''
    def findMin(self):
        if (abs(self.derivVals[1][0]) < 1e-1 and abs(self.secDeriv[1][0]) < 1e-1 and
            self.derivVals[1][0] > 0):
            print("Possible asymptotic min on the negative t-axis.")
        
        if abs(self.derivVals[1][self.numPoints - 1]) < 1e-1 and abs(self.secDeriv[1][self.numPoints -
            1]) < 1e-1 and self.derivVals[1][self.numPoints - 1] < 0:
            print("Possible asymptotic min on the positive t-axis.")

        minIndex = np.argmin(self.derivVals[1])
        maxIndex = np.argmax(self.derivVals[1])
        if self.derivVals[1][maxIndex] <= 0.0:
            print("No other min found.")
            return

        if self.derivVals[1][minIndex] >= 0.0:
            print("No other min found.")
            return

        leftIndex = min(minIndex, maxIndex)
        rightIndex = max(minIndex, maxIndex)
        nIters = 0
        while (leftIndex + 1 < rightIndex):
            midIndex = (leftIndex + rightIndex)//2
            if (self.func.withinEps(self.derivVals[1][midIndex], 0.0)):
                if self.secDeriv[1][midIndex] > 0.0:
                    print("min near {}".format(self.derivVals[0][midIndex]))
                    return
                else:
                    break
            
            if np.sign(self.derivVals[1][leftIndex]) == np.sign(self.derivVals[1][midIndex]):
                leftIndex = midIndex
            else:
                rightIndex = midIndex
        
        if self.secDeriv[1][midIndex] > 0.0:
            print("min between {} and {}".format(self.funcVals[0][leftIndex], 
                self.funcVals[0][rightIndex]))
            return
        print("min not found, possibly stuck around max")
        # End of method findMin
    # End of class FindMin

''' # Method main tests and plots the finding minimum method, first plotting
    # f(t), f'(t), and f''(t) and then calculating for a local root in the given range if possible.
    #
    # @param arguments      Arguments from the command line, specifically the number of points
    #                       to sample, the left bound, and the right bound
    # @postcondition        Three plots have been generated, first with the original function, the
    #                       second with the derivative, the third with the second deriv. Also, the 
    #                       approximated mins have been printed.
    #'''
def main(arguments):
    numPoints = arguments.num_points
    rangeL = arguments.left_bound
    rangeR = arguments.right_bound
    c = CalcFunc(rangeL, rangeR, numPoints)
    vals = c.calcFuncVals()
    derivs = c.calcDerivativeVals()
    secDeriv = c.calcSecDerivativeVals()
    c.saveFuncVals()
    c.saveDerivVals()
    plt.figure()
    plt.plot(vals[0], vals[1])
    plt.figure()
    plt.plot(derivs[0], derivs[1])
    plt.figure()
    plt.plot(secDeriv[0], secDeriv[1])
    f = FindMin(rangeL, rangeR, numPoints)
    f.findMin()
    plt.show()
    # End of method main

'''#
   # This statement functions to run the main function by default. This statement also encodes
   # a command line argument to accept the number of points to sample, the left bound of range,
   # and the right bound of range.
   #'''
if (__name__ == "__main__"):
    commandLineArgs = argparse.ArgumentParser()
    commandLineArgs.add_argument("--left_bound", type = int, default = -10)
    commandLineArgs.add_argument("--right_bound", type = int, default = 10)
    commandLineArgs.add_argument("--num_points", type = int, default = 100)
    arguments = commandLineArgs.parse_args() 
    main(arguments)