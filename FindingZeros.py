import argparse
import math
from cv2 import rectangle
import matplotlib.pyplot as plt
import numpy as np

''' # Class CalcFunc defines a functionality for calculating the values of the function
    # (t, f(t)) and its derivative (t, f'(t)). This class also provides functionalities
    # for saving the values and derivatives to files as a line-delineated sequence of ordered
    # pairs. Finally, this class provides functionalities for approximating the function between
    # discrete data points as a line.
    #
    # @author Rishab Parthasarathy
    # @version 03.01.2022
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
            self.funcVals[1][i] = math.sqrt(abs(evalPoint))

        self.derivVals = np.zeros((2, self.numPoints))
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
        # End of method __init__

    ''' # Method calcFuncVals returns the values of (t, f(t)) in the given range.
        #
        # @param self   the instance of CalcFunc storing the values
        # @return       the values of (t, f(t)) in the given range 
        #'''
    def calcFuncVals(self):
        return self.funcVals
        # End of method calcFuncVals

    ''' # Method calcFuncVal returns the value of f(t) for a given t in the range, with
        # values in between two samples approximated by a line.
        # 
        # @param self   the instance of CalcFunc storing f(t)
        # @param t      the time at which to query
        # @precondition the time is the range of CalcFunc
        # @return       the value of f(t), with values in between two samples approximated
        #               by a line
        #'''
    def calcFuncVal(self, t):
        index = np.searchsorted(self.funcVals[0], t, side = 'left')
        if index == 0:
            return self.funcVals[1][0]

        elif index == self.numPoints:
            return self.funcVals[1][self.numPoints - 1]

        return self.funcVals[1][index - 1] + (self.funcVals[1][index] -
            self.funcVals[1][index - 1])*(t - self.funcVals[0][index - 1])/(self.funcVals[0][index]-
            self.funcVals[0][index - 1])


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
    
    ''' # Method calcDerivVal returns the value of f'(t) for a given t in the range, with
        # values in between two samples approximated by a line.
        # 
        # @param self   the instance of CalcFunc storing f'(t)
        # @param t      the time at which to query
        # @precondition the time is the range of CalcFunc
        # @return       the value of f'(t), with values in between two samples approximated
        #               by a line
        #'''
    def calcDerivVal(self, t):
        index = np.searchsorted(self.derivVals[0], t, side = 'left')
        if index == 0:
            return self.derivVals[1][0]

        elif index == self.numPoints:
            return self.derivVals[1][self.numPoints - 1]

        elif self.withinEps(t, self.derivVals[0][index - 1]):
            return self.derivVals[1][index - 1]
        
        elif self.withinEps(t, self.derivVals[0][index]):
            return self.derivVals[1][index]

        return (self.funcVals[1][index] - self.funcVals[1][index - 1])/(self.funcVals[0][index] -
            self.funcVals[0][index - 1])
    
    ''' # Method calcDerivativeVals returns the values of the derivative (t, f'(t)) 
        # in the given range.
        #
        # @param self   the instance of CalcFunc storing the values
        # @return       the values of (t, f'(t)) in the given range 
        #'''
    def calcDerivativeVals(self):
        return self.derivVals
        # End of method calcDerivativeVals

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

''' # Class Bisection calculates one local zero of the function f(t) by using the bisection method,
    # which progressively halves the interval in which the zero can be found. This class also
    # accounts for edge cases like asymptotic zeroes and unconfirmed zeroes.
    #'''
class Bisection:
    ''' # Method __init__ defines the constructor for the range in which the function is calculated
        # along with the number of points to sample in the range. Then, the method calculates
        # the  values of the derivative of f(t) and its derivative f'(t) at a 
        # uniform sampling of points in the range provided using the CalcFunc class.
        # 
        # @param self       the instance of the Bisection class created
        # @param rangeL     the left boundary of the range
        # @param rangeR     the right boundary of the range
        # @param numPoints  the number of points to sample in the range
        # @postcondition    the instance of Bisection has been instantiated with proper
        #                   boundaries and values of f(t) and f'(t)
        #'''
    def __init__(self, rangeL, rangeR, numPoints):
        self.func = CalcFunc(rangeL, rangeR, numPoints)
        self.rangeL = rangeL
        self.rangeR = rangeR
        self.numPoints = numPoints
        self.funcVals = self.func.calcFuncVals()
        self.derivVals = self.func.calcDerivativeVals()
        # End of method __init__

    ''' # Method findRoot uses the bisection method to find a root of the function f(t) in the 
        # interval. First, the method checks the endpoints for asymptotic zeroes, and then, the
        # max and min values are checked for either no zeroes or unconfirmed zeroes. Finally,
        # the bisection method is used with the min and max values to progressively find the
        # interval in which the possible zero lies.
        #
        # @param self       the instance of Bisection performing the computation
        # @postcondition    the method has identified possible asymptotic zeroes, possible
        #                   unconfirmed zeroes, and a zero calculated from the bisection method
        #                   if possible
        #'''
    def findRoot(self):
        if abs(self.funcVals[1][0]) < 1e-1 and abs(self.derivVals[1][0]) < 1e-1:
            print("Possible asymptotic zero on the negative t-axis.")
        
        if abs(self.funcVals[1][self.numPoints - 1]) < 1e-1 and abs(self.derivVals[1][self.numPoints -
            1]) < 1e-1:
            print("Possible asymptotic zero on the positive t-axis.")

        minIndex = np.argmin(self.funcVals[1])
        maxIndex = np.argmax(self.funcVals[1])
        if self.funcVals[1][maxIndex] <= 0.0:
            if maxIndex == 0 or maxIndex == self.numPoints - 1:
                print("No (other) zeroes/min found.")
                return
            if self.func.withinEps(self.funcVals[1][maxIndex], 0.0):
                print("zero near t = {}".format(self.funcVals[0][maxIndex]))
            elif self.funcVals[1][maxIndex] > -1e-1:
                print("Possible unconfirmed zero near t = {}".format(self.funcVals[0][maxIndex]))
            else:
                print("No (other) zeroes/min found.")
            return

        if self.funcVals[1][minIndex] >= 0.0:
            if minIndex == 0 or minIndex == self.numPoints - 1:
                print("No (other) zeroes/min found.")
                return
            if self.func.withinEps(self.funcVals[1][minIndex], 0.0):
                print("Zero near t = {}".format(self.funcVals[0][minIndex]))
            elif self.funcVals[1][minIndex] < 1e-1:
                print("Possible unconfirmed zero near t = {}".format(self.funcVals[0][minIndex]))
            else:
                print("No (other) zeroes/min found.")
            return

        leftIndex = min(minIndex, maxIndex)
        rightIndex = max(minIndex, maxIndex)
        while (leftIndex + 1 < rightIndex):
            midIndex = (leftIndex + rightIndex)//2
            if (self.func.withinEps(self.funcVals[1][midIndex], 0.0)):
                print("zero near {}".format(self.funcVals[0][midIndex]))
                return
            
            if np.sign(self.funcVals[1][leftIndex]) == np.sign(self.funcVals[1][midIndex]):
                leftIndex = midIndex
            else:
                rightIndex = midIndex
        
        print("zero between {} and {}".format(self.funcVals[0][leftIndex], 
            self.funcVals[0][rightIndex]))
            # End of method findRoot
    # End of class Bisection

''' # Class NewtonRaphson calculates one local zero of the function f(t) using the Newton-Raphson 
    # method, which uses the derivative to progressively approximate where a line approximation 
    # of the function would have a root. This class also accounts for edge cases like asymptotic 
    # zeroes and unconfirmed zeroes.
    #'''
class NewtonRaphson:
    ''' # Method __init__ defines the constructor for the range in which the function is calculated
        # along with the number of points to sample in the range. Then, the method calculates
        # the values of the derivative of f(t) and its derivative f'(t) at a 
        # uniform sampling of points in the range provided using the CalcFunc class.
        # 
        # @param self       the instance of the NewtonRaphson class created
        # @param rangeL     the left boundary of the range
        # @param rangeR     the right boundary of the range
        # @param numPoints  the number of points to sample in the range
        # @postcondition    the instance of NewtonRaphson has been instantiated with proper
        #                   boundaries and values of f(t) and f'(t)
        #'''
    def __init__(self, rangeL, rangeR, numPoints):
        self.func = CalcFunc(rangeL, rangeR, numPoints)
        self.rangeL = rangeL
        self.rangeR = rangeR
        self.numPoints = numPoints
        self.funcVals = self.func.calcFuncVals()
        self.derivVals = self.func.calcDerivativeVals()
        # End of method __init__

    ''' # Method findRoot uses the Newton-Raphson method to find a root of f(t) in the 
        # interval. First, the method checks the endpoints for asymptotic zeroes, and then, the
        # max and min values are checked for either no zeroes or unconfirmed zeroes. Finally,
        # the Newton-Raphson method is used with the provided start point to check for a possible
        # zero, with a set number of iterations set to prevent getting stuck at a local minima or
        # saddle point.
        #
        # @param self       the instance of Bisection performing the computation
        # @param startPoint the t-coordinate at which to begin the Newton-Raphson method
        # @param isMin      
        # @postcondition    the method has identified possible asymptotic zeroes, possible
        #                   unconfirmed zeroes, and a zero calculated from the Newton-Raphson 
        #                   method if possible
        #'''
    def findRoot(self, startPoint):
        if abs(self.funcVals[1][0]) < 1e-1 and abs(self.derivVals[1][0]) < 1e-1:
            print("Possible asymptotic zero on the negative t-axis.")
        
        if abs(self.funcVals[1][self.numPoints - 1]) < 1e-1 and abs(self.derivVals[1][self.numPoints -
            1]) < 1e-1:
            print("Possible asymptotic zero on the positive t-axis.")

        minIndex = np.argmin(self.funcVals[1])
        maxIndex = np.argmax(self.funcVals[1])
        if self.funcVals[1][maxIndex] <= 0.0:
            if maxIndex == 0 or maxIndex == self.numPoints - 1:
                print("No (other) zeroes/min found.")
                return
            if self.func.withinEps(self.funcVals[1][maxIndex], 0.0):
                print("Zero at t = {}".format(self.funcVals[0][maxIndex]))
            elif self.funcVals[1][maxIndex] > -1e-1:
                print("Possible unconfirmed zero near t = {}".format(self.funcVals[0][maxIndex]))
            else:
                print("No (other) zeroes/min found.")
                return
            

        if self.funcVals[1][minIndex] >= 0.0:
            if minIndex == 0 or minIndex == self.numPoints - 1:
                print("No (other) zeroes/min found.")
                return
            if self.func.withinEps(self.funcVals[1][minIndex], 0.0):
                print("Zero at t = {}".format(self.funcVals[0][minIndex]))
            elif self.funcVals[1][minIndex] < 1e-1:
                print("Possible unconfirmed zero near t = {}".format(self.funcVals[0][minIndex]))
            else:
                print("No (other) zeroes/min found.")
                return

        curPoint = startPoint
        curValue = self.func.calcFuncVal(curPoint)
        curDeriv = self.func.calcDerivVal(curPoint)
        iters = 0
        while (not self.func.withinEps(curDeriv, 0.0)) and iters < 1000:
            curPoint = curPoint - curValue/curDeriv
            if curPoint < self.rangeL or curPoint > self.rangeR:
                print("Newton-Raphson did not find zero (out of bounds).")
                return
            curValue = self.func.calcFuncVal(curPoint)
            curDeriv = self.func.calcDerivVal(curPoint)
            iters += 1
        if abs(curValue) > 1e-2:
            print("Newton-Raphson did not find zero.")

        else:
            print("zero found around {}".format(curPoint))
        return
        # End of method findRoot
    # End of class NewtonRaphson


''' # Method main tests and plots the bisection method and Newton-Raphson method, first plotting
    # f(t) and f'(t) and then calculating for a local root in the given range if possible.
    #
    # @param arguments      Arguments from the command line, specifically the number of points
    #                       to sample, the left bound, and the right bound
    # @postcondition        Two plots have been generated, first with the original function, the
    #                       second with the derivative. Also, the approximated roots have been
    #                       printed for bisection and Newton-Raphson.
    #'''
def main(arguments):
    numPoints = arguments.num_points
    rangeL = arguments.left_bound
    rangeR = arguments.right_bound
    c = CalcFunc(rangeL, rangeR, numPoints)
    vals = c.calcFuncVals()
    derivs = c.calcDerivativeVals()
    c.saveFuncVals()
    c.saveDerivVals()
    plt.figure()
    plt.plot(vals[0], vals[1])
    plt.figure()
    plt.plot(derivs[0], derivs[1])
    print("Bisection:")
    b = Bisection(rangeL, rangeR, numPoints)
    b.findRoot()
    print("\nNewton-Raphson:")
    newt = NewtonRaphson(rangeL, rangeR, numPoints)
    newt.findRoot(0.2)
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