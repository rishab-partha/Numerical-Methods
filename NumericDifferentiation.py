import argparse
import math
from cv2 import rectangle
import matplotlib.pyplot as plt
import numpy as np

''' # Class CalcFunc defines a functionality for calculating the values of the function
    # (t, e^(-t^2)) and its derivative (t, -2te^(-t^2)). This class also provides functionalities
    # for saving the values and derivatives to files as a line-delineated sequence of ordered
    # pairs.
    #
    # @author Rishab Parthasarathy
    # @version 02.10.2022
    #'''
class CalcFunc:
    ''' # Method __init__ defines the constructor for the range in which the function is calculated
        # along with the number of points to sample in the range. Then, the method calculates
        # the values of e^(-t^2) and its derivative -2te^(-t^2) at a uniform sampling of points in
        # the range provided.
        # 
        # @param self       the instance of the CalcFunc class created
        # @param rangeR     the right boundary of the range
        # @param rangeL     the left boundary of the range
        # @param numPoints  the number of points to sample in the range
        # @postcondition    the instance of CalcFunc has been instantiated with appropriate values 
        #                   of e^(-t^2) and its derivative
        #'''
    def __init__(self, rangeR, rangeL, numPoints):
        self.numPoints = numPoints
        self.rangeR = rangeR
        self.rangeL = rangeL
        intervalSize = (self.rangeR - self.rangeL) / (self.numPoints - 1)

        self.funcVals = np.zeros((2, self.numPoints))
        for i in range(self.numPoints):
            evalPoint = self.rangeL + i*intervalSize
            self.funcVals[0][i] = evalPoint
            self.funcVals[1][i] = math.e**(-1*evalPoint**2)

        newIntervalSize = (self.rangeR - self.rangeL) / (2*(self.numPoints - 1))
        self.derivVals = np.zeros((2, 2*self.numPoints-1))
        for i in range(2*self.numPoints - 1):
            evalPoint = self.rangeL + i*newIntervalSize
            self.derivVals[0][i] = evalPoint
            self.derivVals[1][i] = -2*evalPoint*math.e**(-1*evalPoint**2)
        # End of method __init__

    ''' # Method calcFuncVals returns the values of (t, e^(-t^2)) in the given range.
        #
        # @param self   the instance of CalcFunc storing the values
        # @return       the values of (t, e^(-t^2)) in the given range 
        #'''
    def calcFuncVals(self):
        return self.funcVals
        # End of method calcFuncVals

    ''' # Method calcDerivativeVals returns the values of the derivative (t, -2te^(-t^2)) 
        # in the given range.
        #
        # @param self   the instance of CalcFunc storing the values
        # @return       the values of (t, -2te^(-t^2)) in the given range 
        #'''
    def calcDerivativeVals(self):
        return self.derivVals
        # End of method calcDerivativeVals

    ''' # Method saveFuncVals saves the values of (t, e^(-t^2)) in the given range to
        # 'function.txt', storing the values as a line separated file of ordered pairs.
        #
        # @param self       the instance of CalcFunc storing the values
        # @postcondition    the values of (t, e^(-t^2)) in the given range have been saved to
        #                   'function.txt' as ordered pairs
        #'''
    def saveFuncVals(self):
        np.savetxt('function.txt', np.column_stack(self.funcVals), fmt='(%f, %f)')
        # End of method saveFuncVals

    ''' # Method saveDerivVals saves the values of (t, -2te^(-t^2)) in the given range to
        # 'deriv.txt', storing the values as a line separated file of ordered pairs.
        #
        # @param self       the instance of CalcFunc storing the values
        # @postcondition    the values of (t, -2te^(-t^2)) in the given range have been saved to
        #                   'deriv.txt' as ordered pairs
        #'''
    def saveDerivVals(self):
        np.savetxt('deriv.txt', np.column_stack(self.derivVals), fmt='(%f, %f)')
        # End of method saveDerivVals
    # End of class CalcFunc

''' # Class SlopeDerivative defines a functionality for calculating the derivatives of the function
    # (t, e^(-t^2)) by computing the slopes using conventional algebra. This class offers three
    # locations to place the slopes: the left, middle, and right of the interval. Finally, this 
    # class provides functionalities for evaluating the effectiveness of various derivative
    # calculations by evaluating the RMS error.
    #'''
class SlopeDerivative:
    ''' # Method __init__ defines the constructor for the range in which the function is calculated
        # along with the number of points to sample in the range. Then, the method calculates
        # the empirical values of e^(-t^2) and its derivative -2te^(-t^2) at a uniform sampling of 
        # points in the range provided using the CalcFunc class.
        # 
        # @param self       the instance of the SlopeDerivative class created
        # @param rangeL     the left boundary of the range
        # @param rangeR     the right boundary of the range
        # @param numPoints  the number of points to sample in the range
        # @postcondition    the instance of SlopeDerivative has been instantiated with appropriate
        #                   boundaries and empirical values of e^(-t^2) and its derivative
        #'''
    def __init__(self, rangeL, rangeR, numPoints):
        self.func = CalcFunc(rangeR, rangeL, numPoints)
        self.rangeL = rangeL
        self.rangeR = rangeR
        self.numPoints = numPoints
        self.funcVals = self.func.calcFuncVals()
        self.derivVals = self.func.calcDerivativeVals()
        # End of method __init__
    
    ''' # Method calcLeftDeriv calculates the derivative of (t, e^(-t^2)) using simple slopes and
        # assigns the discovered value to the left boundary of the interval. Then, this method
        # calculates the RMS error and saves the calculated derivative to the file 'leftderiv.txt'.
        #
        # @param self   the instance of SlopeDerivative producing the calculation
        # @return       the calculated left derivative ordered pairs and RMS error
        #'''
    def calcLeftDeriv(self):
        intervalSize = (self.rangeR - self.rangeL) / (self.numPoints - 1)
        calcDerivVals = np.zeros((2, self.numPoints - 1))
        rmsError = 0.0
        for i in range(self.numPoints - 1):
            evalPointLeft = self.rangeL + i*intervalSize
            calcDerivVals[0][i] = evalPointLeft
            calcDerivVals[1][i] = (self.funcVals[1][i + 1] - self.funcVals[1][i])/intervalSize
            rmsError += (calcDerivVals[1][i] - self.derivVals[1][2*i])**2
        
        rmsError = math.sqrt(rmsError/(self.numPoints - 1))
        np.savetxt('leftderiv.txt', np.column_stack(calcDerivVals), fmt = '(%f, %f)')
        return calcDerivVals, rmsError
        # End of method calcLeftDeriv

    ''' # Method calcMidDeriv calculates the derivative of (t, e^(-t^2)) using simple slopes and
        # assigns the discovered value to the middle of the interval used. Then, this method
        # calculates the RMS error and saves the calculated derivative to the file 'midderiv.txt'.
        #
        # @param self   the instance of SlopeDerivative producing the calculation
        # @return       the calculated middle derivative ordered pairs and RMS error
        #'''
    def calcMidDeriv(self):
        intervalSize = (self.rangeR - self.rangeL) / (self.numPoints - 1)
        calcDerivVals = np.zeros((2, self.numPoints - 1))
        rmsError = 0.0
        for i in range(self.numPoints - 1):
            evalPointLeft = self.rangeL + i*intervalSize
            evalPointRight = self.rangeL + (i+1)*intervalSize
            calcDerivVals[0][i] = (evalPointLeft + evalPointRight)/2
            calcDerivVals[1][i] = (self.funcVals[1][i + 1] - self.funcVals[1][i])/intervalSize
            rmsError += (calcDerivVals[1][i] - self.derivVals[1][2*i+1])**2
        
        rmsError = math.sqrt(rmsError/(self.numPoints - 1))
        np.savetxt('midderiv.txt', np.column_stack(calcDerivVals), fmt = '(%f, %f)')
        return calcDerivVals, rmsError
        # End of method calcMidDeriv
    
    ''' # Method calcRightDeriv calculates the derivative of (t, e^(-t^2)) using simple slopes and
        # assigns the discovered value to the right of the interval used. Then, this method
        # calculates the RMS error and saves the calculated derivative to 'rightderiv.txt'.
        #
        # @param self   the instance of SlopeDerivative producing the calculation
        # @return       the calculated right derivative ordered pairs and RMS error
        #'''
    def calcRightDeriv(self):
        intervalSize = (self.rangeR - self.rangeL) / (self.numPoints - 1)
        calcDerivVals = np.zeros((2, self.numPoints - 1))
        rmsError = 0.0
        for i in range(self.numPoints - 1):
            evalPointRight = self.rangeL + (i+1)*intervalSize
            calcDerivVals[0][i] = evalPointRight
            calcDerivVals[1][i] = (self.funcVals[1][i + 1] - self.funcVals[1][i])/intervalSize
            rmsError += (calcDerivVals[1][i] - self.derivVals[1][2*i+2])**2
        
        rmsError = math.sqrt(rmsError/(self.numPoints - 1))
        np.savetxt('rightderiv.txt', np.column_stack(calcDerivVals), fmt = '(%f, %f)')
        return calcDerivVals, rmsError
        # End of method calcRightDeriv
    # End of class SlopeDerivative

''' # Class ThreePointDerivative defines a functionality for calculating the derivatives of
    # (t, e^(-t^2)) by computing the slopes using the three point derivative convention. The 
    # three-point derivative finds the slopes on the edges of an interval and assigns the average
    # of the two slopes to the middle interval. This class also provides functionalities for 
    # evaluating the effectiveness of various derivative calculations by evaluating the RMS error.
    #'''
class ThreePointDerivative:
    ''' # Method __init__ defines the constructor for the range in which the function is calculated
        # along with the number of points to sample in the range. Then, the method calculates
        # the empirical values of the derivative of e^(-t^2), -2te^(-t^2) at a uniform sampling of 
        # points in the range provided using the CalcFunc class. Finally, the method initializes
        # midpoint slope calculations using the SlopeDerivative class.
        # 
        # @param self       the instance of the ThreePointDerivative class created
        # @param rangeL     the left boundary of the range
        # @param rangeR     the right boundary of the range
        # @param numPoints  the number of points to sample in the range
        # @postcondition    the instance of ThreePointDerivative has been instantiated with proper
        #                   boundaries, empirical values of the derivative of e^(-t^2), and slope
        #                   calculations
        #'''
    def __init__(self, rangeL, rangeR, numPoints):
        self.func = CalcFunc(rangeR, rangeL, numPoints)
        self.slopeDeriv = SlopeDerivative(rangeL, rangeR, numPoints)
        self.rangeL = rangeL
        self.rangeR = rangeR
        self.numPoints = numPoints
        self.calcDerivVals = self.slopeDeriv.calcMidDeriv()[0]
        self.derivVals = self.func.calcDerivativeVals()
        # End of method __init__
    
    ''' # Method calcDeriv calculates the derivative of (t, e^(-t^2)) using the three-point
        # technique. By calculating the slopes at the endpoint of an interval, this method
        # assigns the average of the two slopes to the midpoint of the interval used. 
        # Then, this method calculates the RMS error and saves the calculated derivative to 
        # 'threepointderiv.txt'.
        #
        # @param self   the instance of ThreePointDerivative producing the calculation
        # @return       the calculated three-point derivative ordered pairs and RMS error
        #'''
    def calcDeriv(self):
        threePointDerivVals = np.zeros((2, self.numPoints - 2))
        rmsError = 0.0
        for i in range(self.numPoints - 2):
            threePointDerivVals[0][i] = (self.calcDerivVals[0][i] + self.calcDerivVals[0][i + 1])/2
            threePointDerivVals[1][i] = (self.calcDerivVals[1][i] + self.calcDerivVals[1][i + 1])/2
            rmsError += (threePointDerivVals[1][i] - self.derivVals[1][2*i + 2])**2
        
        rmsError = math.sqrt(rmsError/(self.numPoints - 2))
        np.savetxt('threepointderiv.txt', np.column_stack(threePointDerivVals), fmt = '(%f, %f)')
        return threePointDerivVals, rmsError
        # End of method calcDeriv
    # End of class ThreePointDerivative

''' # Class ParabolaDerivative defines a functionality for calculating the derivatives of
    # (t, e^(-t^2)) by locally approximating the function as a parabola. For each set of three
    # points, this class finds the parabola y = at^2 + bt + c that goes through all three points. 
    # Then, using 2at + b as the derivative, this class approximates the derivative at the points 
    # queried. Finally, this class also provides functionalities for evaluating the effectiveness 
    # of various derivative calculations by evaluating the RMS error.
    #'''
class ParabolaDerivative:
    ''' # Method __init__ defines the constructor for the range in which the function is calculated
        # along with the number of points to sample in the range. Then, the method calculates
        # the empirical values of the derivative of e^(-t^2) and its derivative -2te^(-t^2) at a 
        # uniform sampling of points in the range provided using the CalcFunc class.
        # 
        # @param self       the instance of the ParabolaDerivative class created
        # @param rangeL     the left boundary of the range
        # @param rangeR     the right boundary of the range
        # @param numPoints  the number of points to sample in the range
        # @postcondition    the instance of ParabolaDerivative has been instantiated with proper
        #                   boundaries and empirical values of e^(-t^2) and its derivative
        #'''
    def __init__(self, rangeL, rangeR, numPoints):
        self.func = CalcFunc(rangeR, rangeL, numPoints)
        self.rangeL = rangeL
        self.rangeR = rangeR
        self.numPoints = numPoints
        self.funcVals = self.func.calcFuncVals()
        self.derivVals = self.func.calcDerivativeVals()
        # End of method __init__

    ''' # Method calcDeriv calculates the derivative of (t, e^(-t^2)) using the parabola
        # technique. By fitting each set of three adjacent points to a parabola 
        # y = at^2 + bt + c, this method approximates the derivative as y' = 2at + b. a and
        # b were computationally determined using the formulae:
        #
        #   a = ((y_1 - y_2)*(t_3 - t_2) + (y_1 - y_3)*(t_1 - t_2)) /
        #       ((t_1 - t_2)*(t_2 - t_3)*(t_3 - t_1))
        # 
        #   b = ((y_1 - y_2) - a*(t_1^2 - t_2^2)) / (t_1 - t_2)
        #
        # Then, this method calculates the RMS error and saves the calculated derivative to 
        # 'paraboladeriv.txt'.
        #
        # @param self   the instance of ParabolaDerivative producing the calculation
        # @return       the calculated parabola derivative ordered pairs and RMS error
        #'''
    def calcDeriv(self):
        intervalSize = (self.rangeR - self.rangeL) / (self.numPoints - 1)
        quadDerivValues = np.zeros((2, self.numPoints))
        rmsError = 0.0
        curV = self.funcVals[:,:3]
        a = ((curV[1][0] - curV[1][1])*(curV[0][2] - curV[0][1]))
        a += (curV[1][1] - curV[1][2])*(curV[0][0] - curV[0][1])
        a /= (curV[0][0] - curV[0][1])*(curV[0][1] - curV[0][2])*(curV[0][2] - curV[0][1])
        b = ((curV[1][0] - curV[1][1]) - a*(curV[0][0]**2 - curV[0][1]**2))
        b /= (curV[0][0] - curV[0][1])
        quadDerivValues[0][0] = self.rangeL
        quadDerivValues[1][0] = 2*a*quadDerivValues[0][0] + b
        rmsError += (quadDerivValues[1][0] - self.derivVals[1][0])**2

        for i in range(1, self.numPoints - 1):
            curV = self.funcVals[:, i-1: i + 2]
            a = ((curV[1][0] - curV[1][1])*(curV[0][2] - curV[0][1]))
            a += (curV[1][1] - curV[1][2])*(curV[0][0] - curV[0][1])
            a /= (curV[0][0] - curV[0][1])*(curV[0][1] - curV[0][2])*(curV[0][2] - curV[0][1])
            b = ((curV[1][0] - curV[1][1]) - a*(curV[0][0]**2 - curV[0][1]**2))
            b /= (curV[0][0] - curV[0][1])
            quadDerivValues[0][i] = curV[0][1]
            quadDerivValues[1][i] = 2*a*quadDerivValues[0][i] + b
            rmsError += (quadDerivValues[1][i] - self.derivVals[1][2*i])**2
            # End of for loop
        
        quadDerivValues[0][self.numPoints - 1] = self.rangeR
        quadDerivValues[1][self.numPoints - 1] = 2*a*quadDerivValues[0][self.numPoints - 1] + b
        rmsError += (quadDerivValues[1][self.numPoints-1] - self.derivVals[1][self.numPoints-2])**2
        rmsError = math.sqrt(rmsError/self.numPoints)
        np.savetxt('paraboladeriv.txt', np.column_stack(quadDerivValues), fmt = '(%f, %f)')
        return quadDerivValues, rmsError
        # End of method calcDeriv
    # End of class ParabolaDerivative

''' # Class FivePointStencil defines a functionality for calculating the derivatives of
    # (t, e^(-t^2)) by using the five-point stencil algorithm, which is calculated using
    # third-order Taylor series approximations of the function. Finally, this class also 
    # provides functionalities for evaluating the effectiveness of various derivative 
    # calculations by evaluating the RMS error.
    #'''
class FivePointStencil:
    ''' # Method __init__ defines the constructor for the range in which the function is calculated
        # along with the number of points to sample in the range. Then, the method calculates
        # the empirical values of the derivative of e^(-t^2) and its derivative -2te^(-t^2) at a 
        # uniform sampling of points in the range provided using the CalcFunc class.
        # 
        # @param self       the instance of the FivePointStencil class created
        # @param rangeL     the left boundary of the range
        # @param rangeR     the right boundary of the range
        # @param numPoints  the number of points to sample in the range
        # @postcondition    the instance of FivePointStencil has been instantiated with proper
        #                   boundaries and empirical values of e^(-t^2) and its derivative
        #'''
    def __init__(self, rangeL, rangeR, numPoints):
        self.func = CalcFunc(rangeR, rangeL, numPoints)
        self.rangeL = rangeL
        self.rangeR = rangeR
        self.numPoints = numPoints
        self.funcVals = self.func.calcFuncVals()
        self.derivVals = self.func.calcDerivativeVals()
        # End of method __init__
    
    ''' # Method calcDeriv calculates the derivative of (t, e^(-t^2)) using the five-point stencil
        # technique. By fitting each set of five adjacent points using third-order Taylor series,
        # f'(t) can be approximated as:
        #
        #   f'(t) = (-f(t + 2h) + 8f(t + h) - 8f(t - h) + f(t - 2h)) / 12h
        #
        # Then, this method calculates the RMS error and saves the calculated derivative to 
        # 'fivepointstencil.txt'.
        #
        # @param self   the instance of FivePointStencil producing the calculation
        # @return       the calculated five-point-stencil derivative ordered pairs and RMS error
        #'''
    def calcDeriv(self):
        intervalSize = (self.rangeR - self.rangeL) / (self.numPoints - 1)
        calcDerivVals = np.zeros((2, self.numPoints - 4))
        rmsError = 0.0
        for i in range(self.numPoints - 4):
            calcDerivVals[0][i] = self.rangeL + (i + 2)*intervalSize
            calcDerivVals[1][i] = self.funcVals[1][i] + -8*self.funcVals[1][i + 1]
            calcDerivVals[1][i] += 8*self.funcVals[1][i + 3] + -1*self.funcVals[1][i + 4]
            calcDerivVals[1][i] /= 12*intervalSize
            rmsError += (calcDerivVals[1][i] - self.derivVals[1][2*i + 4])**2
        
        rmsError = math.sqrt(rmsError/(self.numPoints - 4))
        np.savetxt('fivepointstencil.txt', np.column_stack(calcDerivVals), fmt = '(%f, %f)')
        return calcDerivVals, rmsError
        # End of method calcDeriv
    # End of class FivePointStencil

''' # Method main tests and plots the various derivative classes, first plotting the function
    # y = e^(-t^2) and then comparing the optimal derivative y' = -2te^(-t^2) to the various
    # functions and RMS errors computed by the four different derivative variants.
    #
    # @param arguments      Arguments from the command line, specifically the number of points
    #                       to sample
    # @postcondition        Five plots have been generated, first with the original function and
    #                       the following four with comparisons to the various derivative classes
    #'''
def main(arguments):
    numPoints = arguments.num_points

    # Plot e^(-t^2)
    c = CalcFunc(-10, 10, numPoints)
    vals = c.calcFuncVals()
    c.saveFuncVals()
    c.saveDerivVals()
    plt.figure()
    plt.plot(vals[0], vals[1])
    plt.figure()

    # Plot and evaluate various slope derivatives
    s = SlopeDerivative(-10, 10, numPoints)
    derivSource = c.calcDerivativeVals()
    leftDeriv, leftRMS = s.calcLeftDeriv()
    print("Left-sided derivative RMS Error: {}".format(leftRMS))
    midDeriv, midRMS = s.calcMidDeriv()
    print("Mid-sided derivative RMS Error: {}".format(midRMS))
    rightDeriv, rightRMS = s.calcRightDeriv()
    print("Right-sided derivative RMS Error: {}".format(rightRMS))
    plt.plot(leftDeriv[0], leftDeriv[1], 'b')
    plt.plot(rightDeriv[0], rightDeriv[1], 'r')
    plt.plot(midDeriv[0], midDeriv[1], 'g')
    plt.plot(derivSource[0], derivSource[1], 'black')
    plt.figure()

    # Plot and evaluate various three-point derivatives
    tp = ThreePointDerivative(-10, 10, numPoints)
    threePointDeriv, threePointRMS = tp.calcDeriv()
    print("Three-point derivative RMS Error: {}".format(threePointRMS))
    plt.plot(derivSource[0], derivSource[1], 'black')
    plt.plot(threePointDeriv[0], threePointDeriv[1], 'g')
    plt.figure()

    # Plot and evaluate parabola derivatives
    qd = ParabolaDerivative(-10, 10, numPoints)
    quadDeriv, quadRMS = qd.calcDeriv()
    print("Quadratic derivative RMS Error: {}".format(quadRMS))
    plt.plot(derivSource[0], derivSource[1], 'black')
    plt.plot(quadDeriv[0], quadDeriv[1], 'r')
    plt.figure()
    
    # Plot and evaluate five-point stencil derivatives
    fps = FivePointStencil(-10, 10, numPoints)
    fivePointDeriv, fivePointRMS = fps.calcDeriv()
    print("Five Point Stencil derivative RMS Error: {}".format(fivePointRMS))
    plt.plot(derivSource[0], derivSource[1], 'black')
    plt.plot(fivePointDeriv[0], fivePointDeriv[1], 'b')
    plt.show()
    # End of method main

'''#
   # This statement functions to run the main function by default. This statement also encodes
   # a command line argument to accept the number of points to sample.
   #'''
if (__name__ == "__main__"):
    commandLineArgs = argparse.ArgumentParser()
    commandLineArgs.add_argument("--num_points", type = int, default = 100)
    arguments = commandLineArgs.parse_args() 
    main(arguments)