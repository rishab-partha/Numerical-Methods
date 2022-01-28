import argparse
import math

'''# Class QuadraticAlgorithms defines a functionality for calculating the roots of a given
   # quadratic equation of the form: ax^2 + bx + c. This class also provides functionalities
   # for printing the roots and evaluating the quadratic at the roots, which are used to test
   # the limits of double precision in the situation of quadratic computation.
   #
   # @author Rishab Parthasarathy
   # @version 01.28.2022
   #'''
class QuadraticAlgorithms:

   '''# Method __init__ defines the constructor for the QuadraticAlgorithms class with no
      # instance variables.
      # 
      # @param self     The instance of the QuadraticAlgorithms class created
      # @postcondition  The instance of the QuadraticAlgorithms class has been created
      #'''
   def __init__(self):
      return
      # End of method __init__

   '''# Method calcRoots calculates the roots of a quadratic of the form ax^2 + bx + c 
      # using the quadratic formula.
      # 
      # @param self     The instance of the QuadraticAlgorithms class that calculates the roots
      # @param a        The coefficient of the second order term
      # @param b        The coefficient of the first order term
      # @param c        The coefficient of the zeroth order term
      # @precondition   The quadratic defined has only real roots
      # @return         The two real roots of the quadratic
      # @throws         ValueError: math domain error if the quadratic has imaginary roots
      #'''
   def calcRoots(self, a, b, c):
      rootOne = (-b + math.sqrt(b**2 - 4*a*c))/(2*a)
      rootTwo = (-b - math.sqrt(b**2 - 4*a*c))/(2*a)
      return rootOne, rootTwo
      # End of method calcRoots
    
   '''# Method printRoots prints the roots of a quadratic of the form ax^2 + bx + c, and then
      # prints the value of the quadratic evaluated at those roots.
      # 
      # @param self     The instance of the QuadraticAlgorithms class that prints the roots
      # @param a        The coefficient of the second order term
      # @param b        The coefficient of the first order term
      # @param c        The coefficient of the zeroth order term
      # @precondition   The quadratic defined has only real roots
      # @postcondition  The method prints the roots and the quadratic evaluated at the roots
      # @throws         ValueError: math domain error if the quadratic has imaginary roots
      #'''
   def printRoots(self, a, b, c):
      rootOne, rootTwo = self.calcRoots(a, b, c)
      evalRootOne = self.evalFunc(a, b, c, rootOne)
      evalRootTwo = self.evalFunc(a, b, c, rootTwo)
      print("The roots of {}x^2 + {}x + {}: {}, {}".format(a, b, c, rootOne, rootTwo))
      print("These roots evaluated: {}, {}\n".format(evalRootOne, evalRootTwo))
      # End of method printRoots
        
   '''# Method evalFunc evaluates a quadratic of the form ax^2 + bx + c at a given value.
      # 
      # @param self     The instance of the QuadraticAlgorithms class that evaluates the quadratic
      # @param a        The coefficient of the second order term
      # @param b        The coefficient of the first order term
      # @param c        The coefficient of the zeroth order term
      # @param val      The value at which to evaluate the quadratic
      # @return         The quadratic evaluated at the given value
      #'''
   def evalFunc(self, a, b, c, val):
      return a*val**2 + b*val + c
      # End of method evalFunc

   '''# Method testQuadPrecisionLimits calculates the limits of the quadratic formula based on
      # the double precision of Python. Beginning with the quadratic x^2 + 2x + 1, this method
      # continually doubles the coefficient of the linear term until one of the roots produced
      # is truncated to zero. Then, the incorrect roots are evaluated, recalculated using
      # r_1 = c/(a*r_2), and printed/evaluated once more.
      # 
      # @param self     The instance of the QuadraticAlgorithms class that calculates the roots
      # @postcondition  The method has printed and evaluated both the limiting values/evaluations
      #                 of the coefficients/roots and the fixed values/evaluations of the
      #                 coefficients/roots 
      #'''
   def testQuadPrecisionLimits(self):
      flag = True
      a, b, c = 1, 2.0, 1
      while flag:
         rootOne, rootTwo = self.calcRoots(a, b, c)

         if rootOne == 0.0 or rootTwo == 0.0:
            flag = False
            self.printRoots(a, b, c)
            realRootOne = c/(a*rootTwo)
            evalRootOne = self.evalFunc(a, b, c, realRootOne)
            evalRootTwo = self.evalFunc(a, b, c, rootTwo)
            print("The real roots of {}x^2 + {}x + {}: {}, {}".format(a, b, c, realRootOne, rootTwo))
            print("These roots evaluated: {}, {}\n".format(evalRootOne, evalRootTwo))

         b = b * 2.0
         # End of while loop
      # End of method testQuadPrecisionLimits
   # End of class QuadraticAlgorithms

'''# Method main instantiates an instance of the QuadraticAlgorithms class, test the printing of
   # roots for x^2 + 2.1x + 1, and tests the quadratic precision limits.
   #
   # @param arguments   Arguments from the command line
   #'''
def main(arguments):
   tester = QuadraticAlgorithms()
   tester.printRoots(1, 2.1, 1)
   tester.testQuadPrecisionLimits()
   # End of method main

'''#
   # This statement functions to run the main function by default. This statement also encodes
   # a command line argument for future compatibility with any command line prompts.
   #'''
if (__name__ == "__main__"):
   commandLineArgs = argparse.ArgumentParser()
   arguments = commandLineArgs.parse_args() 
   main(arguments)
    