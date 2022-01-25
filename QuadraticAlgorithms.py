import argparse
import math

class QuadraticAlgorithms:
    def __init__(self):
        return

    def calcRoots(self, a, b, c):
        rootOne = (-b + math.sqrt(b**2 - 4*a*c))/(2*a)
        rootTwo = (-b - math.sqrt(b**2 - 4*a*c))/(2*a)
        return rootOne, rootTwo
    
    def printRoots(self, a, b, c):
        rootOne, rootTwo = self.calcRoots(a, b, c)
        print("The roots of {}x^2 + {}x + {}: {}, {}".format(a, b, c, rootOne, rootTwo))

    def testQuadPrecisionLimits(self):
        flag = True
        a, b, c = 1, 2.0, 1
        while flag:
            rootOne, rootTwo = self.calcRoots(a, b, c)
            if rootOne == 0.0 or rootTwo == 0.0:
                flag = False
                self.printRoots(a, b, c)
                realRootOne = c/(a*rootTwo)
                print("The real roots of {}x^2 + {}x + {}: {}, {}".format(a, b, c, realRootOne, rootTwo))
            
            b = b * 1.1
    
def main(arguments):
    tester = QuadraticAlgorithms()
    tester.printRoots(1, 2.1, 1)
    tester.testQuadPrecisionLimits()

if (__name__ == "__main__"):
   commandLineArgs = argparse.ArgumentParser()
   arguments = commandLineArgs.parse_args() 
   main(arguments)
    