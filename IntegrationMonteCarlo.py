''' # This code establishes how to approximate the area of a n-dimensional sphere by performing
    # a Monte Carlo integral approximation.
    #
    # @author Rishab Parthasarathy
    # @version 04.06.2022
    #'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

''' # Method randomHistogram plots a histogram for the given RNG over the range [-2, 2] for a given
    # number of equally sized bins and a given number of points.
    #
    # @param rng        the RNG to test
    # @param numBins    the number of equally sized bins to use
    # @param numPoints  the number of points to test
    # @postcondition    the RNG histogram is plotted
    # '''
def randomHistogram(rng, numBins = 1000, numPoints = 2000000):
    data = 4*rng.random(numPoints) - 2
    plt.figure()
    plt.hist(x=data, bins = numBins, range = (-2, 2), color='#0504aa')

''' # Method actVal calculates the actual proportion of the sphere volume in n-dimensions to the 
    # n-cube using the formula
    #               2*(2*pi)^((n-1)/2) / (n!! * 2^n)
    # where n!! represents the double factorial with a scale factor of sqrt(2 / pi) for even n.
    #
    # @param dim    the number of dimensions of the sphere
    # @return       the calculated empirical volume ratio
    # '''
def actVal(dim):
    ans = 2.0*((2.0*np.pi)**((dim - 1)/2)) / ((2.0)**(dim) * sp.factorial2(dim))
    if dim % 2 == 0:
        ans *= np.sqrt(np.pi/2)
    return ans

''' # Method oneEpoch performs one epoch of the Monte Carlo simulation by randomly querying points
    # inside the n-cube.
    # 
    # @param rng        the RNG used in the Monte Carlo
    # @param dim        the number of dimensions
    # @param epochSize  the number of steps in each epoch
    # @return           the number of points in this epoch falling within the n-sphere
    # '''
def oneEpoch(rng, dim, epochSize):
    data = 2*rng.random((epochSize, dim)) - 1
    return np.count_nonzero(np.sum(data**2, axis = 1) <= 1)

''' # Method train trains the Monte Carlo simulation by randomly querying points
    # inside the n-cube. This method also plots the integral value and error against
    # epoch.
    # 
    # @param rng        the RNG used in the Monte Carlo
    # @param dim        the number of dimensions
    # @param epochSize  the number of steps in each epoch
    # @param numEpochs  the number of epochs to train for
    # @postcondition    the integral value and error are plotted against epoch #
    # @return           the final calculated volume ratio and the empirical value
    # '''
def train(rng, dim, epochSize, numEpochs):
    epochNums = np.linspace(1, numEpochs, num = numEpochs)
    intValue = np.zeros(numEpochs)
    act = actVal(dim)
    cntr = 0
    for i in range(numEpochs):
        cntr += oneEpoch(rng, dim, epochSize)
        intValue[i] = cntr/((i +1)*epochSize)
    plt.figure()
    plt.plot(epochNums, intValue, 'b')
    plt.ylabel("Integral Value")
    plt.xlabel("Epoch #")
    plt.figure()
    plt.plot(np.log(epochNums), np.log(np.abs(intValue - act)), 'g')
    plt.ylabel("Error")
    plt.xlabel("Log Scale of Epoch #")
    return cntr/(epochSize*numEpochs), act

''' # Method main tests the Monte Carlo simulation based on command line arguments.
    # 
    # @param arguments      arguments from the command line
    # @postcondition        the Monte Carlo has run and all appropriate graphs have been plotted.
    # '''
def main(arguments):
    dim = arguments.dim
    epochSize = arguments.epochsize
    numEpochs = arguments.epochs
    rng = np.random.default_rng()
    randomHistogram(rng)
    intVal, act = train(rng, dim, epochSize, numEpochs)
    print("Empirical Value: {}".format(act))
    print("Calculated Value: {}".format(intVal))
    plt.show()

''' #
    # This statement functions to run the main function by default. This statement also encodes
    # a command line argument to accept the config file.
    # '''
if (__name__ == "__main__"):
    commandLineArgs = argparse.ArgumentParser()
    commandLineArgs.add_argument("--dim", type = int, default = 2)
    commandLineArgs.add_argument("--epochsize", type = int, default = 40)
    commandLineArgs.add_argument("--epochs", type = int, default = 10000)
    arguments = commandLineArgs.parse_args()
    main(arguments)