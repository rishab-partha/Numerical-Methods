''' # This code establishes how to fit a function to a set of datapoints using a least-squares
    # approach with gradient descent. 
    #
    # @author Rishab Parthasarathy
    # @version 04.05.2022
    #'''

import argparse
import matplotlib.pyplot as plt
import numpy as np

# Define constants
STEP_SIZE = 1e-7
FPS_STEPS = np.array([-2*STEP_SIZE, -1*STEP_SIZE, STEP_SIZE, 2*STEP_SIZE])
FPS_2D = np.reshape(FPS_STEPS, (1, 4))
FPS_WEIGHTS = np.array([1.0, -8.0, 8.0, -1.0])

''' # Method linear defines a linear function of the form 
    #       f(x) = q_2 x + q_3.
    #
    # This method also processes a matrix of inputs and parameters, allowing
    # multiple tests to be performed simultaneously.
    # 
    # @param inputs the matrix of input values to test
    # @param q      the matrix of parameters to test
    # @return       the matrix w/ values of f(x) for all sets of parameters/input
    # '''
def linear(inputs, q):
    return q[2]*(inputs) + q[3]
    # End of method linear

''' # Method quadratic defines a quadratic function of the form 
    #       f(x) = q_1 x^2 + q_2 x + q_3.
    #
    # This method also processes a matrix of inputs and parameters, allowing
    # multiple tests to be performed simultaneously.
    # 
    # @param inputs the matrix of input values to test
    # @param q      the matrix of parameters to test
    # @return       the matrix w/ values of f(x) for all sets of parameters/input
    # '''
def quadratic(inputs, q):
    return q[1]*(inputs**2) + q[2]*(inputs) + q[3]
    # End of method quadratic

''' # Method cubic defines a cubic function of the form 
    #       f(x) = q_0 x^3 + q_1 x^2 + q_2 x + q_3.
    #
    # This method also processes a matrix of inputs and parameters, allowing
    # multiple tests to be performed simultaneously.
    # 
    # @param inputs the matrix of input values to test
    # @param q      the matrix of parameters to test
    # @return       the matrix w/ values of f(x) for all sets of parameters/input
    # '''
def cubic(inputs, q):
    return q[0]*(inputs**3) + q[1]*(inputs**2) + q[2]*(inputs) + q[3]
    # End of method cubic

''' # Method gaussian defines a gaussian function of the form 
    #       f(x) = q_0^2 e^(-(x - q[1])^2/q[2]^2) + q[3]^2
    #
    # This method also processes a matrix of inputs and parameters, allowing
    # multiple tests to be performed simultaneously.
    # 
    # @param inputs the matrix of input values to test
    # @param q      the matrix of parameters to test
    # @return       the matrix w/ values of f(x) for all sets of parameters/input
    # '''
def gaussian(inputs, q):
    return (q[0]**2)*np.exp(-1*((inputs-q[1])**2)/q[2]**2) + q[3]**2
    # End of method gaussian

''' # Method sine defines a sine function of the form 
    #       f(x) = q_0 sin(q[1] x + q[2]) + q[3].
    #
    # This method also processes a matrix of inputs and parameters, allowing
    # multiple tests to be performed simultaneously.
    # 
    # @param inputs the matrix of input values to test
    # @param q      the matrix of parameters to test
    # @return       the matrix w/ values of f(x) for all sets of parameters/input
    # '''
def sine(inputs, q):
    return q[0]*np.sin(q[1]*inputs + q[2]) + q[3]
    # End of method sine

''' # Method error calculates the error function:
    #       Error = 0.5* sum((y - pred_y)^2)
    # 
    # @param out    the true values of y
    # @param pred   predicted values of y
    # @return       the value of the error fu
    # '''
def error(out, pred):
    return 0.5*np.sum((out - pred)**2)
    # End of method error

''' # Method rmse calculates the root-mean-squared error function:
    #       RMSE = sqrt(sum((y - pred_y)^2)/N)
    # 
    # @param out    the true values of y
    # @param pred   predicted values of y
    # @return       the value of the rms error function
    # '''
def rmse(out, pred):
    return np.sqrt(np.sum((out-pred)**2) / out.shape[0])
    # End of method rmse

''' # Method partial calculates the partial derivatives relative to the parameters q using
    # a matrix implementation of the five-point stencil.
    # 
    # @param inputs the input values to the function
    # @param q      the current values of the parameters
    # @param func   the function being fitted
    # @return       the values of the partial derivatives relative to the parameters
    # '''
def partial(inputs, q, func):
    derivLocs = np.tile(q, (4*q.shape[0], 1)).T
    fps_tot = np.concatenate((np.pad(FPS_2D, ((0,3), (0,0))), np.pad(FPS_2D, ((1,2), (0,0))), 
        np.pad(FPS_2D, ((2,1), (0,0))), np.pad(FPS_2D, ((3,0), (0,0)))), axis = 1)
    derivLocs += fps_tot
    derivLocs = np.reshape(derivLocs, ((q.shape[0], 4*q.shape[0], 1)))
    calcValues = np.reshape(func(inputs, derivLocs).T, (inputs.shape[0], q.shape[0], 4))
    derivs = np.matmul(calcValues, FPS_WEIGHTS)
    return derivs # First dim is input loc, second dim is param #
    # End of method partial

''' # Method gradient calculates the gradient relative to the parameters q using
    # the partial derivatives and the current predictions.
    # 
    # @param inputs     the input values to the function
    # @param outputs    the true output values
    # @param q          the current values of the parameters
    # @param func       the function being fitted
    # @param lam        the learning factor
    # @return           the value of the gradient relative to the parameters
    # '''
def gradient(inputs, outputs, q, func, lam):
    partialDerivs = partial(inputs, q, func)
    curPred = func(inputs, q)
    diff = np.reshape(outputs - curPred, (outputs.shape[0], 1))
    gradient = lam*np.sum(partialDerivs*diff, axis = 0)
    return gradient
    # End of method gradient

''' # Method oneIter makes one iteration of modifying the parameters and learning factors
    # using the calculated gradients with gradient descent. If the error decreases, lambda
    # is increased by 1.1; otherwise, lambda is halved.
    # 
    # @param inputs     the input values to the function
    # @param outputs    the true output values
    # @param q          the current values of the parameters
    # @param func       the function being fitted
    # @param lam        the learning factor
    # @return           the new values of the learning factor and parameters
    # '''
def oneIter(inp, out, q, func, lam):
    pred = func(inp, q)
    error = rmse(out, pred)
    grad = gradient(inp, out, q, func, lam)
    q += grad
    newPred = func(inp, q)
    newError = rmse(out, newPred)
    if newError > error:
        lam = lam / 2
        q -= grad
    else:
        lam = lam * 1.1
        error = newError
    return lam, error
    # End of method oneIter

''' # Method train trains the least squares model for a given number of iterations or until
    # an error threshold is hit, printing and updating errors/graphs each epoch.
    # 
    # @param inputs     the input values to the function
    # @param outputs    the true output values
    # @param q          the current values of the parameters
    # @param func       the function being fitted
    # @param lam        the learning factor
    # @param thres      the error threshold
    # @param maxIters   the maximum number of iterations
    # @param epochIters the number of iterations in an epoch
    # @postcondition    the least squares model has reached max iterations or the error threshold
    # '''
def train(inp, out, q, func, lam = 1e-1, thres = 1e-2, maxIters = 100000, epochIters = 1000):
    pred = func(inp, q)
    error = rmse(out, pred)
    iters = 0
    maxInp = np.max(inp)
    minInp = np.min(inp)
    funcArea = np.linspace(minInp, maxInp, 1000)
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sourceGraph, predGraph = ax.plot(inp, out, 'bo', funcArea, func(funcArea, q), 'g-')
    fig.canvas.draw()
    fig.canvas.flush_events()
    while iters < maxIters and error > thres:
        lam, error = oneIter(inp, out, q, func, lam)
        iters += 1
        if iters % epochIters == 0:
            print("Epoch: {}  Error: {}".format(iters, error))
            predGraph.set_ydata(func(funcArea, q))
            fig.canvas.draw()
            fig.canvas.flush_events()
    print()
    if (iters == maxIters):
        print("Terminated at max # of iterations.")
    else:
        print("Stopped at threshold error:")
    predGraph.set_ydata(func(funcArea, q))
    print("Epochs: {}".format(iters))
    print("Final error: {}".format(error))
    print("Final parameters: {}".format(q))
    plt.ioff()
    plt.show()
    # End of method train

''' # Method readInput reads input data of with space separated values of x and y on each line.
    # 
    # @param inputFile  the input file path
    # @return           the inputs/outputs read from the file
    # '''
def readInput(inputFile):
    fileInput = open(inputFile, 'r')
    allData = fileInput.readlines()
    inp = np.zeros(len(allData))
    out = np.zeros(len(allData))
    for i in range(len(allData)):
        inp[i], out[i] = [float(val) for val in allData[i].split()]
    return inp, out
    # End of method readInput

''' # Method readConfig reads the config file with the function type, the learning factor,
    # error threshold, max iterations, initial parameters, and path of input file.
    # 
    # @param config     the config file path
    # @return           the function type, learning factor, error threshold, max iterations,
    #                   initial parameters, input/output data
    # '''
def readConfig(config):
    fileConfig = open(config, 'r')
    func = [str(val) for val in fileConfig.readline().split()][0]
    lam, thres, numIters = [float(val) for val in fileConfig.readline().split()]
    q = np.asarray([float(val) for val in fileConfig.readline().split()])
    inp, out = readInput([str(val) for val in fileConfig.readline().split()][0])
    return func, lam, thres, numIters, q, inp, out
    # End of method readConfig

''' # Method main runs the least squares model on a given config file.
    #
    # @param config     the config file path
    # @postcondition    the model has either run or has stated that the func was not recognized
    # '''
def main(config):
    func, lam, thres, numIters, q, inp, out = readConfig(config)
    if (func == 'cubic'):
        train(inp, out, q, cubic, lam, thres, numIters)
    elif (func == 'gaussian'):
        train(inp, out, q, gaussian, lam, thres, numIters)
    elif (func == 'sine'):
        train(inp, out, q, sine, lam, thres, numIters)
    elif (func == 'linear'):
        train(inp, out, q, linear, lam, thres, numIters)
    elif (func == 'quadratic'):
        train(inp, out, q, quadratic, lam, thres, numIters)
    else:
        print("Function not recognized!")
    
'''#
   # This statement functions to run the main function by default. This statement also encodes
   # a command line argument to accept the config file.
   #'''
if (__name__ == "__main__"):
    commandLineArgs = argparse.ArgumentParser()
    commandLineArgs.add_argument("--config", type = str, default = "configLeastSquares.txt")
    arguments = commandLineArgs.parse_args()
    main(arguments.config)
