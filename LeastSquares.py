import argparse
from pickle import NONE
import matplotlib.pyplot as plt
import numpy as np

STEP_SIZE = 1e-7
FPS_STEPS = np.array([-2*STEP_SIZE, -1*STEP_SIZE, STEP_SIZE, 2*STEP_SIZE])
FPS_2D = np.reshape(FPS_STEPS, (1, 4))
FPS_WEIGHTS = np.array([1.0, -8.0, 8.0, -1.0])

def cubic(inputs, q):
    return q[0]*(inputs**3) + q[1]*(inputs**2) + q[2]*(inputs) + q[3]
    # End of method cubic

def gaussian(inputs, q):
    return (q[0]**2)*np.exp(-1*((inputs-q[1])**2)/q[2]**2) + q[3]**2
    # End of method gaussian

def sine(inputs, q):
    return q[0]*np.sin(q[1]*inputs + q[2]) + q[3]
    # End of method sine

def error(out, pred):
    return 0.5*np.sum((out - pred)**2)
    # End of method error

def rmse(out, pred):
    return np.sqrt(np.sum((out-pred)**2) / out.shape[0])
    # End of method rmse

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

def gradient(inputs, outputs, q, func, lam):
    partialDerivs = partial(inputs, q, func)
    curPred = func(inputs, q)
    diff = np.reshape(outputs - curPred, (outputs.shape[0], 1))
    gradient = lam*np.sum(partialDerivs*diff, axis = 0)
    return gradient
    # End of method gradient

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
    print("Epochs: {}".format(iters))
    print("Final error: {}".format(error))
    plt.ioff()
    plt.show()
    # End of method train

def readInput(inputFile):
    fileInput = open(inputFile, 'r')
    allData = fileInput.readlines()
    inp = np.zeros(len(allData))
    out = np.zeros(len(allData))
    for i in range(len(allData)):
        inp[i], out[i] = [float(val) for val in allData[i].split()]
    return inp, out
    # End of method readInput


def readConfig(config):
    fileConfig = open(config, 'r')
    func = [str(val) for val in fileConfig.readline().split()][0]
    lam, thres, numIters = [float(val) for val in fileConfig.readline().split()]
    q = np.asarray([float(val) for val in fileConfig.readline().split()])
    inp, out = readInput([str(val) for val in fileConfig.readline().split()][0])
    return func, lam, thres, numIters, q, inp, out
    # End of method readConfig

def main(config):
    func, lam, thres, numIters, q, inp, out = readConfig(config)
    if (func == 'cubic'):
        train(inp, out, q, cubic, lam, thres, numIters)
    elif (func == 'gaussian'):
        train(inp, out, q, gaussian, lam, thres, numIters)
    elif (func == 'sine'):
        train(inp, out, q, sine, lam, thres, numIters)
    else:
        print("Function not recognized!")
    
if (__name__ == "__main__"):
    commandLineArgs = argparse.ArgumentParser()
    commandLineArgs.add_argument("--config", type = str, default = "configLeastSquares.txt")
    arguments = commandLineArgs.parse_args()
    main(arguments.config)
