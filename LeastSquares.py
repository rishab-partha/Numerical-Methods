import argparse
import matplotlib.pyplot as plt
import numpy as np

STEP_SIZE = 1e-7
FPS_STEPS = np.array([-2*STEP_SIZE, -1*STEP_SIZE, STEP_SIZE, 2*STEP_SIZE])
FPS_2D = np.reshape(FPS_STEPS, (1, 4))
FPS_WEIGHTS = np.array([1.0, -8.0, 8.0, -1.0])
LAMBDA = None

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

def rmse(out, pred):
    return np.sqrt(np.sum((out-pred)**2) / out.shape[0])

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

def gradient(inputs, outputs, q, func):
    partialDerivs = partial(inputs, q, func)
    curPred = func(inputs, q)
    diff = np.reshape(outputs - curPred, (outputs.shape[0], 1))
    gradient = LAMBDA*np.sum(partialDerivs*diff, axis = 0)
    return gradient
    # End of method gradient

def oneIter(inp, out, q, func):
    pred = func(inp, q)
    error = rmse(out, pred)
    grad = gradient(inp, out, q, func)
    q += grad
    newPred = func(inp, q)
    newError = rmse(out, newPred)
    if newError > error:
        LAMBDA = LAMBDA / 2
        q -= grad
    else:
        LAMBDA = LAMBDA * 1.1
    # End of method oneIter



partialDeriv = partial(np.array([4, 3, 2, 1, 5]), np.array([1, 2, 3, 4], dtype = float), cubic)
print(partialDeriv)

grad = gradient(np.array([4.0, 3.0, 2.0, 1.0, 5.0]), np.array([113, 56, 24, 11, 196]), 
    np.array([1.0, 2.0, 3.0, 4.0]), cubic)
print(grad)
