''' # This code establishes how to approximate the motion of a pendulum by solving the differential
    # equation using Euler's method and the Euler-Cromer method.
    #
    # @author Rishab Parthasarathy
    # @version 04.27.2022
    # '''
import argparse
import matplotlib.pyplot as plt
import numpy as np

''' # Method thetas calculates the angular displacement of an SHM pendulum at given time points by
    #  
    #       theta(t) = A*cos(Omega*t + phi)
    # 
    # @param inp        the input time points
    # @param A          the amplitude of the oscillation
    # @param Omega      the frequency of oscillations, equal to sqrt(g/l)
    # @param phi        the phase for the beginning of oscillation
    # @return           the value of the angular displacement at all time points
    # '''
def thetas(inp, A, Omega, phi):
    return A*np.cos(Omega*inp + phi)

''' # Method omegas calculates the angular velocity of an SHM pendulum at given time points by
    #  
    #       theta(t) = -A*Omega*sin(Omega*t + phi)
    # 
    # @param inp        the input time points
    # @param A          the amplitude of the oscillation
    # @param Omega      the frequency of oscillations, equal to sqrt(g/l)
    # @param phi        the phase for the beginning of oscillation
    # @return           the value of the angular velocity at all time points
    # '''
def omegas(inp, A, Omega, phi):
    return -A*Omega*np.sin(Omega*inp + phi)

''' # Method euler propagates the approximation of the pendulum differential equation by
    #
    #       omega(t + Delta t) = omega(t) - (g/l) * sin(theta(t))*Delta t
    #       theta(t + Delta t) = theta(t) + omega(t) * Delta t
    # 
    # @param omega      the starting value of angular velocity
    # @param theta      the starting value of angular displacement
    # @param Omega      the frequency of oscillations, equal to sqrt(g/l)
    # @param delta      the change in time over each iteration
    # @param n          the number of time steps
    # @return           the approximated angular displacement and velocity rel. to time
    # '''
def euler(omega, theta, Omega, delta = 0.001, n = 100000):
    omegas = np.zeros(n + 1)
    thetas = np.zeros(n + 1)
    omegas[0] = omega
    thetas[0] = theta
    for i in range(1, n + 1):
        omegas[i] = omegas[i - 1] - (Omega**2)*np.sin(thetas[i - 1])*delta
        thetas[i] = thetas[i - 1] + omegas[i - 1] * delta

    return omegas, thetas

''' # Method eulerCromer propagates the approximation of the pendulum differential equation by
    #
    #       omega(t + Delta t) = omega(t) - (g/l) * sin(theta(t))*Delta t
    #       theta(t + Delta t) = theta(t) + omega(t + Delta t) * Delta t
    # 
    # @param omega      the starting value of angular velocity
    # @param theta      the starting value of angular displacement
    # @param Omega      the frequency of oscillations, equal to sqrt(g/l)
    # @param delta      the change in time over each iteration
    # @param n          the number of time steps
    # @return           the approximated angular displacement and velocity rel. to time
    # '''
def eulerCromer(omega, theta, Omega, delta = 0.001, n = 100000):
    omegas = np.zeros(n + 1)
    thetas = np.zeros(n + 1)
    omegas[0] = omega
    thetas[0] = theta
    for i in range(1, n + 1):
        omegas[i] = omegas[i - 1] - (Omega**2)*np.sin(thetas[i - 1])*delta
        thetas[i] = thetas[i - 1] + omegas[i] * delta

    return omegas, thetas

''' # Method main tests the Euler's method and Euler-Cromer solutions against the SHM solution.
    # 
    # @param A          the amplitude of oscillations
    # @param Omega      the frequency of oscillations, equal to sqrt(g/l)
    # @param phi        the starting phase
    # @param delta      the change in time over each iteration
    # @param n          the number of time steps
    # @postcondition    compare the Euler and Euler-Cromer solutions to SHM with plots
    # '''
def main(A, Omega, phi, delta, n):
    xAxis = np.linspace(0, n*delta, n+1)
    omegaE, thetaE = euler(-A*Omega*np.sin(phi), A*np.cos(phi), Omega, delta, n)
    omegaEC, thetaEC = eulerCromer(-A*Omega*np.sin(phi), A*np.cos(phi), Omega, delta, n)
    plt.figure()
    plt.plot(xAxis, thetas(xAxis, A, Omega, phi), 'r')
    plt.plot(xAxis, thetaE, 'g')
    plt.plot(xAxis, thetaEC, 'b')
    plt.figure()
    plt.plot(xAxis, omegas(xAxis, A, Omega, phi), 'r')
    plt.plot(xAxis, omegaE, 'g')
    plt.plot(xAxis, omegaEC, 'b')
    plt.show()

''' #
    # This statement functions to run the main function by default. This statement also encodes
    # a command line argument to accept the config file.
    # '''
if (__name__ == "__main__"):
    commandLineArgs = argparse.ArgumentParser()
    commandLineArgs.add_argument("--A", type = float, default = 0.01)
    commandLineArgs.add_argument("--Omega", type = float, default = 3.16)
    commandLineArgs.add_argument("--phi", type = float, default = 0)
    commandLineArgs.add_argument("--delta", type = float, default = 0.001)
    commandLineArgs.add_argument("--n", type = int, default = 10000)
    arguments = commandLineArgs.parse_args()
    main(arguments.A, arguments.Omega, arguments.phi, arguments.delta, arguments.n)
