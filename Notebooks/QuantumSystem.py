"""
Trevor Smith
PHYS 472 Spring 2020
Computational Project

This script defines the importable class QuantumSystem, which we
developed in our test notebook HarmonicOscillatorTest.ipynb
"""
from scipy.integrate import simps

class QuantumSystem:
    def __init__(self, E, m=5.6856e-32):
        """
        Takes and stores constants needed for every quantum particle wavefunction.
        Inputs:
            E = particle energy (eV)
            m = particle mass (ev/(angstrom/s)^2)
                default value is equal to 1 electron mass
        """
        self.E = E
        self.m = m
        self.hbar = 6.5821e-16 # eV*s
    
    def V(self, x):
        """
        Function that returns the potential as a function of position
        for a specific quantum system with a 1D potential.
        This is a placeholder function for child classes to override.
        Inputs:
            x = the position to evaluate the potential energy at (angstrom)
        Returns:
            the potential at position x (eV)
        """
        return 0
    
    def TISE(self, psi, x):
        """
        Function to calculate the second derivative of the wavefunction
        with respect to x, by solving the Time Independent Schroedinger
        Equation (TISE).
        Inputs:
            x = position (angstroms)
            psi = wavefunction amplitude at x
        Returns:
            the second derivative of the wavefunction at x
            (amplitude/angstrom^2)
        """
        return 2*self.m/self.hbar**2*(self.V(x)-self.E)*psi
    
    def Step(self, psi0, psidot0, x0, dx):
        """
        Function to approximate the wavefunction amplitude and slope
        at the point x0+dx using a Taylor series expansion
        Inputs:
            psi0 = value of wavefunction at x0 (amplitude)
            psidot0 = wavefunction slope at x0 (amplitude/angstrom)
            x0 = starting position (angstroms)
            dx = size of step (angstroms)
        Returns:
            psi = estimated wavefunction at x0+dx (amplitude)
            psidot = estimated slop of wavefunction at x0+dx
                (amplitude/angstrom)
            x = new position (angstroms)
        """
        # solve TISE to get curvature (psidotdot) at x0
        psidotdot0 = self.TISE(psi0,x0)
        
        # new position after this calculation
        x = x0 + dx
        
        # calculate approximate slope at x
        psidot = psidot0 + dx*psidotdot0
        
        # calculate approximate amplitude at x0+dx (Taylor Series)
        psi = psi0 + dx*psidot0 + (dx**2/2)*psidotdot0
        
        # return new amplitude, slope, and position
        return psi, psidot, x
    
    def Integrate(self, psi0, psidot0, x0, xmax, dx):
        """
        Function that calculates entire wavefunction over a given interval.
        Inputs:
            psi0 = initial amplitude at x0
            psidot0 = initial slope at x0
            x0 = starting position (angstroms)
            xmax = final position to calculate, must be > x0 (angstroms)
        Returns:
            psi = list of amplitude values
            x = list of positions (angstroms) corresponding to psi values
        """
        # initialize lists to store wavefunction data
        psi = [psi0]
        x = [x0]
        
        # we only need to store latest value of psidot, since we only
        # need it for the calculation
        psidot = psidot0
        
        # perform the integration
        # loop while the latest value for x is less than xmax
        while x[-1] < xmax:
            # step to x+dx
            psinew, psidotnew, xnew = self.Step(psi[-1],psidot0,x[-1],dx)

            # add new values to our lists
            psi += [psinew]
            x += [xnew]

            # we only need to save latest value of psidot
            psidot0 = psidotnew
        # return wavefunction data
        return psi, x
        
    def Normalize(self, psi, x):
        """
        Function to normalize a given calculated (real-valued) wavefunction.
        Normalization condition: |psi|^2 = 1
        Wavefunction must converge at both ends.
        Inputs:
            psi = array of amplitude values
            x = array of positions that corresponds to values in psi
                (angstroms)
        Returns:
            psi * calculated normalization constant
        """
        # calculate mod squared of the wavefunction
        # integrate using scipy.integrate.simps
        modsquared = simps(psi**2,x=x)
        
        # normalization constant
        A = 1/modsquared**(0.5)
        
        # return normalized psi
        return A*psi
    
    def set_E(self, E):
        """
        Function to allow changing E after instance is initialized.
        Inputs:
            E = new particle energy (eV)
        Returns:
            none
        """
        self.E = E
