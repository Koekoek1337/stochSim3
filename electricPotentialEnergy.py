import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple

RADIUS = 1

class Coulomb:
    def __init__(self, nParticles: int, seed: int = None) -> None:
        self.generator = np.random.default_rng(seed)
        self.state = np.zeros((nParticles, 2), float)
        self.initParticles()
    
    def initParticles(self):
        """ 
        Generates uniform distribution of the particles by sampling from polar coordinates
        """
        for i in range(self.state.shape[0]):
            dist  = np.sqrt(self.generator.random())
            angle = self.generator.random() * 2 * np.pi

            self.state[i, :] = dist * np.cos(angle), dist * np.sin(angle)
    
    def stateEnergy(self):
        """
        Calculates the energy of the current system state.
        """
        energySum = 0.0
        for i in range(self.state.shape[0]):
            for j in range(i + 1, self.state.shape[0]):
                pyt  = np.square(self.state[i,:] - self.state[j,:])
                E_ij = np.divide(1, np.sqrt(np.sum(pyt)))

                energySum += 2 * E_ij
        return energySum

    

def plotState(system: Coulomb) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the state of `system`.
    """
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    circle = plt.Circle((0,0), 1, color="black", fill=False)
    ax.add_patch(circle)
    ax.scatter(system.state[:,0], system.state[:,1])

    return fig, ax

if __name__ == "__main__":
    system = Coulomb(10)
    fig, ax = plotState(system)
    print(system.stateEnergy())
    plt.show()

