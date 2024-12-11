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
    
    def moveParticle(self, index: int, max_step: float) -> None:
        """
        Moves a single particle randomly and ensures it stays within the circle.
        """
        angle = 2 * np.pi * self.generator.random()
        step = max_step * (2 * self.generator.random() - 1)

        while True:
            new_position = self.state[index] + step * np.array([np.cos(angle), np.sin(angle)])
            if np.linalg.norm(new_position) <= RADIUS:
                self.state[index] = new_position
                break
            angle = 2 * np.pi * self.generator.random()

def simulatedAnnealing(system: Coulomb, max_steps: int, initial_temp: float, cooling_rate: float, max_step: float) -> Coulomb:
    """
    Simulated Annealing to find the minimal energy configuration.
    """
    temp = initial_temp
    best_state = system.state.copy()
    best_energy = system.stateEnergy()

    for step in range(max_steps):
        index = system.generator.integers(system.state.shape[0])

        old_state = system.state[index].copy()
        old_energy = system.stateEnergy()

        system.moveParticle(index, max_step)
        new_energy = system.stateEnergy()

        if new_energy > old_energy:
            acceptance_prob = np.exp(-(new_energy - old_energy) / temp)
            if system.generator.random() > acceptance_prob:
                system.state[index] = old_state

        if new_energy < best_energy:
            best_energy = new_energy
            best_state = system.state.copy()

        temp *= cooling_rate

    system.state = best_state

    return system

def plotState(system: Coulomb) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the state of `system`.
    """
    fig, ax = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(4)
    circle = plt.Circle((0,0), 1, color="black", fill=False)
    ax.add_patch(circle)
    ax.scatter(system.state[:,0], system.state[:,1])

    return fig, ax

if __name__ == "__main__":
    system = Coulomb(10)
    fig, ax = plotState(system)
    print(system.stateEnergy())
    plt.show()