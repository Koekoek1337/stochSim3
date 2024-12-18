import numpy as np
import scipy as sp

from time import time
import scipy.spatial as spatial
import matplotlib.pyplot as plt

from typing import List, Tuple, Generator, Union

RADIUS = 1

class Coulomb:
    def __init__(self, nParticles: int, seed: int = None) -> None:
        self.generator = np.random.default_rng(seed)
        self.state = np.zeros((nParticles, 2), float)
        self.initParticles()
        self.currentEnergy = self.stateEnergy()
        self.newEnergy = self.currentEnergy
        self.bestEnergy = self.currentEnergy
    
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

        Optimized for speed (old method took 5.3 seconds for 1000 particles, current takes 0.06)
        """
        distance = spatial.distance_matrix(self.state, self.state)
        np.fill_diagonal(distance, 1) # The diagonals are all 0, which would result in inf when divided
        energy = np.sum(np.divide(1, distance)) - self.state.shape[0] # Correct for the modified diagonal
        return energy

    def moveParticle(self, index: int, max_step: float) -> None:
        """
        Moves a single particle randomly and ensures it stays within the circle.
        """
        angle = 2 * np.pi * self.generator.random()
        step = max_step * self.generator.random()

        new_position = self.state[index] + step * np.array([np.cos(angle), np.sin(angle)])
        if np.linalg.norm(new_position) >= RADIUS:
            self.state[index] = new_position/np.linalg.norm(new_position)
        else:
            self.state[index] = new_position

    def forceMoveParticle(self, index, max_step):
        forceComp = self.state[:,:] - self.state[index, :]
        dists = spatial.distance_matrix(self.state[index:index+1, :], self.state)
        dists = dists[0,:]
        dists[index] = 1
        forces = np.divide(1, dists)
        forceComp = forceComp * np.asarray([forces, forces]).transpose()
        fx, fy = np.sum(forceComp[:,0]), np.sum(forceComp[:,1])

        forceAngle = np.arctan(fy/fx)

        angle = self.generator.normal(forceAngle, np.pi / 4)
        step = max_step * self.generator.random()

        new_position = self.state[index] + step * np.array([np.cos(angle), np.sin(angle)])
        if np.linalg.norm(new_position) >= RADIUS:
            self.state[index] = new_position/np.linalg.norm(new_position)
        else:
            self.state[index] = new_position


def simulatedAnnealing(system: Coulomb, chain_length: int, max_iters: int, 
                       cooling_scheme: Generator[float, None, None], max_step: float, 
                       save_path: str = None) -> Tuple[Coulomb, np.ndarray[float]]:
    """
    Simulated Annealing to find the minimal energy configuration.
    Optionally logs system states and energies to a NumPy .npy files.

    Args:
        system: Coulomb object representing system of classically charged particles in a round box.
        chain_length: Length of the markov chain, representing the amount of particle moves at the
                      same temperature.
        max_iters: Maximum amount of iters before the calculation is deemed too expensive.
        cooling scheme: Generator instance that represents the cooling scheme in between markov chain steps.

    """
    temp = next(cooling_scheme)
    best_state = system.state.copy()
    best_energy = system.stateEnergy()

    log_dict = {"states": [], "energies": []}
    energy = np.zeros(max_iters)
    system.newEnergy = best_energy
    
    step = 0
    converged = False

    # Modified to work with markov chain length
    while not converged and (step < max_iters or step == None):
        for subStep in range(chain_length):

            index = system.generator.integers(system.state.shape[0])
            old_state = system.state[index].copy()

            system.forceMoveParticle(index, max_step)
            system.newEnergy = system.stateEnergy()

            energy_difference = system.newEnergy - system.currentEnergy

            if energy_difference > 0:
                acceptance_prob = np.exp(-(energy_difference) / temp)
                if system.generator.random() > acceptance_prob:
                    system.state[index] = old_state
                    system.newEnergy = system.currentEnergy
                else:
                    system.currentEnergy = system.newEnergy
            else:
                system.currentEnergy = system.newEnergy
            
            energy[step] = system.currentEnergy

            if system.newEnergy < system.bestEnergy:
                system.bestEnergy = system.newEnergy
                best_state = system.state.copy()
            
            if save_path is not None:
                log_dict["states"].append(system.state.copy())
                log_dict["energies"].append(energy[step])

            step += 1
            if not (step < max_iters or step == None): break

        temp = next(cooling_scheme)
    
    if save_path is not None:
        np.save(f"{save_path}_states.npy", np.array(log_dict["states"]))
        np.save(f"{save_path}_energies.npy", np.array(log_dict["energies"]))

    system.state = best_state

    return system, energy

def optimize(x):
    system = Coulomb(11, seed=42)
    return min(simulatedAnnealing(system, 10, 10000, geometricCooling(10000, x[0]), x[1])[1])

def plotState(state: np.ndarray, animation: bool = False, ax: plt.Axes = None) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:
    """
    Plots the state of `system`.
    """
    if animation is False:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.clear()
    circle = plt.Circle((0,0), RADIUS, color="black", fill=False)
    ax.add_patch(circle)
    ax.scatter(state[:,0], state[:,1])

    if animation is False:
        return fig, ax
    else:
        return ax


def linearCooling(T_Init, dT) -> Generator[float, None, None]:
    """ 
    Yields a linearily cooled themperature over iterations with temperature steps dT. 
    such that T = T_0 - i*dT. Yields 0 if T <= 0.

    Args:
        T_Init: Initial temperature
        dT:     some linear coefficient 0 < dT < T_Init
    """
    yield T_Init

    T = T_Init - dT
    while T_Init > 0:
        yield T
        T -= dT
    
    yield 0


def geometricCooling(T_Init, alpha) -> Generator[float, None, None]:
    """
    Yields a geometrically cooled temperature over iterations such that T = T_0 * a^i

    Args:
        T_Init: Initial temperature
        alpha:  Some value 0 < alpha < 1.
    """
    yield T_Init

    nIter = 1

    while True:
        yield T_Init * np.power(alpha, nIter)
        nIter += 1


def logarithmicCooling(T_Init):
    """ 

    """
    nIter = 1
    c = T_Init * np.log10(1 + nIter)
    yield T_Init

    while True:
        nIter += 1
        yield c / np.log10(1 + nIter)


def arithmeticGeometric(T_init, a, b):
    """
    TODO: Elaborate 

    https://doi.org/10.13053/cys-21-3-2553
    """

    yield T_init
    T = T_init
    while True:
        T = a * T + b
        yield T

if __name__ == "__main__":
    system = Coulomb(1000)
    fig, ax = plotState(system.state)
    tStart = time()
    print(system.stateEnergy())
    print(f"time: {time() - tStart}")
    plt.show()