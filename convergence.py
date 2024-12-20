import numpy as np
import matplotlib.pyplot as plt

import electricPotentialEnergy as pe

from scipy.stats import norm

from typing import List, Tuple

SEED     = 42
MAX_SEED = 9999999999
N_ITER   = 10000
N_PRERUN = 3000

def processData(bestEns: List[np.ndarray[float]]) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Returns mean best over iter for multiple runs and CI
    """
    bestEns = np.asarray(bestEns)
    mean = np.mean(bestEns, axis=0)
    std  = np.std(bestEns, axis=0, ddof=1)
    z_score = norm.ppf(0.975)
    
    return mean, z_score * std

def axLayout(ax: plt.Axes, nParts, nullEn = None):
    NPART_RANGES = {
                    20: (386, 390),
                    50: (2910, 2930)
                    }

    if nullEn:
        ax.axhline(nullEn, ls = ':', label="No Annealing")

    ax.legend()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Best Energy")
    ax.set_ybound(*NPART_RANGES[nParts])


"""
Logarithmic Scheme
"""
def logCoolingConvergence(nParts, tempRange, nIter = 20000, nSample = 10, save=False, nullEn = None):
    meanEns = []
    confidence = []
    for temp in tempRange:
        seedGen = np.random.default_rng(SEED)

        bestEns = []
        for _ in range(nSample):
            seed = seedGen.integers(MAX_SEED)
            system = pe.Coulomb(nParts, seed)
            scheme = pe.logarithmicCooling(temp)
            bestEns.append(np.minimum.accumulate(pe.simulatedAnnealing(system, 1, nIter, scheme, 2, prerun=N_PRERUN)[1]))
        
        bestEns = np.asarray(bestEns)

        m, c = processData(bestEns)

        meanEns.append(m)
        confidence.append(c)
        
    
    fig, ax = plt.subplots()
    
    unit = "$T_{init}$"

    for temp, mean, std in zip(tempRange, meanEns, confidence):
        ax.fill_between(list(range(len(mean))),mean - std, mean + std, alpha=0.2)
        ax.plot(mean, label=f"{unit} = {temp:.2f}")
    
    ax.set_title(f"Logarithmic Cooling scheme in {nParts} particle system\n" + f"For varying values {unit}")
    axLayout(ax, nParts, nullEn)
    
    if save: fig.savefig(f"./schemeCon/LogVarT_{nParts}parts.png", dpi=1200)

    return fig, ax, meanEns


"""
Linear Scheme
"""
def linConvergeLinCoeff(nParts, lcs, temp, nIter = 20000, nSample = 10, save=False, nullEn = None):
    meanEns = []
    confidence  = []
    for lc in lcs:
        seedGen = np.random.default_rng(SEED)

        bestEns = []
        for _ in range(nSample):
            seed = seedGen.integers(MAX_SEED)
            system = pe.Coulomb(nParts, seed)
            scheme = pe.linearCooling(temp, lc)
            bestEns.append(np.minimum.accumulate(pe.simulatedAnnealing(system, 1, nIter, scheme, 2, prerun=N_PRERUN)[1]))
        
        bestEns = np.asarray(bestEns)
        
        m, c = processData(bestEns)
        meanEns.append(m)
        confidence.append(c)
    
    fig, ax = plt.subplots()
    
    for lc, mean, std in zip(lcs, meanEns, confidence):
        ax.fill_between(list(range(len(mean))), mean - std, mean + std, alpha=0.2)
        ax.plot(mean, label=f"lc = {lc:.2e}")

    ax.set_title("Linear Cooling scheme with $T_{init}$ = " + f"{temp} in {nParts} particle system\n" + f"For varying linear coefficients")
    axLayout(ax, nParts, nullEn)

    if save: fig.savefig(f"./schemeCon/linVarLc_T={temp}_{nParts}parts.png", dpi=1200)
    
    return fig, ax, meanEns


def linConvergeTemp(nParts, temps, lc, nIter = 20000, nSample = 10, save=False, nullEn = None):
    meanEns = []
    confidence  = []
    for temp in temps:
        seedGen = np.random.default_rng(SEED)

        bestEns = []
        for _ in range(nSample):
            seed = seedGen.integers(MAX_SEED)
            system = pe.Coulomb(nParts, seed)
            scheme = pe.linearCooling(temp, lc)
            bestEns.append(np.minimum.accumulate(pe.simulatedAnnealing(system, 1, nIter, scheme, 2, prerun=N_PRERUN)[1]))
        
        bestEns = np.asarray(bestEns)
        
        m, c = processData(bestEns)
        meanEns.append(m)
        confidence.append(c)
    
    fig, ax = plt.subplots()
    unit = "$T_{init}$"
    for temp, mean, std in zip(temps, meanEns, confidence):
        ax.fill_between(list(range(len(mean))), mean - std, mean + std, alpha=0.2)
        ax.plot(mean, label=f"{unit} = {temp:.2e}")


    ax.set_title("Linear Cooling scheme with $lc$ = " + f"{lc} in {nParts} particle system\n" + f"For varying {unit}")
    axLayout(ax, nParts, nullEn)

    if save: fig.savefig(f"./schemeCon/linVarT_LC={lc}_{nParts}parts.png", dpi=1200)
    
    return fig, ax, meanEns



"""
Geometric scheme
"""
def geoCoolingConvergenceT(nParts, tempRange, alpha, nIter = 20000, nSample = 10, save=False, nullEn = None):
    meanEns = []
    confidence  = []

    for temp in tempRange:
        seedGen = np.random.default_rng(SEED)
        bestEns = []
        for _ in range(nSample):
            seed = seedGen.integers(MAX_SEED)
            system = pe.Coulomb(nParts, seed)
            scheme = pe.geometricCooling(temp, alpha)
            bestEns.append(np.minimum.accumulate(pe.simulatedAnnealing(system, 1, nIter, scheme, 2, prerun=N_PRERUN)[1]))
    
        bestEns = np.asarray(bestEns)
        
        m, c = processData(bestEns)
        meanEns.append(m)
        confidence.append(c)
    
    fig, ax = plt.subplots()
    unit = "$T_{init}$"
    
    for temp, mean, std in zip(tempRange, meanEns, confidence):
        ax.fill_between(list(range(len(mean))), mean - std, mean + std, alpha=0.2)
        ax.plot(mean, label=f"{unit} = {temp:.2e}")

    ax.set_title("Arithmetic Geometric Cooling scheme with $\\alpha$ = " + f"{alpha} in {nParts} particle system for varying values {unit}")
    axLayout(ax, nParts, nullEn)

    if save: fig.savefig(f"./schemeCon/GeomVarT_A={alpha}_{nParts}parts.png", dpi=1200)
    
    return fig, ax, meanEns

def geoCoolingConvergenceA(nParts, alphaRange, temp, nIter = 20000, nSample = 10, save=False, nullEn = None):
    meanEns = []
    confidence  = []

    for alpha in alphaRange:
        seedGen = np.random.default_rng(SEED)
        bestEns = []

        for _ in range(nSample):
            seed = seedGen.integers(MAX_SEED)
            system = pe.Coulomb(nParts, seed)
            scheme = pe.geometricCooling(temp, alpha)
            bestEns.append(np.minimum.accumulate(pe.simulatedAnnealing(system, 1, nIter, scheme, 2, prerun=N_PRERUN)[1]))

        m, c = processData(bestEns)
        meanEns.append(m)
        confidence.append(c)
    
    fig, ax = plt.subplots()
    unit = "$\\alpha$"

    for alpha, mean, std in zip(alphaRange, meanEns, confidence):
        ax.fill_between(list(range(len(mean))), mean - std, mean + std, alpha=0.2)
        ax.plot(mean, label=f"{unit} = {alpha:.4f}")
    
    ax.set_title("Arithmetic Geometric Cooling scheme with $T_{init}$ = " + f"{temp} in {nParts} particle system\n" + f"For varying values {unit}")
    axLayout(ax, nParts, nullEn)

    if save: fig.savefig(f"./schemeCon/GeomVarA_T={temp}_{nParts}parts.png", dpi=1200)
    
    return fig, ax, meanEns


"""
Arithmetic scheme
"""
def arithCoolingConvergenceB(nParts, bRange, temp, alpha, nIter = 20000, nSample = 10, save=False, nullEn = None):
    meanEns = []
    confidence  = []

    for b in bRange:
        seedGen = np.random.default_rng(SEED)
        bestEns = []

        for _ in range(nSample):
            seed = seedGen.integers(MAX_SEED)
            system = pe.Coulomb(nParts, seed)
            scheme = pe.arithmeticGeometric(temp, alpha, b)
            bestEns.append(np.minimum.accumulate(pe.simulatedAnnealing(system, 1, nIter, scheme, 2, prerun=N_PRERUN)[1]))

        m, c = processData(bestEns)
        meanEns.append(m)
        confidence.append(c)
    
    fig, ax = plt.subplots()
    unit = "$b$"

    for b, mean, std in zip(bRange, meanEns, confidence):
        ax.fill_between(list(range(len(mean))), mean - std, mean + std, alpha=0.2)
        ax.plot(mean, label=f"{unit} = {b:.4f}")
    
    ax.set_title("Arithmetic Geometric Cooling scheme with $T_{init}$ = " + f"{temp} and a = {alpha} \n" + f"in {nParts} particle system for varying values {unit}")
    axLayout(ax, nParts, nullEn)

    if save: fig.savefig(f"./schemeCon/arithVarb_T={temp}_A={alpha}_{nParts}parts.png", dpi=1200)
    
    return fig, ax, meanEns


def arithCoolingConvergenceT(nParts, tempRange, alpha, b, nIter = 20000, nSample = 10, save=False, nullEn = None):
    meanEns = []
    confidence  = []

    for temp in tempRange:
        seedGen = np.random.default_rng(SEED)
        bestEns = []
        for _ in range(nSample):
            seed = seedGen.integers(MAX_SEED)
            system = pe.Coulomb(nParts, seed)
            scheme = pe.arithmeticGeometric(temp, alpha, b)
            bestEns.append(np.minimum.accumulate(pe.simulatedAnnealing(system, 1, nIter, scheme, 2, prerun=N_PRERUN)[1]))
    
        bestEns = np.asarray(bestEns)
        
        m, c = processData(bestEns)
        meanEns.append(m)
        confidence.append(c)
    
    fig, ax = plt.subplots()
    unit = "$T_{init}$"
    
    for temp, mean, std in zip(tempRange, meanEns, confidence):
        ax.fill_between(list(range(len(mean))), mean - std, mean + std, alpha=0.2)
        ax.plot(mean, label=f"{unit} = {temp:.2e}")

    ax.set_title("Geometric Cooling scheme with $b$ = " + f"{b} and a = {alpha}\n" + f"in {nParts} particle system\n" + f"For varying values {unit}")
    axLayout(ax, nParts, nullEn)

    if save: fig.savefig(f"./schemeCon/arithVarT_A={alpha}_b={b}_{nParts}parts.png", dpi=1200)
    
    return fig, ax, meanEns


def arithCoolingConvergenceA(nParts, alphaRange, temp, b, nIter = 20000, nSample = 10, save=False, nullEn = None):
    meanEns = []
    confidence  = []

    for alpha in alphaRange:
        seedGen = np.random.default_rng(SEED)
        bestEns = []

        for _ in range(nSample):
            seed = seedGen.integers(MAX_SEED)
            system = pe.Coulomb(nParts, seed)
            scheme = pe.arithmeticGeometric(temp, alpha, b)
            bestEns.append(np.minimum.accumulate(pe.simulatedAnnealing(system, 1, nIter, scheme, 2, prerun=N_PRERUN)[1]))

        m, c = processData(bestEns)
        meanEns.append(m)
        confidence.append(c)
    
    fig, ax = plt.subplots()
    unit = "$\\alpha$"
    for alpha, mean, std in zip(alphaRange, meanEns, confidence):
        ax.fill_between(list(range(len(mean))), mean - std, mean + std, alpha=0.2)
        ax.plot(mean, label=f"{unit} = {alpha:.4f}")
    
    ax.set_title("Geometric Cooling scheme with $T_{init}$ = " + f"{temp} in {nParts} particle system\n" + f"For varying values {unit}")
    ax.set_title("Geometric Cooling scheme with $a$ = " + f"{alpha} and b = {b}\n" + f"in {nParts} particle system\n" + f"For varying values {unit}")
    axLayout(ax, nParts, nullEn)

    if save: fig.savefig(f"./schemeCon/arithVarA_T={temp}_b={b}_{nParts}parts.png", dpi=1200)
    
    return fig, ax, meanEns


if __name__== "__main__":
    particles = [20]
    iters     = [20000]

    print("Running NullStrat")
    nullRuns = []
    for nPart in particles:
        print(nPart)
        _, _, en = linConvergeLinCoeff(nPart,  [1], 0, 30000, nSample=10, save=True)
        nullRuns.append(np.min(en))
    plt.close()
    
    print("Running log")
    for i, nPart in enumerate(particles):
        print(nPart)
        logCoolingConvergence(nPart , np.linspace(0.1, 10, 5), iters[i] ,nSample=10, save=True, nullEn=nullRuns[i])
    plt.close()
    
    print("Running geo T")
    for i, nPart in enumerate(particles):
        print(nPart)
        geoCoolingConvergenceT(nPart,  np.linspace(1000, 1e4, 5), 0.967, iters[i], nSample=10,  save=True, nullEn=nullRuns[i])
    plt.close()

    print("Running geo A")
    for i, nPart in enumerate(particles):
        print(nPart)
        geoCoolingConvergenceA(nPart,  np.linspace(0.9, 0.99, 5), 5.5e3, iters[i],  nSample=10, save=True, nullEn=nullRuns[i])
    plt.close()

    print("Running Lin LC")
    for i, nPart in enumerate(particles):
        print(nPart)
        linConvergeLinCoeff(nPart,  np.linspace(-1e-4, -1e-2, 5), 5, iters[i],  nSample=10,  save=True, nullEn=nullRuns[i])
    plt.close()

    print("Running Lin T")
    for i, nPart in enumerate(particles):
        print(nPart)
        linConvergeTemp(nPart , np.linspace(1, 10, 5), -0.05, iters[i],  nSample=10,  save=True, nullEn=nullRuns[i])
    plt.close()
    
    """
      print("Runnin Arith B")
    for i, nPart in enumerate(particles):
        print(nPart)
        figArithB0, ax, en = arithCoolingConvergenceB(nPart , np.linspace(-10, 10, 5), 5.5e3, 0.967,  iters[i],  nSample=10,  save=True, nullEn=nullRuns[i])
    plt.close()

    print("Running Arith A")
    for i, nPart in enumerate(particles):
        print(nPart)
        arithCoolingConvergenceT(nPart,  np.linspace(1000, 1e4, 5), 0.967, 0.1,  iters[i],  nSample=10, save=True, nullEn=nullRuns[i])
    plt.close()

    
    """