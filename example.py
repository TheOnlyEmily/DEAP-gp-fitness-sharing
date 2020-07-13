import operator
import math
import random

import numpy

import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from fitness_sharing_function import SemanticFitnessSharingFunction


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

fsf = SemanticFitnessSharingFunction()

def evalSymbRegExp(individual, points):
    func = toolbox.compile(expr=individual)
    semantics = [func(x) for x in points]
    target_semantics = [x**4 + x**3 + x**2 + x for x in points]
    delta_errors = [semantics[i] - target_semantics[i] for i in range(len(points))]

    sqerrors = list(map(lambda v: v**2, delta_errors))
    error = math.fsum(sqerrors) / len(points)
    error_adjust = fsf(delta_errors)

    return (error * error_adjust,)

def evalSymbRegCtrl(individual, points):
    func = toolbox.compile(expr=individual)
    semantics = [func(x) for x in points]
    target_semantics = [x**4 + x**3 + x**2 + x for x in points]
    sqerrors = [(semantics[i] - target_semantics[i])**2 for i in range(len(points))]
    return math.fsum(sqerrors) / len(points),

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def control_main():
    toolbox.register("evaluate", evalSymbRegCtrl, points=[x/10. for x in range(-10,10)])

    random.seed(318)

    pop = toolbox.population(n=75)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    ITERATIONS = 100
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, ITERATIONS, stats=mstats,
                                   halloffame=hof, verbose=False)
    # print log
    return pop, log, hof

def experimental_main():
    toolbox.register("evaluate", evalSymbRegExp, points=[x/10. for x in range(-10,10)])

    random.seed(318)

    pop = toolbox.population(n=75)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    ITERATIONS = 100
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, ITERATIONS, stats=mstats,
                                   halloffame=hof, verbose=False)
    # print log
    return pop, log, hof

if __name__ == "__main__":
    _, c_log, _ = control_main()
    c_fitness_list = c_log.chapters["fitness"].select("min")

    _, e_log, _ = experimental_main()
    e_fitness_list = e_log.chapters["fitness"].select("min")

    plt.plot(c_fitness_list, label="No Fitness Sharing")
    plt.plot(e_fitness_list, label="Fitness Sharing")
    plt.legend()
    plt.ylabel("Fitness")
    plt.xlabel("Generations")
    plt.show()