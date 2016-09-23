__author__ = 'pva701'

import numpy.random as rnd
import numpy as np
from Misc import gen_random_combination

class GeneticOptimization:

    def __crossover(self, x, y):
        indices = np.arange(1, self.dim).reshape(self.dim - 1, 1)
        return np.apply_along_axis(lambda s: np.append(x[:s], y[s:]), 1, indices).ravel()

    def __multi_crossover(self, x, xs):
        return np.apply_along_axis(lambda y: self.__crossover(x, y), 1, xs).ravel()

    def __to_bounds(self, val):
        return val * (self.max_value - self.min_value) + self.min_value

    def __mutation(self, x):
        x[rnd.randint(0, self.dim - 1)] = self.__to_bounds(rnd.rand())
        return x

    def __metric(self, predicted, label):
        return ((predicted - label) ** 2).mean()

    def fit(self, xs, ys, pop_size=50, iters=50,
            error=0.001, mutation_part=0.2, reproduction_part=0.2, min_value=0, max_value=1.,
            verbose=False):

        minimize_f = lambda w: self.__metric(np.apply_along_axis(lambda x: np.dot(w, x), 1, xs), ys)
        dim = len(xs[0])
        self.dim = dim
        self.max_value = max_value
        self.min_value = min_value
        population = self.__to_bounds(rnd.random([pop_size, dim]))

        for it in range(iters):
            cur_population_size = len(population)
            seq_indexes = np.arange(0, cur_population_size).reshape(cur_population_size, 1)

            parents_ids = np.apply_along_axis(
                lambda ind: gen_random_combination(cur_population_size, int(reproduction_part * cur_population_size), ind), 1,
                seq_indexes)

            new_population = np.apply_along_axis(
                lambda ind: self.__multi_crossover(population[ind[0]], population[parents_ids[ind[0]]]), 1,
                seq_indexes).reshape((-1, dim))

            mutation_population = np.apply_along_axis(
                lambda x: self.__mutation(x) if rnd.rand() < mutation_part else x, 1,
                new_population)

            fitness = np.apply_along_axis(minimize_f, 1, mutation_population)

            best_indices = np.argsort(fitness)[:pop_size]
            min_error = fitness[best_indices[0]]
            population = new_population[best_indices]
            if min_error < error:
                if verbose:
                    print("Optimization complete, requirement error reached {}", min_error)
                self.w = population[0]
                return self
            if verbose:
                print("Iteration {} done. Current error {}".format(it + 1, min_error))
        if verbose:
            print("Optimization completed, requirement number of iterations reached")
        self.w = population[0]
        return self

    def predict(self, x):
        return np.dot(self.w, x)

    def weights(self):
        return self.w