__author__ = 'pva701'

import numpy.random as rnd
import numpy as np


class GeneticOptimization:
    def __crossover(self, x, y):
        indices = np.arange(1, self.dim).reshape(self.dim - 1, 1)
        return np.apply_along_axis(lambda s: np.append(x[:s], y[s:]), 1, indices).ravel()

    def __multi_crossover(self, x, xs):
        return np.apply_along_axis(lambda y: self.__crossover(x, y), 1, xs).ravel()

    def __mutation(self, x):
        x[rnd.randint(0, self.dim - 1)] = rnd.rand()
        return x

    def __gen_random_parents(self, ind, n):
        num_parents = int(n * self.reproduction_part)
        if self.reproduction_part > 0.5:
            perm = np.append(np.arange(0, ind), np.arange(ind + 1, n))
            rnd.shuffle(perm)
            return perm[:num_parents]
        else:
            used = [False] * n
            ret = []
            while len(ret) < num_parents:
                parent_id = rnd.randint(0, n)
                if used[parent_id] or parent_id == ind:
                    continue
                used[parent_id] = True
                ret.append(parent_id)
            ret = np.array(ret)
            return ret

    def fit(self, minimize_f, dim, pop_size=50, iters=50,
            error=0.001, mutation_part=0.2, reproduction_part=0.2,
            verbose=False):

        self.dim = dim
        self.reproduction_part = reproduction_part
        population = rnd.random([pop_size, dim])

        for it in range(iters):
            cur_population_size = len(population)
            seq_indexes = np.arange(0, cur_population_size).reshape(cur_population_size, 1)

            parents_ids = np.apply_along_axis(
                lambda ind: self.__gen_random_parents(ind, cur_population_size), 1,
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
                return population[0]
            if verbose:
                print("Iteration {} done. Current error {}".format(it + 1, min_error))
        if verbose:
            print("Optimization completed, requirement number of iterations reached")
        return population[0]
