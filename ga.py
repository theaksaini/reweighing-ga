import numpy as np
import copy
import pandas as pd

class Individual():

    def __init__(self, program, fitness):
        self.program = np.array(program)
        self.fitness = fitness


class GA():
    def __init__(self, ind_size, pop_size, max_gens, random_state, mut_rate, cross_rate, fitness_func):

        """
        Initializer for the GA class. Assume that the fitness function is a maximizing problem.

        Parameters
        ----------
        n_individuals : int
            Number of individuals for use in the population.
        n_generations : int
            Number of generations to evolve for.

        Attributes
        ----------
        best_individual : Individual
            The current best individual so far. 
        """

        self.max_gens = max_gens
        self.pop_size = pop_size
        self.ind_size = ind_size
        self.mut_rate = mut_rate
        self.cross_rate = cross_rate
        self.fitness_func = fitness_func
        self.rng = np.random.default_rng(random_state)
        self.population = []
        self.best_individual = None
        self.evaluated_individuals = pd.DataFrame(columns = ['individual','perf_fitness','fair_fitness'])

    def initialize_population(self):

        """Generates the population list of Individuals."""

        for _ in range(self.pop_size):
            self.population.append(Individual(2*self.rng.random(size=self.ind_size), None))

    def mutation(self, ind):
        ''' With a probability of mut_rate, mutate the individual. '''
        _program = copy.deepcopy(ind.program)
        if self.rng.random()<self.mut_rate:
            for i in range(len(_program)):
                if self.rng.random()<0.5:
                    new_val = _program[i] + self.rng.random()
                else:
                    new_val = _program[i] - self.rng.random()
                _program[i] = min(max(new_val,0),2)
                    
        return Individual(_program, None)


    def crossover(self, ind1, ind2):
        _program1 = copy.deepcopy(ind1.program)
        _program2 = copy.deepcopy(ind2.program)
        
        if self.rng.random()<self.cross_rate:
            child = []
            for i in range(len(_program1)):
                if self.rng.random()<0.5:
                    child.append(_program1[i])
                else:
                    child.append(_program2[i])
        else:
            child = _program1
                    
        return Individual(child, None)
    
    def selection(self):
        # Assume all objectives are to be maximized  
        candidates = self.population
        cases = list(range(len(self.population[0].fitness)))
        self.rng.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            errors_for_this_case = [x.fitness[cases[0]] for x in candidates] 
            median_val = np.median(errors_for_this_case)
            median_absolute_deviation = np.median([abs(x - median_val) for x in errors_for_this_case])
            best_val_for_case = max(errors_for_this_case )
            min_val_to_survive = best_val_for_case - median_absolute_deviation
            candidates = [x for x in candidates if x.fitness[cases[0]] >= min_val_to_survive]
            cases.pop(0)
        
        return self.rng.choice(candidates)

    def evaluate_population(self):

        """
        Sets the fitness of the individual passed in. Make sure higher fitness values are better.

        Parameters
        ----------
        individual : Individual
            Individual for which to assess the fitness.
        fn : function
            Fitness function used to evaluate the fitness.
        """

        for individual in self.population:
            individual.fitness = self.fitness_func(individual.program)
            # Update self.evaluated_individuals
            self.evaluated_individuals.loc[len(self.evaluated_individuals.index)] = {'individual':individual.program,'perf_fitness':individual.fitness[0],'fair_fitness':individual.fitness[1]}

        # Updating the best individual
        candidates = self.population
        cases = list(range(len(self.population[0].fitness)))
    
        while len(cases) > 0 and len(candidates) > 1:
            best_val_for_case = max([x.fitness[cases[0]] for x in candidates])
            candidates = [x for x in candidates if x.fitness[cases[0]] == best_val_for_case]
            cases.pop(0)
        self.best_individual = self.rng.choice(candidates)


    def step_optimize(self):

        """
        Progresses the optimization prcedure by a single iteration.

        Parameters
        ----------
        fn : function
            Fitness function used to evaluate the fitness.
        """

        _population = []
        for i in range(self.pop_size):
            parent_a = self.selection()
            parent_b = self.selection()

            child = self.crossover(parent_a, parent_b)

            child = self.mutation(child)

            _population.append(child)

        self.population = _population

        self.evaluate_population()

        

    def optimize(self):

        """
        Responsible for managing the optimisation process.

        Parameters
        ----------
        fn : function
            Fitness function used to evaluate the fitness.
        """

        print("Generation 1 started:")
        self.initialize_population()
        self.evaluate_population()
        print("Generation 1 ended.")
        for gen in range(self.max_gens-1):
            print("Generation ", gen+2, " started:")
            self.step_optimize()
            print("Generation ", gen+1, " ended.")
            print("Best Individual from so far: ", self.best_individual.fitness)