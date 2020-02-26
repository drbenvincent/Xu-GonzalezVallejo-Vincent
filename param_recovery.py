import numpy as np
import pandas as pd


class ParamRecovery2d:
    def __init__(self, params, param_recovery_func, n_simulations, verbose=True):
        self.params = params
        self.param_recovery_func = param_recovery_func
        self.n_simulations = 10
        self.verbose = verbose

        # create 2d grid of params
        self.true_params = self.make_true_param_grid()

        # start the parameter recovery
        self.results = self.parameter_recovery()

    def make_true_param_grid(self):
        """
        params = {'alpha': np.linspace(0, 50, num=10),
                  'beta': np.linspace(0, 1, num=10)}
        """

        alpha_list = []
        beta_list = []

        param_names = list(self.params.keys())

        for alpha in self.params[param_names[0]]:
            for beta in self.params[param_names[1]]:
                alpha_list.append(alpha)
                beta_list.append(beta)

        true_params = pd.DataFrame(
            {param_names[0]: alpha_list, param_names[1]: beta_list}
        )
        return true_params

    def parameter_recovery(self):
        """
        Run a parameter recovery potentially multiple times. Each time we generate simulated behavioural data and infer the parameters from that data.

        INPUTS
        true_parameters: a dictionary
        simulate_experiment: a function with true_parameters as input
        parameter_estimation: a function with data as input
        """

        print("Commencing parameter recovery. Might take some time...")
        params_inferred = []

        # TODO: iterate over rows of dataframe (self.true_params)
        for i, params_true in enumerate(PARAMS):

            if self.verbose:
                print(f"True parameters ({i} of {len(PARAMS)})")

            for s in range(self.n_simulations):
                if self.verbose:
                    print(f"{s} of {self.n_simulations}")
                data = param_recovery_func(params_true)

            # TODO: concatenate into a dataframe
        return results

    def plot(self,):

        # plot inferred parameters

        # plot true parameters
        return
