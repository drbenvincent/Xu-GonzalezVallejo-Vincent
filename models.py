import pymc3 as pm
import math
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def calc_AUC(logk, logs, discount_function, max_delay=101):
    '''Calculate Area Under Curve measure'''
    delays = np.linspace(0, max_delay, 500)
    df = discount_function(delays, np.exp(logk), np.exp(logs))
    normalised_delays = delays / np.max(delays)
    AUC = np.trapz(df, x=normalised_delays)
    return AUC


def calc_percent_predicted(R_predicted_prob, R_actual):
    nresponses = R_actual.shape[0]
    predicted_responses = np.where(R_predicted_prob>0.5, 1, 0)
    n_correct = sum(np.equal(predicted_responses, R_actual))
    return  n_correct / nresponses


def calc_log_loss(R_predicted_prob, R_actual):
    return log_loss(R_actual, R_predicted_prob)



# define functions to get properties of the given participant from the data

def get_n_participants(data):
    return max(data.id)+1

def get_PID(id, data):
    '''get PID for person with given id'''
    pdata = data.loc[data['id'] == id]
    return pdata['Participant'].reset_index(drop=True)[0]

def get_Ractual(id, data):
    '''get actual responses for person with this id'''
    pdata = data.loc[data['id'] == id]
    return pdata['R'].values

def get_paradigm(id, data):
    '''get experimental paradigm for person with this id'''
    pdata = data.loc[data['id'] == id]
    return pdata['paradigm'].values[0]

# expt 1
def get_reward_magnitude(id, data):
    '''get reward magnitude condition for person with this id'''
    pdata = data.loc[data['id'] == id]
    return pdata['reward_mag'].values[0]

# expt 2
def get_domain(id, data):
    '''get domain for person with this id'''
    pdata = data.loc[data['id'] == id]
    return pdata['domain'].values[0]



class Model():
    '''Base PyMC3 model class'''

    model = None
    
    def sample_from_prior(self):
        with self.model:
            prior = pm.sample_prior_predictive(10_000)
            
        self.prior_samples = prior
        
    def sample_from_posterior(self, sample_options = {'tune': 2000, 'draws': 5000,
                                                      'chains': 4, 'cores': 4,
                                                      'nuts_kwargs': {'target_accept': 0.95},
                                                      'random_seed': 1234}):
        with self.model:
            trace = pm.sample(**sample_options)
            
        self.posterior_samples = trace
        

        
class ModifiedRachlin(Model):
    '''modified Rachlin model with a fixed psychometric slope parameter'''
    
    def __init__(self, data):
        self.data = data    
        self.model = self.build_model()
        
    def V(self, reward, delay, logk, logs):
        '''Calculate the present subjective value of a given prospect'''
        k = pm.math.exp(logk)
        s = pm.math.exp(logs)
        return reward * self.discount_function(delay, k, s)

    @staticmethod
    def discount_function(delay, k, s):
        ''' This is the MODIFIED Rachlin discount function. This is outlined
        in Vincent & Stewart (2018).
        Vincent, B. T., & Stewart, N. (2018, October 16). The case of muddled
        units in temporal discounting. https://doi.org/10.31234/osf.io/29sgd
        '''
        return 1 / (1.0+(k*delay)**s)
    
    @staticmethod
    def Φ(VA, VB, α = 1.7, ϵ=0.01):
        '''Psychometric function which converts the decision variable (VB-VA)
        into a reponse probability. Output corresponds to probability of choosing
        the delayed reward (option B).'''
        return ϵ + (1.0-2.0*ϵ) * (1/(1+pm.math.exp(-α*(VB-VA))))
    
    def build_model(self):
        # decant data into local variables
        RA = self.data['RA'].values
        RB = self.data['RB'].values
        DA = self.data['DA'].values
        DB = self.data['DB'].values
        R = self.data['R'].values
        p = self.data['id'].values
        n_participants = max(self.data.id)+1
                
        # group is a lookup array for group of each participant
        temp = np.array([self.data['id'].values, self.data['condition'].values]).T
        temp = np.unique(temp, axis=0)
        group = temp[:,1]
        
        n_groups = np.max(group)+1
        g = [0, 1, 2, 3]
        
        with pm.Model() as model:
            # Hyperpriors 
            mu_logk = pm.Normal('mu_logk', mu=math.log(1/30), sd=2, shape=n_groups)
            sigma_logk = pm.Exponential('sigma_logk', 10, shape=n_groups)

            mu_logs = pm.Normal('mu_logs', mu=0, sd=0.5, shape=n_groups)
            sigma_logs = pm.Exponential('sigma_logs', 20, shape=n_groups)

            # Priors over parameters for each participant 
            logk = pm.Normal('logk', mu=mu_logk[group], sd=sigma_logk[group], shape=n_participants) 
            logs = pm.Normal('logs', mu=mu_logs[group], sd=sigma_logs[group], shape=n_participants)

            # group level inferences, unattached from the data
            group_logk = pm.Normal('group_logk', mu=mu_logk[g], sd=sigma_logk[g], shape=4) 
            group_logs = pm.Normal('group_logs', mu=mu_logs[g], sd=sigma_logs[g], shape=4)

            # Choice function: psychometric
            P = pm.Deterministic('P', self.Φ(self.V(RA, DA, logk[p], logs[p]),
                                             self.V(RB, DB, logk[p], logs[p])) )

            # Likelihood of observations
            R = pm.Bernoulli('R', p=P, observed=R)
            
        return model

    def calc_results(self, expt):
        rows = []
        n_participants = get_n_participants(self.data)
        for id in range(n_participants):

            # get trace from this participant
            logk = self.posterior_samples['logk'][:,id]
            logs = self.posterior_samples['logs'][:,id]
            P_chooseB = self.posterior_samples['P'][:,id]

            Ppredicted = self.posterior_samples.P[:, self.data['id'] == id]
            Ppredicted = np.median(Ppredicted, axis=0)

            Ractual = get_Ractual(id, self.data)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # make row data
            logk_point_estimate = np.mean(logk)
            logs_point_estimate = np.mean(logs)
            if expt is 1:
                rowdata = {'id': [id],
                        'PID': get_PID(id, self.data),
                        'logk': [logk_point_estimate], 
                        'logs': [logs_point_estimate],
                        'paradigm': [get_paradigm(id, self.data)], 
                        'reward_mag': [get_reward_magnitude(id, self.data)], 
                        'AUC': calc_AUC(logk_point_estimate, logs_point_estimate, self.discount_function), 
                        'percent_predicted': calc_percent_predicted(Ppredicted, Ractual),
                        'log_loss': calc_log_loss(Ppredicted, Ractual)}
            elif expt is 2:
                rowdata = {'id': [id],
                        'PID': get_PID(id, self.data),
                        'logk': [logk_point_estimate], 
                        'logs': [logs_point_estimate],
                        'paradigm': [get_paradigm(id, self.data)], 
                        'domain': [get_domain(id, self.data)], 
                        'AUC': calc_AUC(logk_point_estimate, logs_point_estimate, self.discount_function), 
                        'percent_predicted': calc_percent_predicted(Ppredicted, Ractual),
                        'log_loss': calc_log_loss(Ppredicted, Ractual)}
            rowdata = pd.DataFrame.from_dict(rowdata)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            rows.append(rowdata)
            # print(f'{id+1} of {n_participants}')

        parameter_estimates = pd.concat(rows, ignore_index=True)
        return parameter_estimates


class ModifiedRachlinFreeSlope(Model):
    '''modified Rachlin model with a FREE psychometric slope parameter'''

    def __init__(self, data):
        self.data = data    
        self.model = self.build_model()
        
    def V(self, reward, delay, logk, logs):
        '''Calculate the present subjective value of a given prospect'''
        k = pm.math.exp(logk)
        s = pm.math.exp(logs)
        return reward * self.discount_function(delay, k, s)

    @staticmethod
    def discount_function(delay, k, s):
        ''' This is the MODIFIED Rachlin discount function. This is outlined
        in Vincent & Stewart (2018).
        Vincent, B. T., & Stewart, N. (2018, October 16). The case of muddled
        units in temporal discounting. https://doi.org/10.31234/osf.io/29sgd
        '''
        return 1 / (1.0+(k*delay)**s)
    
    @staticmethod
    def Φ(VA, VB, α, ϵ=0.01):
        '''Psychometric function which converts the decision variable (VB-VA)
        into a reponse probability. Output corresponds to probability of choosing
        the delayed reward (option B).'''
        return ϵ + (1.0-2.0*ϵ) * (1/(1+pm.math.exp(-α*(VB-VA))))
    
    def build_model(self):
        # decant data into local variables
        RA = self.data['RA'].values
        RB = self.data['RB'].values
        DA = self.data['DA'].values
        DB = self.data['DB'].values
        R = self.data['R'].values
        p = self.data['id'].values
        n_participants = max(self.data.id)+1
                
        # group is a lookup array for group of each participant
        temp = np.array([self.data['id'].values, self.data['condition'].values]).T
        temp = np.unique(temp, axis=0)
        group = temp[:,1]
        
        n_groups = np.max(group)+1
        g = [0, 1, 2, 3]
        
        with pm.Model() as model:
            # Hyperpriors 
            mu_logk = pm.Normal('mu_logk', mu=math.log(1/30), sd=2, shape=n_groups)
            sigma_logk = pm.Exponential('sigma_logk', 10, shape=n_groups)

            mu_logs = pm.Normal('mu_logs', mu=0, sd=0.5, shape=n_groups)
            sigma_logs = pm.Exponential('sigma_logs', 20, shape=n_groups)

            # Priors over parameters for each participant 
            logk = pm.Normal('logk', mu=mu_logk[group], sd=sigma_logk[group], shape=n_participants) 
            logs = pm.Normal('logs', mu=mu_logs[group], sd=sigma_logs[group], shape=n_participants)
            # α = pm.Uniform('α', lower=0, upper=50, shape=n_participants)

            BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
            α = BoundedNormal('α', mu=1.7, sd=3, shape=n_participants)

            # group level inferences, unattached from the data
            group_logk = pm.Normal('group_logk', mu=mu_logk[g], sd=sigma_logk[g], shape=4) 
            group_logs = pm.Normal('group_logs', mu=mu_logs[g], sd=sigma_logs[g], shape=4)

            # Choice function: psychometric
            P = pm.Deterministic('P', self.Φ(self.V(RA, DA, logk[p], logs[p]),
                                             self.V(RB, DB, logk[p], logs[p]), α[p]) )

            # Likelihood of observations
            R = pm.Bernoulli('R', p=P, observed=R)
            
        return model

    def calc_results(self, expt):
        rows = []
        n_participants = get_n_participants(self.data)
        for id in range(n_participants):

            # get trace from this participant
            logk = self.posterior_samples['logk'][:,id]
            logs = self.posterior_samples['logs'][:,id]
            α = self.posterior_samples['α'][:,id]
            P_chooseB = self.posterior_samples['P'][:,id]

            Ppredicted = self.posterior_samples.P[:, self.data['id'] == id]
            Ppredicted = np.median(Ppredicted, axis=0)

            Ractual = get_Ractual(id, self.data)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # make row data
            logk_point_estimate = np.mean(logk)
            logs_point_estimate = np.mean(logs)
            α_point_estimate = np.mean(α)
            if expt is 1:
                rowdata = {'id': [id],
                        'PID': get_PID(id, self.data),
                        'logk': [logk_point_estimate], 
                        'logs': [logs_point_estimate],
                        'α': [α_point_estimate],
                        'paradigm': [get_paradigm(id, self.data)], 
                        'reward_mag': [get_reward_magnitude(id, self.data)], 
                        'AUC': calc_AUC(logk_point_estimate, logs_point_estimate, self.discount_function), 
                        'percent_predicted': calc_percent_predicted(Ppredicted, Ractual),
                        'log_loss': calc_log_loss(Ppredicted, Ractual)}
            elif expt is 2:
                rowdata = {'id': [id],
                        'PID': get_PID(id, self.data),
                        'logk': [logk_point_estimate], 
                        'logs': [logs_point_estimate],
                        'α': [α_point_estimate],
                        'paradigm': [get_paradigm(id, self.data)], 
                        'domain': [get_domain(id, self.data)], 
                        'AUC': calc_AUC(logk_point_estimate, logs_point_estimate, self.discount_function), 
                        'percent_predicted': calc_percent_predicted(Ppredicted, Ractual),
                        'log_loss': calc_log_loss(Ppredicted, Ractual)}
            rowdata = pd.DataFrame.from_dict(rowdata)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            rows.append(rowdata)
            # print(f'{id+1} of {n_participants}')

        parameter_estimates = pd.concat(rows, ignore_index=True)
        return parameter_estimates