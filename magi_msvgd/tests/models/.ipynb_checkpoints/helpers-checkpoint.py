import numpy as np

def obs_matrix(tau):
    '''
    Convert list of observed times to matrix of observed times, column 0 the joined times,
    remaining columns contain truthy/falsy indicating whether observed.
    '''
    obs_t = np.unique(np.concatenate(tau))
    
    obs_times = np.zeros([obs_t.shape[0], len(tau)+1])
    obs_times[:,0] = obs_t
    for d, tau_d in enumerate(tau):
        obs_times[np.where(np.isin(obs_times[:,0], tau_d)),d+1] = 1.0
    return obs_times


def discretize_data(self, sample, I):
    '''
    Discretize a dataset at time set I.
    '''
    data_disc = np.full(shape=(len(I), sample.shape[1]), fill_value=np.nan)
    data_disc[:,0] = I
    data_disc[np.isin(data_disc[:,0], sample[:,0])] = sample
    return data_disc