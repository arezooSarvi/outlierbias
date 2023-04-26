import numpy as np


def reshape_predicted(predicted, dlr, sessions):
    max_list_size = np.max(np.diff(dlr))
    reshaped = []
    for qid in range(dlr.shape[0] - 1):
        s_i, e_i = dlr[qid:qid+2]
        extended = predicted[qid][sessions[qid]]
#         if len(extended.shape) == 1:
#             extended = extended[None,:]
        padded = np.pad(extended.astype(np.float), ((0,0,),(0,max_list_size-extended.shape[1],),), 'constant', constant_values=np.nan)
        reshaped.append(padded)
    return np.concatenate(reshaped, 0)

def stack_padded_clicks(clicks, max_list_size):
    stacked = []
    for qid in range(len(clicks)):
        padded = np.pad(clicks[qid].astype(np.float), ((0,0,),(0,max_list_size-clicks[qid].shape[1],),), 'constant', constant_values=np.nan)
        stacked.append(padded)
        
    return np.concatenate(stacked, 0)

def extend_group_ids(group_ids, sessions):
    repeats = [x.shape[0] for x in sessions]
    return np.repeat(group_ids, repeats)



def sigmoid(x):
    xx = np.clip(x, -10, 10)
    s = np.where(x >= 0, 
                    1 / (1 + np.exp(-xx)), 
                    np.exp(xx) / (1 + np.exp(xx)))
    s[x > 10] = 1.
    s[x < -10] = 0.
    return s

def normalize(x):
#     y = x - np.nanmin(x, 1)[:,None]
#     s = np.nanmax(x, 1) - np.nanmin(x, 1) + 1.e-6
#     x = y / s[:,None]
#     return sigmoid((x * 10) - 5)
    return sigmoid(x)
    

def loop_f(f, k):
    def looped(*x):
        for _ in range(k):
            f(*x)
        return f(*x)
    return looped

class Correction():
    def __init__(self, correction, EM_step_size = 0.1):
        self._EM_step_size = EM_step_size
        self.propensity = None
        self.group_ids = None
        self.correction_name = correction
        if correction == 'affine':
            self.expmax = self._affine_expectation
            self.debias = self._affine_debias
        elif correction == 'IPS':
            self.expmax = self._IPS_expectation
            self.debias = self._IPS_debias
        elif correction == 'oracle':
            self.expmax = self._oracle_expectation
            self.debias = self._affine_debias
        elif correction != 'naive':
            raise Exception('correction method not implemented!')
        
        self.expmax = loop_f(self.expmax,2)
            
    def init_params(self, clicks, dlr, group_ids):
        labels = stack_padded_clicks(clicks, np.max(np.diff(dlr)))
        
        self.group_ids = group_ids
        
        self.propensity = np.ones_like(labels) * 0.9
        self.epsilon_p = np.ones_like(labels)
        self.epsilon_n = np.zeros_like(labels)
        
        if self.correction_name == 'affine':
            self.epsilon_p *= 0.9
            self.epsilon_n += 0.1
            
    def load_oracle_values(self, params):
        if self.propensity is None:
            self.propensity = np.ones(self.params_shape) * 1.
            self.epsilon_p = np.ones(self.params_shape) * 1.
            self.epsilon_n = np.ones(self.params_shape) * 0.
            
        for i in range(self.propensity.shape[0]):
            self.propensity[i,:] = np.array(params[str(self.group_ids[i])]['propensity'])[:self.propensity.shape[1]]
            if 'epsilon_p' in params[str(self.group_ids[i])]:
                self.epsilon_p[i,:] = np.array(params[str(self.group_ids[i])]['epsilon_p'])[:self.epsilon_p.shape[1]]
            if 'epsilon_n' in params[str(self.group_ids[i])]:
                self.epsilon_n[i,:] = np.array(params[str(self.group_ids[i])]['epsilon_n'])[:self.epsilon_n.shape[1]]
        
        
    def _oracle_expectation(self, predicted, clicks, sessions, dlr):
        if self.group_ids is None:
            predicted = reshape_predicted(predicted, dlr, sessions)
            self.params_shape = predicted.shape

        
    def _affine_expectation(self, predicted, clicks, sessions, dlr):
        predicted = reshape_predicted(predicted, dlr, sessions)
        labels = stack_padded_clicks(clicks, predicted.shape[1])
            
        gamma = normalize(predicted)
        c_prob = (self.epsilon_p * gamma) + (self.epsilon_n * (1. - gamma))

        p_e1_r1_c1 = (self.epsilon_p * gamma) / c_prob
        p_e1_r1_c0 = self.propensity * \
            (1. - self.epsilon_p) * (gamma) / \
            (1 - self.propensity * c_prob)

        p_e1_r0_c1 = 1. - p_e1_r1_c1
        p_e1_r0_c0 = self.propensity * \
            (1. - self.epsilon_n) * (1 - gamma) / \
            (1 - self.propensity * c_prob)

        p_e0_r1_c1 = p_e0_r0_c1 = 0
        p_e0_r1_c0 = (1 - self.propensity) * gamma / \
            (1 - self.propensity * c_prob)
        p_e0_r0_c0 = (1 - self.propensity) * (1 - gamma) / \
            (1 - self.propensity * c_prob)

        
        propensity = (1 - self._EM_step_size) * self.propensity
        epsilon_p = (1 - self._EM_step_size) * self.epsilon_p
        epsilon_n = (1 - self._EM_step_size) * self.epsilon_n
        
        
        propensity_mat = labels + (1 - labels) * (p_e1_r0_c0 + p_e1_r1_c0)

        epsilon_p_mat_nom = labels * p_e1_r1_c1
        epsilon_p_mat_denom = (labels * p_e1_r1_c1) + ((1. - labels) * p_e1_r1_c0)
        epsilon_p_mat_denom[epsilon_p_mat_denom < 1.e-6] = 1.

        epsilon_n_mat_nom = labels * p_e1_r0_c1
        epsilon_n_mat_denom = (labels * p_e1_r0_c1) + ((1. - labels) * p_e1_r0_c0)
        epsilon_n_mat_denom[epsilon_n_mat_denom < 1.e-6] = 1.
        
        unique_groups = np.unique(self.group_ids)
        
        for group in unique_groups:
            mask = self.group_ids == group
            propensity[mask,:] += self._EM_step_size * np.nanmean(propensity_mat[mask,:], axis=0, keepdims=True)
            epsilon_p[mask,:] += self._EM_step_size * np.nanmean(epsilon_p_mat_nom[mask,:], axis=0, keepdims=True) /\
                                                      np.nanmean(epsilon_p_mat_denom[mask,:], axis=0, keepdims=True)
            epsilon_n[mask,:] += self._EM_step_size * np.nanmean(epsilon_n_mat_nom[mask,:], axis=0, keepdims=True) /\
                                                      np.nanmean(epsilon_n_mat_denom[mask,:], axis=0, keepdims=True)
        
        self.propensity = propensity
        self.epsilon_p = epsilon_p
        self.epsilon_n = epsilon_n
        
        
        def p_r1(clicks, sessions, big_list_index):
            y = clicks + 0.
            inv_index = np.argsort(sessions, axis=1)
            for i in range(sessions.shape[0]):
                y[i,:] = y[i,inv_index[i]]
            session_p_r1 = y * (p_e1_r1_c1[big_list_index, :y.shape[1]]) + (1 - y) * (p_e1_r1_c0 + p_e0_r1_c0)[big_list_index, :y.shape[1]]
            
            return np.clip(session_p_r1.mean(0), 0, 1)
        
        return p_r1
    
    def _affine_debias(self, clicks, sessions, biglist_index):
        beta = self.propensity[biglist_index,:] * self.epsilon_n[biglist_index,:]
        alpha = self.propensity[biglist_index,:] * self.epsilon_p[biglist_index,:] - beta
        
        beta = beta[:, :sessions.shape[1]]
        alpha = alpha[:, :sessions.shape[1]]
        alpha[alpha <= 1.e-6] = 1000. # relevant and non-relevant not distinguishable -> assume non relevant

        gamma = (clicks - beta) / alpha
            
        inv_index = np.argsort(sessions, axis=1)
        for i in range(sessions.shape[0]):
            gamma[i,:] = gamma[i,inv_index[i]]
        gamma = gamma.mean(axis=0)
#         return gamma
        return np.clip(gamma, 0, 100)
        
            
        
    def _IPS_expectation(self, predicted, clicks, sessions, dlr):
        predicted = reshape_predicted(predicted, dlr, sessions)
        labels = stack_padded_clicks(clicks, predicted.shape[1])
            
        gamma = normalize(predicted)
        
        
        p_e1_r0_c0 = self.propensity * \
            (1 - gamma) / (1 - self.propensity * gamma)
        p_e0_r1_c0 = (1 - self.propensity) * gamma / \
            (1 - self.propensity * gamma)
            
        
        propensity = (1 - self._EM_step_size) * self.propensity
        
        
        propensity_mat = labels + (1 - labels) * (p_e1_r0_c0)

        unique_groups = np.unique(self.group_ids)
         
        for group in unique_groups:
            mask = self.group_ids == group
            propensity[mask,:] += self._EM_step_size * np.nanmean(propensity_mat[mask,:], axis=0, keepdims=True)
        
        self.propensity = propensity
        
        
        def p_r1(clicks, sessions, big_list_index):
            y = clicks + 0.
            inv_index = np.argsort(sessions, axis=1)
            for i in range(sessions.shape[0]):
                y[i,:] = y[i,inv_index[i]]
            session_p_r1 = y + (1 - y) * (p_e0_r1_c0)[big_list_index, :y.shape[1]]
            
            return np.clip(session_p_r1.mean(0), 0, 1)
        
        return p_r1
    
    
    def _IPS_debias(self, clicks, sessions, biglist_index):
        alpha = self.propensity[biglist_index,:]
        
        alpha = alpha[:, :sessions.shape[1]]
        alpha[alpha <= 1.e-6] = 1000.
        gamma = clicks / alpha
            
        inv_index = np.argsort(sessions, axis=1)
        for i in range(sessions.shape[0]):
            gamma[i,:] = gamma[i,inv_index[i]]
        gamma = gamma.mean(axis=0)
#         return gamma
        return np.clip(gamma, 0, 100)
    
    def get_params(self):
        params = {}
        if self.propensity is not None:
            unique_groups = np.unique(self.group_ids)
            for group in unique_groups:
                group_id = np.where(self.group_ids==group)[0][0]
                params[str(group)] = {
                    'propensity': list(self.propensity[group_id, :]),
                    'count': len(np.where(self.group_ids==group)[0])
                }
                if hasattr(self, 'epsilon_p'):
                    params[str(group)]['epsilon_p'] = list(self.epsilon_p[group_id, :])
                if hasattr(self, 'epsilon_n'):
                    params[str(group)]['epsilon_n'] = list(self.epsilon_n[group_id, :])
                
        return params

