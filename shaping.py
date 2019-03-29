import numpy as np

def early_penalty_late_reward(R, T, tau=1e6, p0=0.75, pt=0.05, r0=0.25,
        rt=0.95):
    '''For bipolar reward, decrease penalty for failure and increase reward for
    success over time.
    
    R: reward,
    T: step number,
    tau: time constant,
    p0: initial gain of negative reward,
    pt: terminal gain of negative reward,
    r0: initial gain of positive reward,
    rt: terminal gain of positive reward.
    '''
    if R < 0:
        return (pt + (p0 - pt) * np.exp(-T / tau)) * R
    return (rt + (r0 - rt) * np.exp(-T / tau)) * R
