from hmjp_env.hmjp_sustain import HMJPEnvSustAct, HMMParametersSust
from hmjp_env.hmjp import HMJPEnv, HMJPParameters
import matplotlib.pyplot as plt

import numpy as np
from scipy import integrate
from scipy.stats import probplot, expon, kstest


def residuals(arrivals, intensity, t_vec):
    thetas = np.zeros(len(arrivals) - 1)
    for i in range(1, len(arrivals)):
        mask = (t_vec > arrivals[i-1]) & (t_vec < arrivals[i])
        thetas[i - 1] = integrateintensity(t_vec[mask], intensity[mask])
    return thetas


def integrateintensity(t_vec, intensity):
    return integrate.simpson(intensity, t_vec)


def _qqplot(self, theoretical, empirical):

    # fig = plt.figure()
    percs = np.linspace(0, 100, 201)
    qn_a = np.percentile(theoretical, percs)
    qn_b = np.percentile(empirical, percs)

    plt.plot(qn_a, qn_b, ls="", marker="o")

    x = np.linspace(np.min((qn_a.min(), qn_b.min())), np.max((qn_a.max(), qn_b.max())))
    fig, ax = plt.subplots()
    ax.plot(x, x, color="k", ls="--")
    # fig.suptitle('Q-Q plot', fontsize=20)
    ax.xlabel('Theoretical Quantiles', fontsize=18)
    ax.ylabel('Empirical Quantiles', fontsize=16)
    return fig


def modelcheck(thetas):
    # returns 2-sided ks test of Null hypothesis testsuit and thetas are drawn from the same distribution exp(1)
    # return[0] - statistic
    # return[1] - p-value
    # alternatively use this
    # testsuite = np.random.exponential(1, len(thetas))
    # stats.ks_2samp(testsuite, thetas)
    k = probplot(thetas, dist=expon, fit=True, plot=plt, rvalue=False)
    plt.plot(k[0][0], k[0][0], 'k--')
    plt.show()
    return kstest(thetas, 'expon')


def simulate_env(env):
    env.reset()
    # for step in range(n_steps):
    # env.step(1)
    done = False
    rew_vec = []
    ws = []
    ls = []
    i = 0

    while env.current_time < env.t_terminal:
        if env.current_step == 500:
            s_next, rew, done, _ = env.step(0)
        else:
            s_next, rew, done, _ = env.step(0)
        rew_vec.append(rew)
        ws.append(s_next[1])
        ls.append(s_next[0])
        i += 1

    return env


if __name__ == '__main__':
    env = []

    params0 = HMMParametersSust()
    env.append(HMJPEnvSustAct(params=params0, t_terminal=1000, track_intensity=True, reward_shaping='continuous'))

    params1 = HMJPParameters()
    env.append(HMJPEnv(params=params1, t_terminal=1000, track_intensity=True, reward_shaping='discrete'))

    process_index = 1

    simulate_env(env[process_index])
    res = residuals(env[process_index].arrivals, env[process_index].true_intensity(), env[process_index].time_vec)
    # print(res)
    print(modelcheck(res))

    plt.plot(env[process_index].intensity_estimate_vec)


