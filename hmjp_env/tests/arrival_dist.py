from hmmjp_env.hmjp_sustain import HMMJPEnvSustAct, HMMParameters
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

def comp_arr_sequence():
    pass


if __name__ == '__main__':
    params = HMMParameters()
    env = HMMJPEnvSustAct(params=params, t_terminal=1000)

    #n_steps = 20000
    env.reset()
    #for step in range(n_steps):
    env.step(0)
    while not env.done:
        if env.current_step == 500:
            env.step(0)
        else:
            env.step(0)

    print(env.arrivals)
    res = residuals(env.arrivals, env.intensity_estimate_vec, env.time_vec)
    print(res)
    print(modelcheck(res))


