from hmmjp_env.hmmjpenv import HMMJPEnv, HMMParameters
import matplotlib.pyplot as plt

if __name__ == '__main__':
    params = HMMParameters()
    env = HMMJPEnv(params=params, t_terminal=100)

    n_steps = 10000
    env.reset()
    accummulated_under_observed = []
    for step in range(n_steps):
        if not env.done:
            env.step(0)
            accummulated_under_observed.append(env.accumulated_under_intensity_obs)

    # print(accummulated_under_observed)
    # plt.plot(accummulated_under_observed)
    # plt.show()
    #
    # plt.plot(env.intensity_estimate_vec)
    # plt.axhline(env.lambdainf, color='red')
    # plt.show()

    fig = env.plot_Pi_realization()
    fig.show()

    fig2 = env.plot_intensity_realization()
    fig2.show()

    # print(env.arrivals)
    # fig, ax = plt.subplots()
    # [ax.axvline(ar) for ar in env.arrivals]
    # ax.fill_betweenx([0, 1], 0, 5, color='salmon', alpha=0.5)
    # fig.show()