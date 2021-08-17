from hmmjp_env.hmmjpenv import HMMJPEnv, HMMParameters
import matplotlib.pyplot as plt

if __name__ == '__main__':
    params = HMMParameters()
    env = HMMJPEnv(params=params, t_terminal=100, track_intensity=True)

    #n_steps = 20000
    env.reset()
    #for step in range(n_steps):
    env.step(1)
    done = False
    rew_vec = []
    while not done:
        if env.current_step == 500:
            s_next, rew, done, _ = env.step(1)
        else:
            s_next, rew, done, _ = env.step(0)
        rew_vec.append(rew)
    print('hulipero')
    print(rew_vec)
    plt.plot(rew_vec)
    plt.show()

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