from hmjp_env.hmjp_sustain import HMJPEnvSustAct, HMMParametersSust
from hmjp_env.hmjp import HMJPEnv, HMJPParameters
import matplotlib.pyplot as plt

def simulate_env(env):
    env.reset()
    # for step in range(n_steps):
    # env.step(1)
    done = False
    rew_vec = []
    ws = []
    ls = []
    i = 0

    while env.current_time < 100:
        if env.current_step == 500:
            s_next, rew, done, _ = env.step(0)
        else:
            s_next, rew, done, _ = env.step(0)
        rew_vec.append(rew)
        ws.append(s_next[1])
        ls.append(s_next[0])
        i += 1

    plt.plot(rew_vec)
    plt.show()

    plt.plot(ws, ls, marker='x')
    plt.show()

    fig = env.plot_Pi_realization()
    fig.show()

    fig2 = env.plot_intensity_realization()
    fig2.show()

    print(f'Episode Length: {i}')

if __name__ == '__main__':
    # params = HMMParametersSust()
    # env = HMJPEnvSustAct(params=params, t_terminal=100, track_intensity=True, reward_shaping='continuous')

    params = HMJPParameters()
    env = HMJPEnv(params=params, t_terminal=100, track_intensity=True, reward_shaping='discrete')
    simulate_env(env)