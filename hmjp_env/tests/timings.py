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
    import timeit

    env = []

    params1 = HMJPParameters()
    env.append(HMJPEnv(params=params1, t_terminal=100, track_intensity=True, reward_shaping='discrete'))
    env.append(HMJPEnv(params=params1, t_terminal=100, track_intensity=False, reward_shaping='discrete'))



    print(timeit.timeit("simulate_env(env[0])", globals=globals(), number=100))

    print(timeit.timeit("simulate_env(env[1])", globals=globals(), number=100))