import gym
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass()
class HMMParameters:
    """
    Sample Controlled HMM Parameter dataclass
    """
    p: np.ndarray = np.array([[0.95, 0.1], [0.1, 0.5]])
    Lamb: np.ndarray = np.array([[0.2, 1.0], [3.0, 2.0]])
    lamb: tuple = (0.1, 1.5)
    pa: float = 0.65
    c: float = 10


class HMMJPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, params, t_terminal=100, track_intensity=True):

        self.dt = 0.05
        self.params = params
        self.track_intensity = track_intensity
        self.intensity_estimate = 0.0
        self.t_terminal = t_terminal
        self.true_intensity = track_intensity
        self.reset()

    @property
    def delta1(self):
        return (self.params.lamb[1] - self.params.lamb[0]) + (1 - self.params.p[self.controlled, 1]) * \
               self.params.Lamb[self.controlled, 1] + \
               (1 - self.params.p[self.controlled, 0]) * self.params.Lamb[self.controlled, 0]

    @property
    def delta2(self):
        return np.sqrt(
            self.delta1 ** 2 - 4 * (self.params.lamb[1] - self.params.lamb[0]) *
            self.params.Lamb[self.controlled, 0]
            * (1 - self.params.p[self.controlled, 0]))

    @property
    def pih1(self):
        pih1 = np.divide(self.delta1 - self.delta2,
                         2 * (self.params.lamb[1] - self.params.lamb[0]))
        assert pih1 <= 1, 'pih1 has to be lesser or equal 1.'
        return pih1

    @property
    def pih2(self):
        pih2 = np.divide(self.delta1 + self.delta2,
                         2 * (self.params.lamb[1] - self.params.lamb[0]))
        assert pih2 >= 1, 'pih1 has to be greater or equal 1.'
        return pih2

    @property
    def lambdainf(self):
        return self.params.lamb[0] + (self.params.lamb[ 1] - self.params.lamb[0]) * self.pih1

    def reset(self):
        if self.track_intensity:
            self.time_vec = np.arange(0, self.t_terminal + self.dt * 1, self.dt)
            self.intensity_estimate_vec = np.zeros_like(self.time_vec)
            self.Pih_estimate_vec = np.zeros_like(self.time_vec)

        # observable state variables
        self.current_time = 0.0
        self.current_step = 0
        # dummy variable indicating whether the process is currently controlled
        self.controlled = 0
        # what si the initial intensity guess?

        self.arrivals = [0.0]
        self.done = False

        # unobservable states
        # HMM state state = {0, 1} ~ {L, H}
        self.M = np.random.choice((0, 1))
        self.true_intensity = []
        self.state_arrivals = [0.0]
        self.state_history = [self.M]

        # auxiliary HMM variables
        self.Pih_estimate = self.pih1
        self.Pi_last_jump = self.pih1

        # auxiliary variables simulation
        self.accumulated_under_intensity_obs = 0.0
        self.accumulated_under_intensity_latent = 0.0
        self._draw_observable_transition = np.random.exponential(1)
        self._draw_latent_transition = np.random.exponential(1)

    def decay_Pih(self):
        # probability that HMM is in the state H
        denominator = 1 + np.divide(self.Pi_last_jump - self.pih1,
                                    self.pih2 - self.Pi_last_jump) * np.exp(
            -(self.pih2 - self.pih1) * (self.current_time - self.arrivals[-1]))
        self.Pih_estimate = self.pih2 - np.divide(self.pih2 - self.pih1, denominator)

    def comp_intensity(self):
        if self.controlled == 0:
                self.intensity_estimate = self.params.lamb[1] * self.Pih_estimate + self.params.lamb[0] * (
                    1 - self.Pih_estimate)
        else:
            self.intensity_estimate = self.lambdainf

    def state_transition(self, action: int):
        """
        Determines whether a state transition has arrived.
        :return: None
        """
        if action > 0:
            if np.random.random() < self.params.pa:
                # successful switch to state H
                self.M = 1
            self.controlled = 1

        else:
            self.accumulated_under_intensity_latent += self.dt * self.params.Lamb[self.controlled, self.M]
            if self.accumulated_under_intensity_latent >= self._draw_latent_transition:
                # state transition has occurred
                draw = np.random.random()
                if draw < self.params.p[self.controlled, self.M]:
                    # state remains the same
                    pass
                else:
                    # state changes
                    self.M = np.abs(self.M - 1)

                self.state_arrivals.append(self.current_time)
                self.state_history.append(self.M)
                self.accumulated_under_intensity_latent = 0.0
                self._draw_latent_transition = np.random.exponential(1.0)

    def arrival_event(self):
        self.accumulated_under_intensity_obs += self.dt * self.params.lamb[self.M]
        if self.accumulated_under_intensity_obs >= self._draw_observable_transition:
            self.arrivals.append(self.current_time)
            self.accumulated_under_intensity_obs = 0.0
            self._draw_observable_transition = np.random.exponential(1.0)
            # jump in estimated  intensity
            self.Pih_estimate = np.divide(self.Pih_estimate * self.params.lamb[1],
                                          self.params.lamb[1] * self.Pih_estimate + self.params.lamb[0] * (
                                                  1 - self.Pih_estimate))
            self.Pi_last_jump = self.Pih_estimate
            rew = self.reward()
            self.controlled = 0
        else:
            self.decay_Pih()
            rew = 0
        self.comp_intensity()
        return rew

    def step(self, action: int):
        if action > 0:
            cost = self.params.c
        else:
            cost = 0.0

        self.current_step += 1
        self.current_time += self.dt
        self.state_transition(action)
        rew = self.arrival_event()
        rew -= cost
        # if self.current_step > 2000:
        #     print('Here Jerry!')

        if self.track_intensity:
            self.intensity_estimate_vec[self.current_step - 1] = self.intensity_estimate
            self.Pih_estimate_vec[self.current_step - 1] = self.Pih_estimate

        if self.current_time > self.t_terminal:
            # terminal condition
            self.done = True


        return self.intensity_estimate, rew, self.done, None

    def render(self, mode='human'):
        pass

    # plotting methods
    def plot_intensity_realization(self):
        if not self.track_intensity:

            raise EnvironmentError('Intensity was not tracked.')

        fill_colors = ['salmon', 'green']
        fig, ax = plt.subplots()
        # [ax.axvline(ar, linewidth=0.5, color='black') for ar in self.arrivals]
        # ax1 = ax.twinx()
        for i, st_arr in enumerate(self.state_arrivals[1:], start=1):
            col = fill_colors[self.state_history[i - 1]]
            ax.fill_betweenx([0, self.params.lamb[1]], self.state_arrivals[i - 1], self.state_arrivals[i], color=col,
                             alpha=0.5)
        ax.plot(self.time_vec, self.intensity_estimate_vec)
        ax.axhline(self.params.lamb[0], color='black', linewidth=0.5)
        ax.axhline(self.params.lamb[1], color='black', linewidth=0.5)
        ax.axhline(self.lambdainf, color='black', linewidth=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Intensity')

        return fig

    def plot_Pi_realization(self):
        if not self.track_intensity:
            raise EnvironmentError('Intensity was not tracked.')
        fill_colors = ['salmon', 'green']
        fig, ax = plt.subplots()
        ax.plot(self.time_vec, self.Pih_estimate_vec)
        ax.axhline(self.pih1, color='red', linewidth=0.5)
        # [ax.axvline(line, color='blue') for line in self.state_arrivals]
        # ax.axhline(self.pih2, color='black', linewidth=0.5)
        for i, st_arr in enumerate(self.state_arrivals[1:], start=1):
            col = fill_colors[self.state_history[i - 1]]
            ax.fill_betweenx([0, 1.0], self.state_arrivals[i - 1], self.state_arrivals[i], color=col, alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel(r'$\Pi_H(t)$')

        return fig

    def reward(self):
        return 1.0

    def __str__(self):
        return f'{self.__class__} -- pi1: {self.pih1}, pih2: {self.pih2}, lambdainf: {self.lambdainf}'


if __name__ == '__main__':
    pars = HMMParameters()
    # print(pars.lamb[0])
    env = HMMJPEnv(pars)
    print(env)
    for i in range(10000):
        if not env.done:
            env.step(1)

    print(env)
