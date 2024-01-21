import numpy as np
from copy import deepcopy as copy_deepcopy
from scipy import sparse
import MDPSolver
import sys
import matplotlib.pyplot as plt

import gym
from gym import spaces as gym_spaces


class Environment:
    def __init__(self, env_args, M=None):
        # initialise MDP parameters
        self.R_max = env_args["R_max"]
        self.terminal_state = env_args["terminal_state"]
        self.actions = {"left": 0, "right": 1, "pick": 2}
        self.actions_names = ["left", "right", "pick"]
        self.n_actions = len(self.actions)
        self.chain_len = env_args["chain_len"]
        self.n_states = self.chain_len * 2 if self.terminal_state == 0 else self.chain_len * 2 + 1
        self.key_position = 1
        self.gamma = env_args["gamma"]
        self.randomMoveProb = env_args["randomMoveProb"]
        self.n2_subgoal = env_args["n2_subgoal"]
        self.InitD = self.get_init_D()
        self.reward = self.get_reward()
        self.T = self.get_transition_matrix_line_key(self.n_states, self.n_actions, self.randomMoveProb,
                                                     self.terminal_state)
        self.T_sparse_list = self.get_transition_sparse_list()
        self.goal_states = self.get_goal_states()
        self.state_feature_matrix = self.get_state_feature_matrix()
        self.state_action_feature_matrix = self.get_state_action_feature_matrix()

        # # declare open gym necessary attributes
        self.observation_space = gym_spaces.Box(low=0.0, high=1.0, shape=(self.n_states,),
                                                dtype=float)
        self.action_space = gym_spaces.Discrete(self.n_actions)
        self.state = None
        self.H = env_args["H"]
        self.current_step = 1

    def reset(self):
        # making a deepcopy of the state is important here
        state = np.random.choice(range(self.n_states), p=self.InitD)
        self.state = copy_deepcopy(state)
        return state

    # enddef

    def step(self, action: int):
        done = False
        # return: next_state, reward, done, info
        self.state, reward = self.sample_next_state_and_reward(self.state, action)

        if self.terminal_state == 1 and self.state == self.n_states - 1:
            done = True

        return self.state, reward, done, None

    # enddef

    def get_reward(self):
        reward = np.zeros((self.n_states, self.n_actions))
        if self.terminal_state == 1:
            reward[-2, 1] = self.R_max
            # reward[-2, 0] = self.R_max
        else:
            reward[-1, 1] = self.R_max
            # reward[-1,0] = self.R_max

        return reward

    # enddef

    def sample_next_state_and_reward(self, s, a):
        reward = self.reward[s, a]
        next_s = self.sample_next_state(s, a)
        return next_s, reward

    # enddef

    def get_state_feature_matrix(self):
        state_feature_matrix = np.zeros((self.n_states, self.n_states))
        for s in range(self.n_states):
            state_feature_matrix[s, s] = 1
        return state_feature_matrix

    # enddef

    def get_state_action_feature_matrix(self):
        state_action_feature_matrix = np.zeros((self.n_states * self.n_actions,
                                                self.n_states * self.n_actions))
        for s_a in range(self.n_states * self.n_actions):
            state_action_feature_matrix[s_a, s_a] = 1
        return state_action_feature_matrix

    # enddef

    def sample_next_state_and_reward(self, s, a):
        reward = self.reward[s, a]
        next_s = self.sample_next_state(s, a)
        return next_s, reward

    # enddef

    def sample_next_state(self, s, a):
        next_s = np.random.choice(np.arange(0, self.n_states, dtype="int"),
                                  size=1, p=self.T[s, :, a])[0]
        return next_s

    # enddef

    def get_goal_states(self):
        if self.terminal_state == 1:
            goal_states = [self.n_states - 2]
        else:
            goal_states = [self.n_states - 1]
        return goal_states

    # enddef

    def get_init_D(self):
        InitD = np.zeros(self.n_states) / self.n_states
        InitD[self.chain_len // 2] = 1
        return InitD

    # enddef

    def get_next_state(self, s_t, a_t):
        next_state = np.random.choice(np.arange(0, self.n_states, dtype="int"), size=1, p=self.T[s_t, :, a_t])[0]
        return next_state

    # enddef

    def get_transition_sparse_list(self):
        T_sparse_list = []
        for a in range(self.n_actions):
            T_sparse_list.append(sparse.csr_matrix(self.T[:, :, a]))  # T_0
        return T_sparse_list

    # endef

    def get_M_0(self):
        M_0 = (self.n_states, self.n_actions, self.reward, self.T, self.gamma, self.terminal_state)
        return M_0

    # enddef

    def draw_policy(self, pi):
        string = "["
        for s in range(self.n_states):
            if pi[s] == 0:
                string += "<-,"
            if pi[s] == 1:
                string += "->,"
        string += ']'
        print(string)

    # enddef

    def plot_reward(self, reward, fignum=1, show=True):

        M = reward.shape[0]
        N = 1
        r_0 = reward[:, 0].reshape(M, N)
        r_1 = reward[:, 1].reshape(M, N)

        xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))  # vertices of the little squares
        xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
        x = np.concatenate([xv.ravel(), xc.ravel()])
        y = np.concatenate([yv.ravel(), yc.ravel()])
        cstart = (M + 1) * (N + 1)  # indices of the centers

        values = np.array([r_0, r_1])

        triangles1 = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M) for j in range(N) for i in
                      range(M)]

        triangles2 = [(i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M) for j in range(N) for i in range(M)]
        triangul = [Triangulation(x, y, triangles) for triangles in [triangles1, triangles2]]

        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)
        fig.num = fignum

        imgs = [
            ax.tripcolor(t, val.ravel(), cmap='RdYlGn', edgecolors="white", vmin=np.min(reward), vmax=np.max(reward))
            for t, val in zip(triangul, values)]

        for val, dir in zip(values, [(0, 1), (0, -1)]):
            for i in range(M):
                for j in range(N):
                    v = val[i, j]
                    ax.text(i + 0.3 * dir[1], j + 0.3 * dir[0], f'{v:.2f}',
                            color='black',
                            ha='center', va='center')

        cbar = fig.colorbar(imgs[0], ax=ax)
        if show:
            plt.show()

    # enddef

    def get_transition_matrix_line_key(self, n_states, n_actions, randomMove, terminal_state):
        unif_prob = randomMove
        success_prob = 1 - randomMove
        P = np.zeros((n_states, n_states, n_actions))

        for s in range(self.chain_len):
            if s > 0 and s < self.chain_len - 1:
                # left action
                P[s, s + 1, 0] = unif_prob
                P[s, s - 1, 0] = success_prob
                # right action
                P[s, s - 1, 1] = unif_prob
                P[s, s + 1, 1] = success_prob

                # pick action
                if self.key_position == s:
                    P[s, self.chain_len, 2] = 1
                else:
                    P[s, s, 2] = 1

            if s == 0:
                if self.terminal_state == 1:
                    # left action
                    P[s, -1, 0] = 1

                    # right action
                    P[s, -1, 1] = unif_prob
                    P[s, s + 1, 1] = success_prob
                else:
                    # left action
                    P[s, s + 1, 0] = unif_prob
                    P[s, s, 0] = success_prob

                    # right action
                    P[s, s, 1] = unif_prob
                    P[s, s + 1, 1] = success_prob

                # pick action
                P[s, s, 2] = 1

            if s == self.chain_len - 1:
                # left action
                P[s, s, 0] = unif_prob
                P[s, s - 1, 0] = success_prob
                # right action
                P[s, s - 1, 1] = unif_prob
                P[s, s, 1] = success_prob

                # pick action
                P[s, s, 2] = 1

        for s in range(self.chain_len, n_states):

            if s > self.chain_len and s < n_states - 1:
                # left action
                P[s, s + 1, 0] = unif_prob
                P[s, s - 1, 0] = success_prob
                # right action
                P[s, s - 1, 1] = unif_prob
                P[s, s + 1, 1] = success_prob

                # pick action
                P[s, s, 2] = 1

            if s == self.chain_len:
                # left action
                P[s, s + 1, 0] = unif_prob
                P[s, s, 0] = success_prob
                # right action
                P[s, s, 1] = unif_prob
                P[s, s + 1, 1] = success_prob

                # pick action
                P[s, s, 2] = 1

            if s == n_states - 1:
                # left action
                P[s, s, 0] = unif_prob
                P[s, s - 1, 0] = success_prob
                # right action
                P[s, s - 1, 1] = unif_prob
                P[s, s, 1] = success_prob

                # pick action
                P[s, s, 2] = 1

        if terminal_state == 1:
            # Now, if the MDP, has a terminal state
            # second last state should transit to tje terminal state
            P[n_states - 2, :, :] = 0
            P[n_states - 2, n_states - 1, 1] = 1

            # Left action
            P[n_states - 2, n_states - 1, 0] = unif_prob
            P[n_states - 2, n_states - 3, 0] = success_prob
            # pick action
            P[n_states - 2, n_states - 2, 2] = 1

            # For the terminal state, all actions lead to the terminal state itself
            P[n_states - 1, :, :] = 0
            P[n_states - 1, n_states - 1, :] = 1

        return P
    # enddef

    def haskey(self, s):
        if s >= self.chain_len:
            return True
        return False
    #enddef



# endclass

########################################
if __name__ == "__main__":
    pass
