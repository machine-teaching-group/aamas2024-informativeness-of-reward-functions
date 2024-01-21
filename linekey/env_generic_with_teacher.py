import numpy as np
import torch
import gym
import copy
import linekey_reward_design
import MDPSolver
import os
import utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EnvTeacher(gym.Env):

    def __init__(self, env, args, teacher_name, epsilon_greedy):
        super(EnvTeacher, self).__init__()
        # self.teachers = ["Orig", "ExploB", "SelfRS", "ExploRS", "SORS_with_Rbar", "LIRPG_without_metagrad"]
        self.teachers = ["Orig", "Ada_TL", "EXPRD", "Invariance",
                         "Ada_LL", "Ada_TL_Transfer", "Ada_TL_25", "Ada_TL_50",
                         "Ada_TL_Uniform", "Ada_TL_reinforce_policy"]
        if teacher_name not in self.teachers:
            print("Error!!!")
            print(teacher_name)
            print("Teacher name should be one of the following: {}".format(self.teachers))
            exit(0)
        self.env = env
        self.args = args
        self.teacher_name = teacher_name
        self.epsilon_greedy = epsilon_greedy
        self.use_clipping = args["use_clipping"]

        # declare open gym necessary attributes
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.gamma = env.gamma

        # self.phi_SelfRS = np.zeros((self.n_states, self.n_actions))
        self.phi_Ada_TL = np.zeros((self.n_states, self.n_actions))
        self.phi_Ada_TL_Transfer = np.zeros((self.n_states, self.n_actions))
        self.phi_EXPRD = np.zeros((self.n_states, self.n_actions))
        self.phi_Invariance = np.zeros((self.n_states, self.n_actions))
        self.phi_Ada_LL = np.zeros((self.n_states, self.n_actions))
        # self.phi_SelfRS = np.zeros((self.n_states, self.n_actions))
        self.V = np.zeros(self.n_states)
        self.curr_state = None
        self.first_succ_episode_number = None
        self.count_episode = 0.0
        self.goal_visits = 0.0
        self.episode_goal_visited = None
        self.tol = 1e-6

        # ### Teacher's optimal attributes #####
        # self.Q_T_optimal, self.V_T_optimal, _, self.pi_teacher_stoch_optimal = MDPSolver.valueIteration(self.env,
        #                                                                         self.env.reward)
        # self.A_T_optimal = self.Q_T_optimal - np.repeat(self.V_T_optimal,
        #                                                 self.env.n_actions).reshape(self.env.n_states, self.env.n_actions)
        # _, self.mu_s_T_optimal = MDPSolver.compute_mu_s_a_given_policy_linear_program(env,
        #                                                                       policy=self.pi_teacher_stoch_optimal)



        if self.teacher_name == "Ada_TL":
            _, _, self.pi_d_T, self.pi_s_T = MDPSolver.valueIteration(self.env,
                                                                      self.env.reward)

            self.Q_T, self.V_T, self.A_T, self.mu_s_T = \
                utils.compute_teacher_quantities_given_env_pi(self.env,
                                                              self.pi_s_T,
                                                            epsilon=self.epsilon_greedy,
                                                              tol=self.tol)
            self.delta_s_array_orig = utils.get_delta_s_given_policy(self.env,
                                                                  self.pi_d_T,
                                                                  self.pi_s_T,
                                                                     tol=self.tol)
        elif self.teacher_name == "EXPRD":
            _, _, self.pi_d_T, self.pi_s_T = MDPSolver.valueIteration(self.env,
                                                                      self.env.reward)
            self.delta_s_array_orig = utils.get_delta_s_given_policy(self.env,
                                                                  self.pi_d_T,
                                                                  self.pi_s_T, tol=self.tol)
            self.H_set = [1, 4, 8, 16, 32]
            self.s_active = range(0, self.env.n_states)

            _, self.phi_EXPRD = linekey_reward_design.reward_design_model_based(env_0=self.env,
                                                                                     pi_d_T=self.pi_d_T,
                                                                                     pi_s_T=self.pi_s_T,
                                                                                     pi_d_L=None,
                                                                                     pi_s_L=None,
                                                                                     R_max=self.env.R_max,
                                                                                     H_set=self.H_set,
                                                                                     s_active=self.s_active,
                                                                                     delta_s_array=self.delta_s_array_orig,
                                                                                     Adv_orig_pi_T=None,
                                                                                     mu_pi_T_s=None,
                                                                                     tol=self.tol,
                                                                                     teacher_name=self.teacher_name)

        elif self.teacher_name == "Invariance":
            _, _, self.pi_d_T, self.pi_s_T = MDPSolver.valueIteration(self.env,
                                                                      self.env.reward)
            self.delta_s_array_orig = utils.get_delta_s_given_policy(self.env,
                                                                     self.pi_d_T,
                                                                     self.pi_s_T, tol=self.tol)

            _, self.phi_Invariance = linekey_reward_design.reward_design_model_based(env_0=self.env,
                                                                                 pi_d_T=self.pi_d_T,
                                                                                 pi_s_T=self.pi_s_T,
                                                                                 pi_d_L=None,
                                                                                 pi_s_L=None,
                                                                                 R_max=self.env.R_max,
                                                                                 H_set=None,
                                                                                 s_active=None,
                                                                                 delta_s_array=self.delta_s_array_orig,
                                                                                 Adv_orig_pi_T=None,
                                                                                 mu_pi_T_s=None,
                                                                                 tol=self.tol,
                                                                                 teacher_name=self.teacher_name)


    def step(self, action):
        self.curr_state = self.env.state
        if (self.curr_state == self.n_states - 2) and \
            (not self.episode_goal_visited):
            self.episode_goal_visited = True
            self.goal_visits += 1.0

        next_state, reward_orig, done, info = self.env.step(action)

        if self.teacher_name in ["Orig"]:
            r_hat = reward_orig

        elif self.teacher_name in ["Ada_TL", "EXPRD", "Invariance"]:
            r_hat = self.get_r_addition_term(self.curr_state, action)

        # elif self.teacher_name in ["Non_Ada_PBRS", "Non_Ada_PBRS_sub"]:
        #     r_hat = self.get_r_addition_term(self.curr_state, action)


        # elif self.teacher_name in ["ExploB", "ExploRS"]:
        #     r_hat = reward_orig + self.get_r_addition_term(self.curr_state, action, next_state)
        #
        #     #update R_explore
        #     self.update_ExploB_given_state(self.curr_state)
        #
        # elif self.teacher_name in ["SORS_with_Rbar"]:
        #     r_hat = reward_orig + self.get_r_addition_term(self.curr_state, action)
        #
        # elif self.teacher_name in ["LIRPG_without_metagrad"]:
        #     r_hat = reward_orig + self.get_r_addition_term(self.curr_state, action)

        else:
            print("Error in TeacherEnv.step()  ")
            print("Teacher name should be one of the following: {}".format(self.teachers))
            exit(0)

        return next_state, r_hat, done, info
    # enddef

    def reset(self):
        self.episode_goal_visited = False
        return self.env.reset()
    # enddef

    def get_r_addition_term(self, state, action):
        if self.teacher_name == "Orig":
            return 0

        elif self.teacher_name in ["Ada_TL", "Ada_TL_25", "Ada_TL_50",
                                   "Ada_TL_Uniform", "Ada_TL_reinforce_policy"]:
            return self.R_Ada_TL(state, action)

        elif self.teacher_name in ["EXPRD"]:
            return self.R_EXPRD(state, action)

        elif self.teacher_name in ["Invariance"]:
            return self.R_Invariance(state, action)

        elif self.teacher_name == "Ada_TL_Transfer":
            return self.R_Ada_TL_Transfer(state, action)

        elif self.teacher_name == "Ada_LL":
            return self.R_Ada_LL(state, action)

        elif self.teacher_name == "Non_Ada_PBRS":
            return self.R_Non_Ada_PBRS(state, action)

        elif self.teacher_name == "Non_Ada_PBRS_sub":
            return self.R_Non_Ada_PBRS_sub(state, action)

        elif self.teacher_name == "Non_Ada_TPopL":
            return self.R_Non_Ada_TPopL(state, action)

        else:
            print("Error in TeacherEnv.get_r_addition_term()  ")
            print("Teacher name should be one of the following: {}".format(self.teachers))
            exit(0)
    # enddef

    def R_Ada_TL(self, state, action):
        return self.phi_Ada_TL[state, action]
    # enddef

    def R_EXPRD(self, state, action):
        return self.phi_EXPRD[state, action]
    #enddef
    def R_Invariance(self, state, action):
        return self.phi_Invariance[state, action]
    #enddef

    def R_Ada_TL_Transfer(self, state, action):
        return self.phi_Ada_TL_Transfer[state, action]
    # enddef

    def R_Ada_LL(self, state, action):
        return self.phi_Ada_LL[state, action]
    # enddef

    def R_Non_Ada_PBRS(self, state, action):
        return self.phi_Non_Ada_PBRS[state, action]
    #enddef

    def R_Non_Ada_PBRS_sub(self, state, action):
        return self.phi_Non_Ada_PBRS_sub[state, action]
    #enddef

    def R_Non_Ada_TPopL(self, state, action):
        return self.phi_Non_Ada_TPopL[state, action]
    #enddef

    def update(self, D, agent=None):

        if self.teacher_name in ["Orig", "EXPRD", "Invariance"]:
            pass

        elif self.teacher_name in ["Ada_TL", "Ada_TL_25", "Ada_TL_50",
                                   "Ada_TL_Uniform", "Ada_TL_reinforce_policy"]:
            self.update_Ada_TL(D, agent)

        elif self.teacher_name == "Ada_TL_Transfer":
            self.update_Ada_TL_Transfer(D)


        elif self.teacher_name == "Ada_LL":
            return self.update_Ada_LL(D)

        elif self.teacher_name == "Non_Ada_PBRS":
            pass

        elif self.teacher_name == "Non_Ada_PBRS_sub":
            pass

        elif self.teacher_name == "Non_Ada_TPopL":
            pass

        else:
            print("Error in TeacherEnv.update()  ")
            print("Teacher name should be one of the following: {}".format(self.teachers))
            exit(0)
    #enddef

    def update_Ada_TL(self, D, agent):

        _, self.phi_Ada_TL = linekey_reward_design.reward_design_model_based(env_0=self.env,
                                                                              pi_d_T=self.pi_d_T,
                                                                              pi_s_T=self.pi_s_T,
                                                                              pi_d_L=None,
                                                                              pi_s_L=agent.actor_policy,
                                                                              R_max=self.env.R_max,
                                                                              H_set=None,
                                                                              s_active=None,
                                                                              delta_s_array=self.delta_s_array_orig,
                                                                              Adv_orig_pi_T=self.A_T,
                                                                              mu_pi_T_s=self.mu_s_T,
                                                                              tol=self.tol,
                                                                              teacher_name=self.teacher_name)

    def update_Ada_EXPRD(self, D, agent):
        pass
    def update_Ada_Invariance(self, D, agent):
        pass

    def update_Ada_TL_Transfer(self, D):
        postprocessed_D = self.postprocess_data(D)

        phi_Ada_TL_Transfer_grad_accumulator = np.zeros((self.n_states, self.n_actions))

        for episode in postprocessed_D:
            for state_i, action_i, _, _, _, pi_L_given_s, _, G_bar in episode:
                # compute gradient
                A_pi_T_R_orig_s_a = self.A_T[state_i, action_i]
                A_pi_L_R_orig_s_a = np.sum(self.A_T[state_i, :] * pi_L_given_s)
                phi_Ada_TL_Transfer_grad_accumulator[state_i, action_i] += pi_L_given_s[action_i] * self.mu_s_T[state_i] * \
                                                                (A_pi_T_R_orig_s_a - A_pi_L_R_orig_s_a)
                phi_Ada_TL_Transfer_grad_accumulator[state_i, :] -= pi_L_given_s[:] * self.mu_s_T[state_i] * \
                                                           pi_L_given_s[action_i] *\
                                                           (A_pi_T_R_orig_s_a - A_pi_L_R_orig_s_a)

        for i in range(self.args["K_update_phi_Ada_TL"]):
            # Update phi_Ada_TL
            self.phi_Ada_TL_Transfer += self.args["eta_phi_Ada_TL"] * phi_Ada_TL_Transfer_grad_accumulator

        if self.args["use_clipping"]:
            self.phi_Ada_TL_Transfer[self.phi_Ada_TL_Transfer > self.args["clipping_epsilon"]] = self.args["clipping_epsilon"]
            self.phi_Ada_TL_Transfer[self.phi_Ada_TL_Transfer < -self.args["clipping_epsilon"]] = -self.args["clipping_epsilon"]
        # print(self.phi_Ada_TL)
        return
    #enddef

    def update_Ada_LL(self, D):

        postprocessed_D = self.postprocess_data(D)

        phi_Ada_LL_grad_accumulator = np.zeros((self.n_states, self.n_actions))

        for episode in postprocessed_D:
            for state_i, action_i, _, _, _, pi_given_s, _, G_bar in episode:
                phi_Ada_LL_grad_accumulator[state_i, action_i] += 1.0 * \
                                                                    (pi_given_s[action_i] * (G_bar - self.V[state_i]))
                phi_Ada_LL_grad_accumulator[state_i, :] += - pi_given_s[:] * \
                                                             (pi_given_s[action_i] * (G_bar - self.V[state_i]))

        # Update phi_
        for i in range(self.args["K_update_phi_Ada_LL"]):
            self.phi_Ada_LL += self.args["eta_phi_Ada_LL"] * phi_Ada_LL_grad_accumulator

        if self.args["use_clipping"]:
            self.phi_Ada_LL[self.phi_Ada_LL > self.args["clipping_epsilon"]] = self.args["clipping_epsilon"]
            self.phi_Ada_LL[self.phi_Ada_LL < -self.args["clipping_epsilon"]] = -self.args["clipping_epsilon"]
        # print(self.phi_SelfRS)

        # declare gradient for critic to do average update
        critic_grad = np.zeros(self.n_states)
        total_critic = 0.0
        # V update
        for episode in postprocessed_D:
            for state, _, _, _, _, _, _, G_bar in episode[::-1]:
                delta = (G_bar - self.V[state])
                critic_grad[state] += delta
                total_critic += 1.0

        # Update gradient critic
        self.V += self.args["eta_critic"] * critic_grad / total_critic
        # print(self.V)
        return
    # enddef

    def get_potential_based_reward(self, env, tol=1e-10):
        R_pot = copy.deepcopy(env.reward)
        Q, V, pi_d, pi_s = MDPSolver.valueIteration(env, env.reward, tol=tol)
        for s in range(env.n_states):
            for a in range(env.n_actions):
                sum_over_next_states = 0
                for n_s in range(env.n_states):
                    sum_over_next_states += env.gamma * env.T[s, n_s, a] * V[n_s]
                R_pot[s, a] += sum_over_next_states - V[s]

        return R_pot
    # enddef

    def get_potential_based_reward_sub_opt(self, env, tol=1e-10):
        policy_sub_opt = self.read_sub_opt_policy_from_file()
        R_pot = copy.deepcopy(env.reward)
        _, V_pi = MDPSolver.computeValueFunction_bellmann(env, policy_sub_opt, env.reward)
        for s in range(env.n_states):
            for a in range(env.n_actions):
                sum_over_next_states = 0
                for n_s in range(env.n_states):
                    sum_over_next_states += env.gamma * env.T[s, n_s, a] * V_pi[n_s]
                R_pot[s, a] += sum_over_next_states - V_pi[s]

        return R_pot
    # enddef

    def get_TPopL_reward(self):
        population_policies = self.read_population_policies_from_file()

        phi_non_Ada_TPopL_grad_accumulator = np.zeros((self.n_states, self.n_actions))
        for pi_L in population_policies:
            for state in range(self.env.n_states):
                for action in range(self.env.n_actions):
                    # compute gradient
                    A_pi_T_R_orig_s_a = self.A_T[state, action]
                    A_pi_L_R_orig_s_a = np.sum(self.A_T[state, :] * pi_L[state, :])
                    phi_non_Ada_TPopL_grad_accumulator[state, action] += pi_L[state, action] * self.mu_s_T[state] * \
                                                                    (A_pi_T_R_orig_s_a - A_pi_L_R_orig_s_a) * \
                                                                    (1 - pi_L[state, action])

        for i in range(self.args["K_update_Non_Ada_TPopL"]):
            # Update phi_Ada_TL
            self.phi_Non_Ada_TPopL += self.args["eta_phi_Non_Ada_TPopL"] * \
                                      phi_non_Ada_TPopL_grad_accumulator * \
                                      (1/len(population_policies))

        if self.args["use_clipping"]:
            self.phi_Non_Ada_TPopL[self.phi_Non_Ada_TPopL > self.args["clipping_epsilon"]] = self.args["clipping_epsilon"]
            self.phi_Non_Ada_TPopL[self.phi_Non_Ada_TPopL < -self.args["clipping_epsilon"]] = -self.args["clipping_epsilon"]

        self.phi_Non_Ada_TPopL[self.phi_Non_Ada_TPopL > 0] = self.args["clipping_epsilon"]
        self.phi_Non_Ada_TPopL[self.phi_Non_Ada_TPopL < 0] = -self.args["clipping_epsilon"]

        # print(self.phi_Non_Ada_TPopL)
        # print(len(population_policies))
        # print("=========================")
        # #
        # for s in range(self.env.n_states):
        #     for a in range(self.env.n_actions):
        #         if self.pi_teacher_stoch[s, a] > 0:
        #             print("opt={}".format(self.phi_Non_Ada_TPopL[s, a]))
        #         else:
        #             print("Non-opt={}".format(self.phi_Non_Ada_TPopL[s, a]))
        # input()
        # # exit()
        pass
    #enddef

    def read_population_policies_from_file(self):
        env = self.env
        path = self.args["PopL_policies_subgoal=0.0".format(self.env.n2_subgoal)]

        population_array = []
        pi_array = np.loadtxt(path).flatten()
        pi_array = pi_array.reshape(len(pi_array) // (env.n_states * env.n_actions),
                                    env.n_states, env.n_actions)

        dict_exp_reward_pi = {}

        _, V_pi = MDPSolver.computeValueFunction_bellmann(env, pi_array[0], env.reward, tol=1e-6)
        exp_reward_old = np.dot(env.InitD, V_pi)
        for pi in pi_array:
            _, V_pi = MDPSolver.computeValueFunction_bellmann(env, pi, env.reward, tol=1e-6)
            exp_reward = np.dot(env.InitD, V_pi)
            key = np.round(exp_reward, 5)
            if key not in dict_exp_reward_pi:
                dict_exp_reward_pi[key] = pi

        # sort dict by key
        import collections
        sorted_dict_exp_reward_pi = collections.OrderedDict(
            sorted(dict_exp_reward_pi.items()))

        final_pi_pop_array = []
        last_exp_reward = -np.Inf

        for key_exp_reward, value_pi in sorted_dict_exp_reward_pi.items():
            if np.abs(key_exp_reward - last_exp_reward) > self.args["epsilon_gap_improvent_PopL"]:
                last_exp_reward = copy.deepcopy(key_exp_reward)
                final_pi_pop_array.append(value_pi)

        return np.array(final_pi_pop_array)

    # enddef

    def read_sub_opt_policy_from_file(self):
        env = self.env
        path = self.args["sub_opt_policy_path".format(self.env.n2_subgoal)]

        policy = np.loadtxt(path)
        return policy
    #endddef

    def softmax_prob(self, a, b):
        return np.exp(a) / (np.exp(a) + np.exp(b))

    def postprocess_data(self, D):

        postprocessed_D = []

        # episode --> [state, action, \hat{r}, next_state, \hat(G), \pi(.|state)]
        for episode in D:
            # postProcessed episode --> [state, action, \hat{r}, next_state, \hat(G), \pi(.|state), \bar{r}, \bar{G}]
            postprocessed_epidata = self.get_postposessed_episode(self.env, episode)

            # add postprocessed episode
            postprocessed_D.append(postprocessed_epidata)

        return postprocessed_D
    #enddef

    def get_postposessed_episode(self, env_orig, episode):

        postprocessed_epidata = []
        for t in range(len(episode)):
            state, action, r_hat, next_state, G_hat, pi_given_s = episode[t]

            # get original reward
            r_bar = self.get_original_reward(env_orig, state, action)

            # postProcessed episode --> [state, action, \hat{r}, next_state, \hat(G), \pi(.|state), \bar{r}, \bar{G}]
            e_t = [state, action, r_hat, next_state, G_hat, pi_given_s, r_bar, 0.0]

            postprocessed_epidata.append(e_t)

        # compute return \bar{G} for every (s, a)
        G_bar = 0  # original return
        for i in range(len(postprocessed_epidata) - 1, -1, -1):  # iterate backwards
            _, _, _, _, _, _, r_bar, _ = postprocessed_epidata[i]
            G_bar = r_bar + env_orig.gamma * G_bar
            postprocessed_epidata[i][7] = G_bar  # update G_bar in episode

        if postprocessed_epidata[0][7] > 0.0 and self.first_succ_episode_number is None:
            self.first_succ_episode_number = copy.deepcopy(self.count_episode)
        else:
            self.count_episode += 1.0

        return postprocessed_epidata
    #enddef

    def get_original_reward(self, env_orig, state, action):
        r_bar = env_orig.reward[state, action]
        return r_bar
    #enddef

    def get_pairwise_data_using_return(self, postprocessed_D):
        pairwise_data = []

        for i, episode_i in enumerate(postprocessed_D):
            for j, episode_j in enumerate(postprocessed_D):
                G_bar_i = episode_i[0][7]
                G_bar_j = episode_j[0][7]
                if G_bar_i > G_bar_j:
                    # \tau_i > \tau_j
                    pairwise_data.append([i, j])

        return pairwise_data
    # enndef

    def indicator(self, state, action, s, a):
        return 1.0 if (state == s and action == a) else 0.0
    #enddef

    # def compute_teacher_quantities_given_env_pi(self, env, policy):
    #     Q_T, V_T = MDPSolver.computeValueFunction_bellmann(env, policy,
    #                                                                  env.reward, tol=self.tol)
    #     A_T = Q_T - np.repeat(V_T, env.n_actions).reshape(env.n_states,
    #                                                                           env.n_actions)
    #     _, mu_s_T = MDPSolver.compute_mu_s_a_given_policy_linear_program(env,
    #                                                                               policy=policy)
    #     return Q_T, V_T, A_T, mu_s_T

    def get_R_hat_vecor(self):

        if self.teacher_name in ["Orig"]:
            r_hat_vec = self.env.reward

        elif self.teacher_name in ["EXPRD"]:
            r_hat_vec = self.phi_EXPRD

        elif self.teacher_name in ["Invariance"]:
            r_hat_vec = self.phi_Invariance

        elif self.teacher_name in ["Ada_TL", "Ada_TL_25", "Ada_TL_50",
                                   "Ada_TL_Uniform", "Ada_TL_reinforce_policy"]:
            r_hat_vec = self.env.reward + self.phi_Ada_TL

        elif self.teacher_name in ["Ada_TL_Transfer"]:
            r_hat_vec = self.env.reward + self.phi_Ada_TL_Transfer


        elif self.teacher_name in ["Ada_LL"]:
            r_hat_vec = self.env.reward + self.phi_Ada_LL

        else:
            print("Error in TeacherEnv.get_R_hat_vecor()  ")
            print("Teacher name should be one of the following: {}".format(self.teachers))
            exit(0)
        return r_hat_vec
    #enddef

    def get_R_phi(self):

        if self.teacher_name in ["Orig"]:
            r_phi = self.env.reward

        elif self.teacher_name in ["EXPRD"]:
            r_phi = self.phi_EXPRD

        elif self.teacher_name in ["Invariance"]:
            r_phi = self.phi_Invariance

        elif self.teacher_name in ["Ada_TL", "Ada_TL_25", "Ada_TL_50",
                                   "Ada_TL_Uniform", "Ada_TL_reinforce_policy"]:
            r_phi = self.phi_Ada_TL

        elif self.teacher_name in ["Ada_TL_Transfer"]:
            r_phi = self.phi_Ada_TL_Transfer

        elif self.teacher_name in ["Ada_LL"]:
            r_phi = self.phi_Ada_LL

        else:
            print("Error in TeacherEnv.get_R_hat_vecor()  ")
            print("Teacher name should be one of the following: {}".format(self.teachers))
            exit(0)
        return r_phi
    #enddef


    def get_IR(self, R_vector, pi_L):
        Adv_T_orig_opt = self.A_T_optimal
        mu_T_s_optimal = self.mu_s_T_optimal
        Adv_pi_L_orig_opt = np.sum(Adv_T_orig_opt * pi_L, axis=1)

        Q_H_1 = R_vector.flatten()  # Q function for H=1
        R_times_pi = R_vector.flatten() * pi_L.flatten()

        V_H_1 = np.sum(R_times_pi.reshape(self.env.n_states,
                                               self.env.n_actions), axis=1)  # V for H=1
        V_H_1_s_a = np.repeat(V_H_1, self.env.n_actions)
        ##################

        Adv_pi_L_H_1 = Q_H_1 - V_H_1_s_a

        # objective computation
        mu_T_s_a = np.repeat(mu_T_s_optimal, self.env.n_actions)
        Adv_pi_L_orig_a = np.repeat(Adv_pi_L_orig_opt, self.env.n_actions)
        pi_L_a = pi_L.flatten()
        Adv_T_orig_s_a = Adv_T_orig_opt.flatten()

        scalar_term = mu_T_s_a * (Adv_T_orig_s_a - Adv_pi_L_orig_a) * pi_L_a

        IR = np.sum(Adv_pi_L_H_1 * scalar_term)
        return IR
    #enddef

    def get_IR_S(self, R_vector, pi_L):
        Adv_T_orig_opt = self.A_T_optimal
        mu_T_s_optimal = self.mu_s_T_optimal
        Adv_pi_L_orig_opt = np.sum(Adv_T_orig_opt * pi_L, axis=1)

        Q_H_1 = R_vector.flatten()  # Q function for H=1
        R_times_pi = R_vector.flatten() * pi_L.flatten()

        V_H_1 = np.sum(R_times_pi.reshape(self.env.n_states,
                                               self.env.n_actions), axis=1)  # V for H=1
        V_H_1_s_a = np.repeat(V_H_1, self.env.n_actions)
        ##################

        Adv_pi_L_H_1 = Q_H_1 - V_H_1_s_a

        # objective computation
        mu_T_s_a = np.repeat(mu_T_s_optimal, self.env.n_actions)
        Adv_pi_L_orig_a = np.repeat(Adv_pi_L_orig_opt, self.env.n_actions)
        pi_L_a = pi_L.flatten()
        Adv_T_orig_s_a = Adv_T_orig_opt.flatten()

        scalar_term = mu_T_s_a * (Adv_T_orig_s_a - Adv_pi_L_orig_a) * pi_L_a

        IR_s_a = Adv_pi_L_H_1 * scalar_term

        IR_S_array = []
        for s in range(self.env.n_states):
            zeros_array = np.zeros(self.env.n_states * self.env.n_actions)
            zeros_array[s*self.env.n_actions: s*self.env.n_actions + self.env.n_actions] = 1.0
            IR_S_array.append(np.dot(zeros_array, IR_s_a))
        return IR_S_array
    #enddef

