import numpy as np
import os
import copy
import MDPSolver


def sample_trajectory_given_state_action(env, agent, start_state, start_action=None, epsilon_reinforce=0.0, H=100):
    env_copy = copy.deepcopy(env)
    _ = env_copy.reset()
    env_copy.state = start_state

    curr_state = start_state
    epidata = []
    iter = 0

    # rollout an entire episode
    while True:

        if start_action is not None and iter == 0:
            action = start_action
        else:
            # pick the action based on the demonstration
            action, _ = agent.predict(curr_state)
        # take a step
        next_state, reward, done, _ = env_copy.step(action)

        # get action distribution of the state
        # print(curr_state)
        pi_given_s = agent.get_action_distribution(curr_state)

        # store the data in epidata # e_t --> [state, action, \hat{r}, next_state, \hat(G), \bar{r}, \bar{G}, \pi(.|state)]
        e_t = [curr_state, action, reward, next_state, 0.0, pi_given_s]
        epidata.append(e_t)

        if done or H < iter:
            break

        # update curr_state
        curr_state = next_state
        iter += 1


    return epidata
#enddef

def generate_sampled_data(env_teacher, agent):

    # reset the environment: Get initial state and demonstration
    curr_state = env_teacher.reset()

    # episode data (epidata)
    epidata = [] # --> [state, action, \hat{r}, next_state, \hat(G), \bar{r}, \bar{G}, \pi(.|state)]

    # rollout an entire episode
    while True:

        # pick the action based on the demonstration
        action, _ = agent.predict(curr_state)

        # take a step
        next_state, reward_hat, done, _ = env_teacher.step(action)

        # get action distribution of the state
        pi_given_s = agent.get_action_distribution(curr_state)

        # store the data in epidata # e_t --> [state, action, \hat{r}, next_state, \hat(G), \bar{r}, \bar{G}, \pi(.|state)]
        e_t = [curr_state, action, reward_hat, next_state, 0.0, pi_given_s]
        epidata.append(e_t)

        if done:
            break

        # update curr_state
        curr_state = next_state

    # compute \hat{G} for every (s, a)
    G_hat = 0  # shaped return
    for i in range(len(epidata) - 1, -1, -1):  # iterate backwards
        _, _, r_hat, _, _, _ = epidata[i]
        G_hat = r_hat + env_teacher.gamma * G_hat
        epidata[i][4] = G_hat  # update G_hat in episode

    return epidata
#enddef



# def get_policy_given_theta__(env, theta):
#     n_actions = env.n_actions
#     n_states = env.n_states
#     if theta.ndim == 1:
#         theta = theta.reshape(n_states, n_actions)
#     # this is for stable softmax (substract max)
#     theta = theta - np.repeat(np.max(theta, axis=1), n_actions).reshape(n_states, n_actions)
#
#     policy = np.exp(theta)/np.repeat(np.sum(np.exp(theta), axis=1),
#                             n_actions).reshape(n_states, n_actions)
#     return policy
# #enddef


def evaluate_agent(env_orig, env_teacher, agent, n_episode):

    episode_reward_array_env_orig = []
    episode_reward_array_env_teacher = []

    for i in range(n_episode):
        episode_reward_env_orig = 0
        episode_reward_env_teacher = 0
        episode = generate_sampled_data(env_teacher,  agent)

        for t in range(len(episode)):
            _, _, r_hat, _, _, _ = episode[t]

            # episode_reward_env_orig += env_orig.gamma**t * r_bar
            episode_reward_env_teacher += env_teacher.gamma**t * r_hat

        episode_reward_array_env_orig.append(episode_reward_env_orig)
        episode_reward_array_env_teacher.append(episode_reward_env_teacher)

    return np.average(episode_reward_array_env_orig), np.average(episode_reward_array_env_teacher)
#enddef


def write_into_file(accumulator, exp_iter, out_folder_name="out_folder", out_file_name="out_file"):
    directory = 'results/{}'.format(out_folder_name)
    filename = out_file_name + '_' + str(exp_iter) + '.txt'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    filepath = directory + '/' + filename
    print("output file name  ", filepath)
    f = open(filepath, 'w')
    for key in accumulator:
        f.write(key + '\t')
        temp = list(map(str, accumulator[key]))
        for j in temp:
            f.write(j + '\t')
        f.write('\n')
    f.close()
#enddef

def compute_teacher_quantities_given_env_pi_H(env, policy):
    Q_T = get_Q_H_given_policy(env, policy)
    V_T = np.max(Q_T, axis=1)

    A_T = Q_T - np.repeat(V_T, env.n_actions).reshape(env.n_states,
                                                      env.n_actions)
    _, mu_s_T = MDPSolver.compute_mu_s_a_given_policy_linear_program(env,
                                                                     policy=policy)
    return Q_T, V_T, A_T, mu_s_T


def compute_teacher_quantities_given_env_pi(env, pi_s_T, epsilon, tol):
    # compute pi_stochastic_epsilon_greedy
    pi_stochastic_epsilon_greedy = copy.deepcopy(pi_s_T)
    pi_stochastic_epsilon_greedy *= (1.0 - epsilon)
    pi_stochastic_epsilon_greedy += epsilon / env.n_actions

    Q_T, V_T = MDPSolver.compute_Q_V_Function_given_policy_st(env, pi_s_T,
                                                              env.reward,
                                                              tol=tol)
    A_T = Q_T - np.repeat(V_T, env.n_actions).reshape(env.n_states,
                                                      env.n_actions)

    _, mu_s_T_eps_greedy = MDPSolver.compute_mu_s_a_given_policy_linear_program(env,
                                                                     policy=pi_stochastic_epsilon_greedy)
    mu_s_T_eps_greedy[-1] = 0.0
    mu_s_T_eps_greedy = mu_s_T_eps_greedy / sum(mu_s_T_eps_greedy)
    return Q_T, V_T, A_T, mu_s_T_eps_greedy

# def compute_teacher_quantities_given_env_pi_soft(env, policy):
#     Q_T, V_T = MDPSolver.computeValueFunction_bellmann(env, policy,
#                                                        env.reward, tol=1e-10)
#     A_T = Q_T - np.repeat(V_T, env.n_actions).reshape(env.n_states,
#                                                       env.n_actions)
#
#     # compute soft pi
#     temp_Q = copy.deepcopy(Q_T)
#     # Softmax by row to interpret these values as probabilities.
#     temp_Q -= temp_Q.max(axis=1).reshape((env.n_states, 1))  # For numerical stability.
#     pi_T_soft = np.exp(temp_Q) / np.exp(temp_Q).sum(axis=1).reshape((env.n_states, 1))
#
#     # compute mu_pi_soft_T
#     _, mu_s_T_soft = MDPSolver.compute_mu_s_a_given_policy_linear_program(env,
#                                                                      policy=pi_T_soft)
#     mu_s_T_soft[-1] = 0.0
#     mu_s_T_soft = mu_s_T_soft / sum(mu_s_T_soft)
#     return Q_T, V_T, A_T, mu_s_T_soft

# def compute_teacher_quantities_given_env_pi_eps_greedy(env, policy_st, epsilon=0.1):
#     Q_T, V_T = MDPSolver.computeValueFunction_bellmann(env, policy_st,
#                                                        env.reward, tol=1e-10)
#     A_T = Q_T - np.repeat(V_T, env.n_actions).reshape(env.n_states,
#                                                       env.n_actions)
#
#     # compute pi_stochastic_epsilon_greedy
#     pi_stochastic_epsilon_greedy = copy.deepcopy(policy_st)
#     pi_stochastic_epsilon_greedy *= (1.0 - epsilon)
#     pi_stochastic_epsilon_greedy += epsilon / env.n_actions
#
#     # compute mu_pi_eps_greedy
#     _, mu_s_T_eps = MDPSolver.compute_mu_s_a_given_policy_linear_program(env,
#                                                                      policy=pi_stochastic_epsilon_greedy)
#     mu_s_T_eps[-1] = 0.0
#     mu_s_T_eps = mu_s_T_eps / sum(mu_s_T_eps)
#     # print(mu_s_T_eps)
#     # exit()
#     return Q_T, V_T, A_T, mu_s_T_eps

def calculate_P_pi_star(env, pi_star):
    n_states = env.n_states
    n_actions = env.n_actions
    P_0 = env.T

    P_pi_star = np.zeros((n_states * n_actions, n_states * n_actions))

    for s in range(n_states):
        for a in range(n_actions):
            for s_n in range(n_states):
                for a_n in range(n_actions):
                    P_pi_star[s * n_actions + a, s_n * n_actions + a_n] = \
                        P_0[s, s_n, a] if a_n == pi_star[s_n] else 0
    return P_pi_star
# enddef
def get_Q_H_given_policy(env, policy):
    P_pi_star = calculate_P_pi_star(env, policy)
    P_H = get_P_H(env_0=env, P_pi_star=P_pi_star, H=env.H)
    Q_H = (P_H @ env.reward.flatten()).reshape(env.n_states, env.n_actions)
    return Q_H
# enddef

def get_P_H(env_0, P_pi_star, H):
    #local (H horizon) optimality constraints
    accumulator = np.eye(P_pi_star.shape[0])
    accumulator_P_star_mult = copy.deepcopy(accumulator)
    gamma = env_0.gamma

    for h in range(1, H):
        accumulator_P_star_mult = accumulator_P_star_mult @ P_pi_star
        accumulator += (gamma**h) * accumulator_P_star_mult

    return accumulator
#enddef

def get_delta_s_given_policy_H(env_given, pi_target_d, pi_target_s, tol=1e-10):
    # Q_pi, _ = MDPSolver.compute_Q_V_Function_given_policy(env_given, pi_target_d,
    #                                                       env_given.reward, tol=tol)
    Q_pi = get_Q_H_given_policy(env_given, policy=pi_target_d)
    # Q_pi_iter, _, _, _ = MDPSolver.valueIteration_H(env_0, env_0.reward)
    n_states = env_given.n_states
    delta_s_array = []
    for s in range(n_states):
        s_a_array = []
        for a in range(env_given.n_actions):
            if pi_target_s[s, a] == 0:
                s_a_array.append(Q_pi[s, pi_target_d[s]] - Q_pi[s, a])
        if len(s_a_array) == 0:
            s_a_array.append(0)
        delta_s_array.append(min(s_a_array))
    return delta_s_array

def get_delta_s_given_policy(env_given, pi_target_d, pi_target_s, tol=1e-10):
    Q_pi, _ = MDPSolver.compute_Q_V_Function_given_policy(env_given, pi_target_d,
                                                          env_given.reward, tol=tol)

    n_states = env_given.n_states
    delta_s_array = []
    for s in range(n_states):
        s_a_array = []
        for a in range(env_given.n_actions):
            if pi_target_s[s, a] == 0:
                s_a_array.append(Q_pi[s, pi_target_d[s]] - Q_pi[s, a])
        if len(s_a_array) == 0:
            s_a_array.append(0)
        delta_s_array.append(min(s_a_array))
    return delta_s_array

def calculate_I_pi_star(env, pi_star):
    n_states = env.n_states
    n_actions = env.n_actions
    I_pi_star = np.zeros((n_states * n_actions, n_states * n_actions))

    for s in range(n_states):
        opt_act = pi_star[s]
        I_pi_star[s * n_actions:s * n_actions + n_actions, s * n_actions + opt_act] = 1
        # print(s*n_actions,":",s*n_actions+n_actions)
        # print(s*n_actions + opt_act)
        # input()
    return I_pi_star
# enddef