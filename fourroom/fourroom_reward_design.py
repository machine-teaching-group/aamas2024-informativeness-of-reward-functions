import sys
import numpy as np
import copy
import cvxpy as cp
import itertools as it
import utils

accumulator_for_file = {}

def convert_det_to_stochastic_policy(env, deterministicPolicy):
    # Given a deterministic Policy, I will return a stochastic policy
    stochasticPolicy = np.zeros((env.n_states, env.n_actions))
    if env.terminal_state == 1:
        n_states = env.n_states-1
    else:
        n_states = env.n_states
    for i in range(n_states):
        stochasticPolicy[i][deterministicPolicy[i]] = 1
    return stochasticPolicy
#enddef

def reward_design_model_based(env_0,  pi_d_T, pi_s_T, pi_d_L, pi_s_L, R_max, H_set=None, s_active=None,
                    delta_s_array=None, Adv_orig_pi_T=None, mu_pi_T_s=None,  tol=1e-6, teacher_name=None):
    if teacher_name == "Ada_TL":
        return reward_design_model_based_Ada_TL(env_0,  pi_d_T, pi_s_T, pi_d_L, pi_s_L, R_max,
                    delta_s_array, Adv_orig_pi_T, mu_pi_T_s,  tol)
    elif teacher_name == "EXPRD":
        return reward_design_model_based_EXPRD(env_0,  pi_d_T, pi_s_T, R_max, H_set, s_active=s_active,
                    delta_s_array=delta_s_array, tol=tol)

    elif teacher_name == "Invariance":
        return reward_design_model_based_invariant(env_0,  pi_d_T, pi_s_T, R_max,
                    delta_s_array, tol=tol)
    else:
        print("teachername is not corrent: {}".format(teacher_name))
        exit()
    #enddef


def reward_design_model_based_invariant(env_0,  pi_d_T, pi_s_T, R_max,
                    delta_s_array=None, tol=1e-10):

    n_states = env_0.n_states if env_0.terminal_state == 0 else env_0.n_states - 1 #chech if we have terminal state
    n_states_all = env_0.n_states

    n_actions = env_0.n_actions
    gamma = env_0.gamma
    #variable declaration
    # reward \in R^|S|*|A|
    w = cp.Variable(env_0.state_action_feature_matrix[1].shape)
    R = np.array(env_0.state_action_feature_matrix) @ w

    # set_of_pi_d = get_set_of_determinisric_policies(env_0, pi_d, dict_s_opt_actions_arr)

    ##### delta_s_array =====================
    delta_s_array = np.array(delta_s_array)

    constraints = []
    delta_s_eps_h_s_array_diff = []

    I_pi_star = utils.calculate_I_pi_star(env_0, pi_d_T)
    P_pi_star = utils.calculate_P_pi_star(env_0, pi_d_T)
    I = np.eye(n_states_all * n_actions)


    #global optimality
    A = (I_pi_star - I) @ np.linalg.inv((I - gamma * P_pi_star))
    b = np.zeros(n_states_all * n_actions)
    # set zero for optimal actions
    for s in range(n_states_all):
        for a in range(n_actions):
            if pi_s_T[s, a] == 0:
                b[s * n_actions + a] = delta_s_array[s]

    A_x = A @ R

    # global optimality constraints
    for i in range(A_x.shape[0]):
        cons = (A_x[i] >= b[i])
        constraints.append(cons)
    #  R_max bounds
    for s in range(n_states_all):
        for a in range(n_actions):
            cons_1 = (R[s * n_actions + a] >= -R_max)
            cons_2 = (R[s * n_actions + a] <= R_max)
            constraints.append(cons_1)
            constraints.append(cons_2)


    obj = cp.Minimize(1)
    prob = cp.Problem(obj, constraints)
    # Solve the problem
    prob.solve(solver=cp.ECOS, max_iters=1000, feastol=tol, abstol=tol, reltol=tol)
    # prob.solve()
    # prob.solve()
    obj_value = copy.deepcopy(prob.value)
    # get solution
    R_sol = R.value

    # round solution to the precision
    R_sol = np.round(R_sol, 8)

    ## zero out the elemets close to zero
    R_sol[np.where((-tol <= R_sol) & (R_sol <= tol))] = 0.0
    # get final solutions
    reward = R_sol.reshape(n_states_all, n_actions)
    # epsilon_inf = R_sol[n_states * n_actions]
    # epsilon_H_arr = R_sol[n_states * n_actions + 1:]
    return obj_value, reward
# enddef


def reward_design_model_based_EXPRD(env_0,  pi_d_T, pi_s_T, R_max, H_set, s_active =None,
                    delta_s_array=None, tol=1e-10):

    if s_active is None:
        s_active = env_0.goal_state# last or second last state as goal

    n_states = env_0.n_states if env_0.terminal_state == 0 else env_0.n_states - 1 #chech if we have terminal state
    n_states_all = env_0.n_states

    n_actions = env_0.n_actions
    gamma = env_0.gamma
    #variable declaration
    # reward \in R^|S|*|A|
    w = cp.Variable(env_0.state_action_feature_matrix[1].shape)
    R = np.array(env_0.state_action_feature_matrix) @ w

    # set_of_pi_d = get_set_of_determinisric_policies(env_0, pi_d, dict_s_opt_actions_arr)

    ##### delta_s_array =====================
    # delta_s_array = np.array(delta_s_array) / (1+1e-5)

    constraints = []
    delta_s_eps_h_s_array_diff = []

    I_pi_star = utils.calculate_I_pi_star(env_0, pi_d_T)
    P_pi_star = utils.calculate_P_pi_star(env_0, pi_d_T)
    I = np.eye(n_states_all * n_actions)


    #global optimality
    A = (I_pi_star - I) @ np.linalg.inv((I - gamma * P_pi_star))
    b = np.zeros(n_states_all * n_actions)
    # set zero for optimal actions
    for s in range(n_states_all):
        for a in range(n_actions):
            if pi_s_T[s, a] == 0:
                b[s * n_actions + a] = delta_s_array[s]

    A_x = A @ R

    # global optimality constraints
    for i in range(A_x.shape[0]):
        cons = (A_x[i] >= b[i])
        constraints.append(cons)

    # Q_H constraints for every H
    for i, H in enumerate(H_set):
        accumulator = get_A_local_h(env_0, P_pi_star, I, H=H)

        A_local = (I_pi_star - I) @ (accumulator)

        A_x_local = A_local @ R

        # calculate eps_h array
        for s in range(n_states_all):
            s_a_array = []
            for a in range(n_actions):
                if pi_s_T[s, a] == 0:
                    s_a_array.append(delta_s_array[s] - A_x_local[s * n_actions + a])
            if len(s_a_array) != 0:
                delta_s_eps_h_s_array_diff.append(cp.max(cp.hstack(s_a_array)))


    # converte back to cvxpy variable
    delta_s_eps_h_s_array_diff = cp.hstack(delta_s_eps_h_s_array_diff)

    #  R_max bounds
    for s in range(n_states_all):
        for a in range(n_actions):
            cons_1 = (R[s * n_actions + a] >= -R_max)
            cons_2 = (R[s * n_actions + a] <= R_max)
            constraints.append(cons_1)
            constraints.append(cons_2)

    # sparsity constraints
    for s in range(n_states_all):
        for a in range(n_actions):
            if s not in s_active:
                cons = (R[s * n_actions + a] == 0)
                constraints.append(cons)

    IR = - (1 / len(H_set)) * (1 / n_states_all) * \
           cp.sum(cp.pos(delta_s_eps_h_s_array_diff))


    obj = cp.Minimize(-IR)
    prob = cp.Problem(obj, constraints)
    # Solve the problem
    prob.solve(solver=cp.ECOS, max_iters=1000, feastol=tol, abstol=tol, reltol=tol)
    # prob.solve()
    # prob.solve()
    obj_value = copy.deepcopy(prob.value)
    # get solution
    R_sol = R.value

    # round solution to the precision
    R_sol = np.round(R_sol, 8)

    ## zero out the elemets close to zero
    R_sol[np.where((-tol <= R_sol) & (R_sol <= tol))] = 0.0
    # get final solutions
    reward = R_sol.reshape(n_states_all, n_actions)
    # epsilon_inf = R_sol[n_states * n_actions]
    # epsilon_H_arr = R_sol[n_states * n_actions + 1:]
    return obj_value, reward
# enddef

def reward_design_model_based_Ada_TL(env_0,  pi_d_T, pi_s_T, pi_d_L, pi_s_L, R_max,
                    delta_s_array=None, Adv_orig_pi_T=None, mu_pi_T_s=None,  tol=1e-6):

    n_states_all = env_0.n_states
    #variable declaration
    # reward \in R^|S|*|A|
    w = cp.Variable(env_0.state_action_feature_matrix[1].shape)
    R = np.array(env_0.state_action_feature_matrix) @ w

    ##### delta_s_array =====================
    delta_s_array = np.array(delta_s_array)

    constraints = []
    P_pi_star = utils.calculate_P_pi_star(env_0, pi_d_T)
    I_pi_star = utils.calculate_I_pi_star(env_0, pi_d_T)
    I = np.eye(env_0.n_states * env_0.n_actions)

    # global optimality

    # P_H = utils.get_P_H(env_0=env_0, P_pi_star=P_pi_star, H=env_0.H)

    # A_H = (I_pi_star - I) @ P_H

    A = (I_pi_star - I) @ np.linalg.inv((I - env_0.gamma * P_pi_star))


    #global optimality
    b = np.zeros(n_states_all * env_0.n_actions)
    # set zero for optimal actions
    for s in range(env_0.n_states):
        for a in range(env_0.n_actions):
            if pi_s_T[s, a] == 0:
                b[s * env_0.n_actions + a] = delta_s_array[s]

    # A_x_H_orig = P_H @ env_0.reward.flatten()
    # A_x_H = A_H @ R
    A_x = A @ R

    # H step global optimality constraints
    for i in range(A_x.shape[0]):
        cons = (A_x[i] >= b[i])
        constraints.append(cons)

    # calculate Advantage, w,r,t, original R
    # Adv_orig_pi_T = calculate_ADV_stoch(env_0, env_0.reward, pi_s_T, P_pi_star, I)
    Adv_orig_pi_T_L = np.repeat(np.sum((pi_s_L.flatten() * Adv_orig_pi_T.flatten()).reshape(env_0.n_states, env_0.n_actions),
                                     axis=1), env_0.n_actions)
    # _, mu_pi_T_s = calc_mu_s_a_stoch(env_0, pi_s_T)
    Adv_H_1 = R
    Adv_H_1_pi_L_s = cp.sum(cp.reshape(cp.multiply(pi_s_L.flatten(),
                                        Adv_H_1),(env_0.n_states, env_0.n_actions)),
                                     axis=1)
    Adv_H_1_pi_L_s_a = cp.hstack([item for item in Adv_H_1_pi_L_s for i in range(env_0.n_actions)])
    mu_pi_L_s_a, _ = calc_mu_s_a_stoch(env_0, pi_s_L)

    obj = cp.multiply(pi_s_L.flatten(), cp.multiply(Adv_orig_pi_T.flatten() - Adv_orig_pi_T_L,
                                                    Adv_H_1 - Adv_H_1_pi_L_s_a)) \
          @ cp.multiply(mu_pi_T_s.repeat(env_0.n_actions), mu_pi_L_s_a.flatten())

    #  R_max bounds
    for s in range(env_0.n_states):
        for a in range(env_0.n_actions):
            cons_1 = (R[s * env_0.n_actions + a] >= -R_max)
            cons_2 = (R[s * env_0.n_actions + a] <= R_max)
            constraints.append(cons_1)
            constraints.append(cons_2)

    obj = cp.Maximize(obj)
    prob = cp.Problem(obj, constraints)
    # Solve the problem
    prob.solve(solver=cp.ECOS, max_iters=10000, feastol=tol, abstol=tol, reltol=tol)
    obj_value = copy.deepcopy(prob.value)
    # get solution
    if R.value is not None:
        R_sol = R.value
        # round solution to the precision
        R_sol = np.round(R_sol, 8)
    else:
        print("Solution Cannot found in optimization problem")
        exit(0)
    ## zero out the elemets close to zero
    R_sol[np.where((-tol <= R_sol) & (R_sol <= tol))] = 0.0
    # get final solutions
    reward = R_sol.reshape(n_states_all, env_0.n_actions)
    # epsilon_inf = R_sol[n_states * n_actions]
    # epsilon_H_arr = R_sol[n_states * n_actions + 1:]
    return obj_value, reward
# enddef




def get_A_local_h(env_0, P_pi_star, I, H):
    #local (H horizon) optimality constraints
    accumulator = copy.deepcopy(I)
    accumulator_P_star_mult = copy.deepcopy(I)
    gamma = env_0.gamma

    for h in range(1, H):
        accumulator_P_star_mult = accumulator_P_star_mult @ P_pi_star
        accumulator += (gamma**h) * accumulator_P_star_mult

    return accumulator
#enddef

def get_set_of_determinisric_policies(env_0, pi_d, dict_s_opt_actions):
    n_states = env_0.n_states
    opt_action_set = []

    for s in range(n_states):
        if s in dict_s_opt_actions:
            opt_action_set.append(dict_s_opt_actions[s])
        else:
            opt_action_set.append([pi_d[s]])
    set_det_policies_tuple = list(it.product(*opt_action_set))

    # convert to array
    set_det_policies = []
    for pi_d in set_det_policies_tuple:
        set_det_policies.append(np.array(pi_d))
    return set_det_policies


# enddef

def get_mat_for_concatenation(env_0, pi_t, j, H_set, delta_s_array):
    n_states = env_0.n_states
    n_actions = env_0.n_actions
    mat_for_concat_H = np.zeros((n_states*n_actions, n_states*n_actions))
    #set zero for optimal actions
    for s in range(n_states):
        for a in range(n_actions):
            if a != pi_t[s]:
                mat_for_concat_H[s * n_actions + a, s * n_actions + a] = - delta_s_array[s] / 2
    return mat_for_concat_H
#enddef


def calculate_P_pi_star_stoch(env, pi_star):
    n_states = env.n_states
    n_actions = env.n_actions
    P_0 = env.T

    P_pi_star = np.zeros((n_states * n_actions, n_states * n_actions))

    for s in range(n_states):
        for a in range(n_actions):
            for s_n in range(n_states):
                for a_n in range(n_actions):
                    P_pi_star[s * n_actions + a, s_n * n_actions + a_n] = \
                        P_0[s, s_n, a] * pi_star[s_n,a_n]
    return P_pi_star
# enddef

def calc_mu(env, pi, InitD):
    gamma = env.gamma
    P = env.T
    states_count = env.n_states
    T = np.zeros((states_count, states_count))
    for s1 in range(states_count):
        for s2 in range(states_count):
            T[s1, s2] = P[s1, s2, pi[s1]]

    A = np.transpose(np.identity(states_count) - gamma * T)
    b = (1 - gamma) * InitD

    mu = np.linalg.solve(A, b)
    return mu
#enddef

def calc_mu_stoch(env, pi_stoch, InitD):
    gamma = env.gamma
    states_count = env.n_states
    T = get_T_pi_stoch(env, pi_stoch)

    A = np.transpose(np.identity(states_count) - gamma * T)
    b = (1 - gamma) * InitD

    mu = np.linalg.solve(A, b)
    return mu
#enddef

def calc_mu_s_a_stoch(env, policy):
    mu_s_a = np.zeros((env.n_states, env.n_actions))
    if len(policy.shape) == 1:
        changed_policy = convert_det_to_stochastic_policy(env, policy)
    else:
        changed_policy = policy

    P_pi = get_T_pi_stoch(env, changed_policy)

    A = np.transpose(np.identity(env.n_states) - env.gamma * P_pi)
    b = (1 - env.gamma) * env.InitD
    mu_s = np.linalg.solve(A, b)

    mu_s[-1] = 0.0
    mu_s = mu_s / sum(mu_s)

    for a in range(env.n_actions):
        mu_s_a[:, a] = policy[:, a] * mu_s[:]
    return mu_s_a, mu_s

#enddef

def get_T_pi_stoch(env, policy):
    T_pi = np.zeros((env.n_states, env.n_states))
    for n_s in range(env.n_states):
        for a in range(env.n_actions):
            T_pi[:, n_s] += policy[:, a] * env.T[:, n_s, a]

    return T_pi
# enddef

def calculate_ADV_H(env, reward, pi, P_pi_star, I, H):
    n_actions = env.n_actions
    # local (H horizon) optimality constraints
    accumulator = copy.deepcopy(I)
    accumulator_P_star_mult = copy.deepcopy(I)
    gamma = env.gamma

    for h in range(0, H + 1):
        accumulator_P_star_mult = accumulator_P_star_mult @ P_pi_star
        accumulator += (gamma ** h) * accumulator_P_star_mult

    Q_H = accumulator @ reward.flatten()

    V_H_pi_array = []

    for s in range(env.n_states):
        action = pi[s]
        v_s = Q_H[s * n_actions + action]
        V_H_pi_array.append(v_s)

    V_H = [item for item in V_H_pi_array for i in range(n_actions)]
    Adv_H = (Q_H - cp.hstack(V_H))
    return Adv_H
#enddef

def calculate_ADV(env, reward, pi, P_pi_star, I):
    #calculate Q
    Q_orig_linear = np.linalg.inv((I - env.gamma * P_pi_star)) @ reward.flatten()

    # V_orig_linear = calc_V_values(env_0, env_0.reward.flatten(), opt_pi_d_t)
    V_linear = np.take_along_axis(Q_orig_linear.reshape(env.n_states, env.n_actions),
                                  pi[:, None], axis=1).flatten()

    # compute Advantage
    Advantage = (Q_orig_linear - np.repeat(V_linear, env.n_actions))
    return Advantage
#enddef

def calculate_ADV_stoch(env, reward, pi, P_pi_star_toch, I):
    #calculate Q
    Q_orig_linear = np.linalg.inv((I - env.gamma * P_pi_star_toch)) @ reward.flatten()


    # V_orig_linear = calc_V_values(env_0, env_0.reward.flatten(), opt_pi_d_t)
    V_linear = np.sum(np.reshape(np.multiply(pi.flatten(),
                                        Q_orig_linear),(env.n_states, env.n_actions)),
                                     axis=1)

    # compute Advantage
    Advantage = (Q_orig_linear - np.repeat(V_linear, env.n_actions))
    return Advantage
#enddef

########################################
if __name__ == "__main__":
    pass
