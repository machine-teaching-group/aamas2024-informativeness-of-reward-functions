import numpy as np
import argparse
import env_linekey_discrete
import os


def save_theta_into_file(theta, directory,  exp_iter):
    import os
    theta = np.asarray(theta)
    file_path = directory + "policy_{}.txt".format(exp_iter)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(file_path, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(theta.shape))

        np.savetxt(outfile, theta, fmt='%-7.9f')
    # enddef


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--n_agents', default=50, type=int, help='# of agents')
    args = parser.parse_args()
    n_agents = args.n_agents

    directory = "learner_population/"

    if os.path.isdir(directory):
        print("Directory <<{}>> Exists!!".format(directory))
        exit()

    R_max = 10
    gamma = 0.95
    H = 30
    env_args = {
        "R_max": R_max,
        "chain_len": 10,
        "n_actions:": 3,
        "gamma": gamma,
        "H": H,
        "randomMoveProb": 0.1,
        "terminal_state": 1,
        "n2_subgoal": 0.0,
    }

    env = env_linekey_discrete.Environment(env_args)

    # for i in range(1, n_agents+1):
    #     actor_theta_array = np.random.uniform(low=-1, high=1,
    #                                           size=(env.n_states,
    #                                                 env.n_actions))
    #     save_theta_into_file(actor_theta_array, directory, i)

    key_location = 1

    actor_theta_array_1 = np.zeros((env.n_states, env.n_actions))

    actor_theta_array_1[key_location] = np.array([2, 0., 0.])

    save_theta_into_file(actor_theta_array_1, directory, 0)

    #========================

    actor_theta_array_2 = np.zeros((env.n_states, env.n_actions))
    actor_theta_array_2[key_location] = np.array([0., 2,  0.])
    save_theta_into_file(actor_theta_array_2, directory, 1)

    # ========================

    actor_theta_array_3 = np.zeros((env.n_states, env.n_actions))
    actor_theta_array_3[env.chain_len//2] = np.array([0, 2, 0.])

    save_theta_into_file(actor_theta_array_3, directory, 2)

    # ========================

    actor_theta_array_4 = np.zeros((env.n_states, env.n_actions))
    actor_theta_array_4[env.chain_len//2] = np.array([0, 0., 2])
    save_theta_into_file(actor_theta_array_4, directory, 3)

    # ========================

    actor_theta_array_5 = np.zeros((env.n_states, env.n_actions))
    actor_theta_array_5[env.chain_len] = np.array([0., 0., 2])

    save_theta_into_file(actor_theta_array_5, directory, 4)
