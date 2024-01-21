import numpy as np
import env_linekey_discrete
import agent_reinforce_tabular as agent_reinforce
# import agent_Q_learning_tabular as agent_Q_learning
import env_generic_with_teacher as Teacher_env
import utils

from collections import deque as collections_deque
import parameters
import argparse

import plottingscript
import os


class teaching():
    # hyperparameters
    hyper_parameters = parameters.parameters()
    teacher_args = hyper_parameters.teacher_args_linekey
    agent_args = hyper_parameters.agent_args
    teaching_args = hyper_parameters.teaching_args

    def __init__(self, env_orig, teacher_name="", agent_type="", exp_iter=1,
                 use_pool=None, epsilon_greedy=None):
        self.env_orig = env_orig
        self.teacher_name = teacher_name
        self.agent_type = agent_type
        self.env_teacher = None
        self.exp_iter = exp_iter
        self.use_pool = use_pool
        self.epsilon_greedy = epsilon_greedy

        if agent_type == "reinforce":
            ## use no clipping for reinforce
            # self.teacher_args["use_clipping"] = True
            self.env_teacher = Teacher_env.EnvTeacher(env_orig, self.teacher_args, teacher_name, epsilon_greedy)
            ## define agent
            self.agent = agent_reinforce.Agent(self.env_teacher, self.agent_args, exp_iter, use_pool=use_pool)
            self.teaching_args["N"] = self.teaching_args["N_reinforce"]

        else:
            print(" Wrong agent type")
            exit(0)
        self.accumulator = {}
    #enddef

    def end_to_end_training(self):


        N = self.teaching_args["N"]
        N_r = self.teaching_args["N_r"]
        N_p = self.teaching_args["N_p"]
        buffer_size = self.teaching_args["buffer_size"]
        buffer_size_recent = self.teaching_args["buffer_size_recent"]
        agent_evaluation_step = self.teaching_args["agent_evaluation_step"]

        D = collections_deque(maxlen=buffer_size)

        expected_reward_array_env_orig = []
        expected_reward_arrray_env_teacher = []
        IR_array = []

        for i in range(N):

            if i % 100 == 0:
                R_designed = self.env_teacher.get_R_phi()
                self.save_designed_rewards_into_file(R_designed, agent_type,
                                        teacher_name, "linekey",
                                        exp_iter=self.exp_iter, step=i, gamma=self.env_orig.gamma,
                                                     H=self.env_orig.H, usepool=self.use_pool)

            if i % agent_evaluation_step == 0:
                # evaluate learner's current policy on orig environment
                expected_reward_G_bar, expected_reward_G_hat = self.evaluate_agent(self.env_orig, self.env_teacher, self.agent,
                                                                                    n_episode=5)
                expected_reward_array_env_orig.append(expected_reward_G_bar)
                expected_reward_arrray_env_teacher.append(expected_reward_G_hat)
                print("=====================")
                print("===============Iter = {}/{}========================".format(i, N))
                print("Teacher = {}".format(self.teacher_name))
                print("Exp reward = {}".format(np.round(expected_reward_array_env_orig[-1], 4)))
                print("Goal visitation count = {}".format(self.env_teacher.goal_visits))
                print("===================================================")
            # rollout a trajectory --> [state, action, r, next_state, G_r,  \pi(.|state)] r = \hat(r) on teacher's env
            episode = utils.generate_sampled_data(self.env_teacher, self.agent)  # --> rewrite

            # add to buffer
            D.append(episode)

            # teacher update
            if (i + 1) % N_r == 0:

                self.env_teacher.update(D,  self.agent)

            # learner's update
            if (i + 1) % N_p == 0:

                # print("=== Learner update ===")

                self.agent.update(list(D)[-buffer_size_recent:])

        self.accumulator["expected_reward_teacher={}".format(self.env_teacher.teacher_name)] = \
            np.array(expected_reward_array_env_orig)
        self.accumulator["IR_teacher={}".format(self.env_teacher.teacher_name)] = \
            np.array(IR_array)

        return self.accumulator
    #enddef

    def save_policy_into_file(self, policy, agent_type,
                                        teacher_name, env_name,
                                        exp_iter, step):
        import os
        policy = np.asarray(policy)
        directory = "results/policies/agent={}/{}_n2_subgoal={}/teacher={}/exp_iter={}/"\
            .format(agent_type, env_name, self.env_orig.n2_subgoal, teacher_name, exp_iter)
        file_path = directory + "policy_step={}.txt".format(step)
        if not os.path.isdir(directory):
            os.makedirs(directory)

        with open(file_path, 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(policy.shape))

            np.savetxt(outfile, policy, fmt='%-7.9f')
        # enddef

    def save_designed_rewards_into_file(self, reward, agent_type,
                                        teacher_name, env_name,
                                        exp_iter, step, gamma=None, H=None, usepool=None):
        import os
        reward = np.asarray(reward)
        directory = "results/pool_learner_UsePool={}_gamma={}_H={}/designed_rewards/agent={}/teacher={}/exp_iter={}/"\
            .format( usepool, gamma, H, agent_type, teacher_name, exp_iter)
        file_path = directory + "R_designed_step={}.txt".format(step)
        if not os.path.isdir(directory):
            os.makedirs(directory)

        with open(file_path, 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(reward.shape))

            np.savetxt(outfile, reward, fmt='%-7.9f')
        # enddef

    def evaluate_agent(self, env_orig, env_teacher, agent, n_episode):
        episode_reward_array_env_orig = []
        episode_reward_array_env_teacher = []

        for i in range(n_episode):
            episode_reward_env_orig = 0
            episode_reward_env_teacher = 0
            episode = utils.generate_sampled_data(env_teacher, agent)
            postProcessed_episode = self.env_teacher.get_postposessed_episode(env_orig, episode)

            for t in range(len(postProcessed_episode)):
                _, _, r_hat, _, _, _, r_bar, _ = postProcessed_episode[t]

                episode_reward_env_orig += env_orig.gamma**t * r_bar
                episode_reward_env_teacher += env_teacher.gamma ** t * r_hat

            episode_reward_array_env_orig.append(episode_reward_env_orig)
            episode_reward_array_env_teacher.append(episode_reward_env_teacher)

        return np.average(episode_reward_array_env_orig), np.average(episode_reward_array_env_teacher)
    #enddef

#endclass

def calculate_average(dict_accumulator, number_of_iterations):
    for key in dict_accumulator:
        dict_accumulator[key] = dict_accumulator[key]/number_of_iterations
    return dict_accumulator
#enddef
def accumulator_function(tmp_dict, dict_accumulator):
    for key in tmp_dict:
        if key in dict_accumulator:
            dict_accumulator[key] += np.array(tmp_dict[key])
        else:
            dict_accumulator[key] = np.array(tmp_dict[key])
    return dict_accumulator
#enddef



if __name__ == "__main__":

    final_dict_accumulator = {}

    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--agent', default='reinforce', type=str, help='agent type')
    parser.add_argument('--teacher', default='Orig', type=str, help='teacher type')
    parser.add_argument('--n_averaged', default=2, type=int, help='N runs to average')
    parser.add_argument('--use_pool', default="False", type=str, help='Flag different learners')
    parser.add_argument('--epsilon_greedy', default=0.3, type=float, help='epsilon_greedy for calculating mu')


    args = parser.parse_args()
    agent_type = args.agent
    teacher = args.teacher
    use_pool = eval(args.use_pool)
    n_averaged = args.n_averaged

    epsilon_greedy = args.epsilon_greedy

    # teachers = [teacher]
    teachers = ["EXPRD", "Invariance", "Orig", "Ada_TL"]
    learner_population ="learner_population/"

    R_max = 10
    gamma = 0.95
    H = 30

    ### file names
    out_folder_name_plots_convergence = "plots/convergence/"
    out_folder_name_plots_designed_rewards = "plots/visualization/exp_number={}/"
    result_directory_convergence = "pool_learner_UsePool={}_gamma={}_H={}/convergence/agent={}/".format(use_pool, gamma, H, args.agent)
    result_directory_designed_rewards = "pool_learner_UsePool={}_gamma={}_H={}/designed_rewards/agent={}/".format(use_pool, gamma, H, args.agent)


    for i in range(0, args.n_averaged):

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


        env_orig = env_linekey_discrete.Environment(env_args)

        dict_accumulator = {}
        for teacher_name in teachers:

            teaching_obj = teaching(env_orig, teacher_name,
                                    agent_type, exp_iter=(i+1),
                                    use_pool=use_pool,
                                    epsilon_greedy=epsilon_greedy)

            dict_acc = teaching_obj.end_to_end_training()
            dict_accumulator.update(dict_acc)

            output_directory = result_directory_convergence + "teacher={}".format(teacher_name)

            utils.write_into_file(accumulator=dict_accumulator, exp_iter=i + 1,
                              out_folder_name=output_directory)

    ### When experiment finishes plot the results

    #(1) plots for reward visualizations
    plottingscript.plot_rewards(teachers, use_pool, result_directory_designed_rewards,
                                out_folder_name_plots_designed_rewards, n_runs=n_averaged)

    #(2) plots for convergence
    plottingscript.plot_convergence(teachers,
                                    use_pool,
                                    result_directory_convergence,
                                    out_folder_name_plots_convergence, n_runs=n_averaged)
    #delete results file
    os.system("rm -rf results/")