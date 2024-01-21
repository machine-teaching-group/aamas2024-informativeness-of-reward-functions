import numpy as np

class parameters:
    def __init__(self):

        # Teacher's args 4room
        self.teacher_args_linekey = {"eta_phi_Ada_TL": 0.01,
                                   "eta_phi_Ada_LL": 0.01,
                                   "eta_phi_Non_Ada_TPopL": 0.01,
                                    "K_update_phi_Ada_TL": 100,
                                    "K_update_phi_Ada_LL": 1,
                                    "K_update_Non_Ada_TPopL": 1000,
                                   "eta_critic": 0.01,
                                   "clipping_epsilon": 1.0,
                                   "use_clipping": True,
                                   "path_for_10_percent_sub_opt_policy": "sub_opt_policies/10_precent/agent=reinforce/room_n2_subgoal=0.0/exp_iter=1/pi_sub_optimal_10_percent.txt",
                                   "path_for_25_percent_sub_opt_policy": "sub_opt_policies/25_precent/agent=reinforce/room_n2_subgoal=0.0/exp_iter=1/pi_sub_optimal_25_percent.txt",
                                   "path_for_50_percent_sub_opt_policy": "sub_opt_policies/50_precent/agent=reinforce/room_n2_subgoal=0.0/exp_iter=1/pi_sub_optimal_50_percent.txt",
                                   "path_for_reinforce_policy": "reinforce_policy/policy.txt"
                               }

        # Learner's args
        self.agent_args = {"eta_actor": 0.01,
                            "Q_epsilon": 0.05,
                             "Q_alpha": 0.1,
                           "learner_population": "learner_population/"
                          }

        # Teaching's args
        self.teaching_args = {
            "N_reinforce": 50010,
            "N_Q_learning": 50001,
            "N_r": 5,
            "N_p": 2,
            "buffer_size": 10,
            "buffer_size_recent": 5,
            "agent_evaluation_step": 100
        }
    #endde
#endcalss