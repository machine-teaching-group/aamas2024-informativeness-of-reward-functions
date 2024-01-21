import numpy as np
import os
from matplotlib import pyplot as plt


def plot_reward_action(reward, output_folder, out_file_name, color_ranges):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, sharex=True)
    # fig.suptitle("Without key")
    plt.rcParams["figure.figsize"] = [3.3, 3.8]
    #UP action

    ax1[0].imshow(np.flip(reward[:, 0][:-1].reshape(7, 7), axis=0),
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)

    UP_action_rewarrd_to_plot = np.flip(reward[:, 0][:-1].reshape(7, 7), axis=0)
    # text portion
    ind_array = np.arange(0, 7, 1)
    x, y = np.meshgrid(ind_array, ind_array)
    # x = np.rot90(x)
    # y = np.rot90(y)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if UP_action_rewarrd_to_plot[int(x_val), int(y_val)] < 0:
            c=r"$-$"
        elif UP_action_rewarrd_to_plot[int(x_val), int(y_val)] > 0:
            c=r"$+$"
        else:
            c=""
        ax1[0].text(y_val, x_val, c, va='center', ha='center', fontsize=12)


    ax1[0].set_yticks([])
    ax1[0].set_xlabel("up", fontsize=15)
    #########################################
    #########################################
    #########################################

    #LEFT action

    # fig, (ax) = plt.subplots(nrows=1, sharex=True)
    ax1[1].imshow(np.flip(reward[:, 1][:-1].reshape(7, 7), axis=0),
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    LEFT_action_rewarrd_to_plot = np.flip(reward[:, 1][:-1].reshape(7, 7), axis=0)

    # text portion
    ind_array = np.arange(0, 7, 1)
    x, y = np.meshgrid(ind_array, ind_array)
    # x = np.rot90(x)
    # y = np.rot90(y)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if LEFT_action_rewarrd_to_plot[int(x_val), int(y_val)] < 0:
            c = r"$-$"
        elif LEFT_action_rewarrd_to_plot[int(x_val), int(y_val)] >0:
            c = r"$+$"
        else:
            c = ""
        ax1[1].text(y_val, x_val, c, va='center', ha='center', fontsize=12)

    ax1[1].set_yticks([])
    ax1[1].set_xticks([])
    ax1[1].set_xlabel("left", fontsize=15)

    #########################################
    #########################################
    #########################################

    #DOWN action
    ax2[0].imshow(np.flip(reward[:, 2][:-1].reshape(7, 7), axis=0),
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    DOWN_action_rewarrd_to_plot = np.flip(reward[:, 2][:-1].reshape(7, 7), axis=0)

    ind_array = np.arange(0, 7, 1)
    x, y = np.meshgrid(ind_array, ind_array)
    # x = np.rot90(x)
    # y = np.rot90(y)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if DOWN_action_rewarrd_to_plot[int(x_val), int(y_val)] < 0:
            c = r"$-$"
        elif DOWN_action_rewarrd_to_plot[int(x_val), int(y_val)] > 0:
            c = r"$+$"
        else:
            c = ""
        ax2[0].text(y_val, x_val, c, va='center', ha='center', fontsize=12)
    ax2[0].set_yticks([])
    ax2[0].set_xlabel("down", fontsize=15)


    #########################################
    #########################################
    #########################################

    # RIGHT action

    ax2[1].imshow(np.flip(reward[:, 3][:-1].reshape(7, 7), axis=0),
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    RIGHT_action_rewarrd_to_plot = np.flip(reward[:, 3][:-1].reshape(7, 7), axis=0)
    ind_array = np.arange(0, 7, 1)
    x, y = np.meshgrid(ind_array, ind_array)
    # x = np.rot90(x)
    # y = np.rot90(y)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if RIGHT_action_rewarrd_to_plot[int(x_val), int(y_val)] <0:
            c = r"$-$"
        elif RIGHT_action_rewarrd_to_plot[int(x_val), int(y_val)] >0:
            c = r"$+$"
        else:
            c = ""
        ax2[1].text(y_val, x_val, c, va='center', ha='center', fontsize=12)
    ax2[1].set_yticks([])
    ax2[1].set_xlabel("right", fontsize=15)

    #########################################
    #########################################
    #########################################

    # plt.xticks([0, 6], ["1","7"], fontsize=13)
    # plt.show()
    plt.tight_layout()
    # plt.ylabel(, fontsize=16)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    plt.savefig(output_folder+out_file_name, bbox_inches='tight')
    plt.close()
    pass

def plot_rewards(teachers, Use_pool, result_directory, output_directory, n_runs):
    steps_to_plot = [1000, 2000, 3000, 100000, 200000]
    # steps_to_plot = [0, 100, 200]

    for teacher in teachers:
        input_directory = "results/{}/teacher={}/".format(result_directory, teacher)

        for run in range(1, n_runs+1):
            input_directory_for_specific_iter = "{}/exp_iter={}/".format(input_directory, run)
            file_name = "R_designed_step={}.txt"
            for step in steps_to_plot:
                if teacher == "Ada_TL":
                    teacher_name = "ExpAdaRD"
                else:
                    teacher_name = teacher
                out_file_name = "teacher={}_Use_pool={}_R_designed_expNumber={}_iteration={}.pdf".format(teacher_name, Use_pool, run,step)
                file_to_plot = input_directory_for_specific_iter + file_name.format(step)
                R_shaped = np.loadtxt(file_to_plot)

                plot_reward_action(R_shaped, output_directory.format(run),
                                   out_file_name,
                                   color_ranges=10)
    pass

def plot_convergence(teachers, Use_pool, result_directory, output_directory, n_runs):
    dict = input_from_different_files(result_directory, teachers, n_file=n_runs, t=200001)
    plot_convergence_given_dict(dict, teachers, out_folder_name=output_directory,
                                each_number=100, Use_pool=Use_pool)
    pass

def input_from_different_files(base_dir, teachers, n_file=1, t=0.0):
    dict_file = {}
    std_matrix_Orig = []
    std_matrix_Ada_TL = []
    std_matrix_EXPRD = []
    std_matrix_Invariance = []



    for teacher in teachers:
        file_path = "results/{}/teacher={}/".format(base_dir, teacher)
        # directory_to_input = input_dir_path_q_learning_convergence_numbers
        expname = '/out_file_'
        list_files = range(1, n_file+1)
        n_file = len(list_files)
        for file in list_files:
            with open(file_path + expname + str(file) + '.txt') as f:
                # with open(file_name) as f:
                print(file_path + expname + str(file) + '.txt')
                for line in f:
                    read_line = line.split()
                    if read_line[0] != '#':
                        if read_line[0] == 'initStates' or read_line[0] == 'active_state_opt_states' or read_line[
                            0] == 'sgd_state_opt_states' or read_line == 'template_list_for_initStates':
                            dict_file[read_line[0]] = np.array(list(map(int, read_line[1:t])))
                        elif read_line[0] in dict_file.keys():
                            dict_file[read_line[0]] += np.array(list(map(float, read_line[1:t])))
                        else:
                            dict_file[read_line[0]] = np.array(list(map(float, read_line[1:t])))
                        if read_line[0] == "expected_reward_teacher=Orig":
                            std_matrix_Orig.append(np.array(list(map(float, read_line[1:t]))))

                        if read_line[0] == "expected_reward_teacher=Ada_TL":
                            std_matrix_Ada_TL.append(np.array(list(map(float, read_line[1:t]))))

                        if read_line[0] == "expected_reward_teacher=EXPRD":
                            std_matrix_EXPRD.append(np.array(list(map(float, read_line[1:t]))))

                        if read_line[0] == "expected_reward_teacher=Invariance":
                            std_matrix_Invariance.append(np.array(list(map(float, read_line[1:t]))))

    for key, value in dict_file.items():
        dict_file[key] = value / (n_file)

    if len(std_matrix_Orig) >=1:
        dict_file["SE_Orig"] = np.std(std_matrix_Orig, ddof=1,
                                        axis=0) / np.sqrt(len(std_matrix_Orig))
    if len(std_matrix_Ada_TL) >= 1:
        dict_file["SE_Ada_TL"] = np.std(std_matrix_Ada_TL, ddof=1,
                                      axis=0) / np.sqrt(len(std_matrix_Ada_TL))
    if len(std_matrix_EXPRD) >= 1:
        dict_file["SE_EXPRD"] = np.std(std_matrix_EXPRD, ddof=1,
                                          axis=0) / np.sqrt(len(std_matrix_EXPRD))
    if len(std_matrix_Invariance) >= 1:
        dict_file["SE_Invariance"] = np.std(std_matrix_Invariance, ddof=1,
                                      axis=0) / np.sqrt(len(std_matrix_Invariance))

    return dict_file

def plot_convergence_given_dict(dict_file_q, teachers,
                                out_folder_name="",
                                each_number=0,
                                Use_pool=False):
    plt.switch_backend('agg')
    import matplotlib as mpl

    mpl.rcParams['font.serif'] = ['times new roman']
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']

    mpl.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 34})
    mpl.rc('legend', **{'fontsize': 30})
    mpl.rc('text', usetex=True)
    fig_size = [7, 4.8]

    if not os.path.isdir(out_folder_name):
        os.makedirs(out_folder_name)

    plt.figure(3, figsize=fig_size)

    keys = ["expected_reward_teacher={}".format(a) for a in teachers]

    for key in keys:
        if key == "expected_reward_teacher=Ada_TL":
            # picked_states = dict_file[]
            plt.errorbar(range(0, len(dict_file_q[key][::each_number])),
                         dict_file_q[key][::each_number],
                         label=r"$\textsc{Ada-}\text{TL}$",
                         color='g', marker=".", lw=4, markersize=18,
                         yerr=dict_file_q["SE_Ada_TL"][::each_number])



        if key == "expected_reward_teacher=Invariance":
            # picked_states = dict_file[]
            plt.errorbar(range(0, len(dict_file_q[key][::each_number])),
                         dict_file_q[key][::each_number],
                         label=r"$\textsc{Invar}$",
                         color='#FF8849', marker="<", ls=":", lw=4, markersize=10,
                         yerr=dict_file_q["SE_Invariance"][::each_number])

        elif key == "expected_reward_teacher=EXPRD":
            plt.errorbar(range(0, len(dict_file_q[key][::each_number])),
                         dict_file_q[key][::each_number],
                         label=r"$\textsc{ExpRD}$",
                         color='#0a81ab', marker="s", ls="-.", markersize=10, lw=4,
                         yerr=dict_file_q["SE_EXPRD"][::each_number])

        elif key == "expected_reward_teacher=Orig":
            plt.errorbar(range(0, len(dict_file_q[key][::each_number])),
                         dict_file_q[key][::each_number],
                         label=r"$\textsc{Orig}$ ",
                         color="#8b0000", marker="^", ls="-.", markersize=10, lw=2.5,
                         yerr=dict_file_q["SE_Orig"][::each_number])

    plt.legend(loc='upper center', bbox_to_anchor= (0.5, 1.4),
               ncol=2, borderaxespad=0)
    plt.ylabel(r"Expected reward")
    plt.xlabel(r'Episode (x$10^{4}$)')
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
               ["0", "", "", "", "", "5", "", "", "", "", "10", "", "", "", "", "15", "", "", "", "", "20"])
    plt.yticks([0, 2, 4, 6])

    outFname = os.getcwd() + "/" + out_folder_name + "/convergence_room_Use_pool={}.pdf".format(Use_pool)
    plt.savefig(outFname, bbox_inches='tight')
    plt.close()

    pass


if __name__ == "__main__":

  pass