import numpy as np
import os
from matplotlib import pyplot as plt

import pygraphviz as pgv
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def plot_reward_action(reward_orig_abstracted_reward, output_folder, out_file_name, color_ranges):
    plt.rcParams["figure.figsize"] = [3.5, 4]
    fig, (ax1,ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6, sharex=True)
    # fig.suptitle("Without key")

    ax1.imshow([reward_orig_abstracted_reward[:, 0][:10]],
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    ax1.set_yticks([])
    ax1.set_ylabel(r"$\textbf{l-}$", fontsize=16)
    # ax1.ylabel('ylabel', fontsize=16)

    # ax1.set_title('Action Left')

    # fig, (ax) = plt.subplots(nrows=1, sharex=True)
    ax2.imshow([reward_orig_abstracted_reward[:, 1][:10]],
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    ax2.set_yticks([])
    ax2.set_ylabel(r"$\textbf{r-}$", fontsize=16)
    # ax2.ylabel('ylabel', fontsize=16)
    # ax2.set_title('Action Right')


    # fig, (ax) = plt.subplots(nrows=1, sharex=True)
    ax3.imshow([reward_orig_abstracted_reward[:, 2][:10]],
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    ax3.set_yticks([])
    # ax3.set_xlim(0, 9)
    ax3.set_ylabel(r"$\textbf{p-}$", fontsize=16)
    # ax3.ylabel('ylabel', fontsize=16)
    # ax3.set_title('Action Pick')

    # plt.tight_layout()
    # plt.savefig("fig_without_key.pdf", bbox_inches='tight')


    # fig, (ax1,ax2, ax3) = plt.subplots(nrows=6, sharex=True)


    # fig, (ax) = plt.subplots(nrows=1, sharex=True)
    ax4.imshow([reward_orig_abstracted_reward[:, 0][10:20]], cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    ax4.set_yticks([])
    # ax5.set_title('Action Right')
    ax4.set_ylabel(r"$\textbf{lK}$", fontsize=16)
    # ax5.ylabel('ylabel', fontsize=16)

    ax5.imshow([reward_orig_abstracted_reward[:, 1][10:20]],
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    ax5.set_yticks([])
    # ax4.set_title('Action Left')
    ax5.set_ylabel(r"$\textbf{rK}$", fontsize=16)
    # ax4.ylabel('ylabel', fontsize=16)


    # fig, (ax) = plt.subplots(nrows=1, sharex=True)
    # print(reward_orig_abstracted_reward[:, 2][10:20])
    # input()
    ax6.imshow([reward_orig_abstracted_reward[:, 2][10:20]],
               cmap="RdBu", aspect="auto", vmin=-color_ranges, vmax=color_ranges)
    ax6.set_yticks([])
    ax6.set_xlim(-0.5, 9.5)
    ax6.set_ylabel(r"$\textbf{pK}$", fontsize=16)
    # ax6.set_title('Action Pick')
    # ax2.set_xticks()
    plt.xticks(
               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], fontsize=16)
    # plt.tight_layout()
    plt.tight_layout()
    # plt.ylabel(, fontsize=16)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    plt.savefig(output_folder+out_file_name, bbox_inches='tight')
    plt.close()
    pass

def plot_rewards(teachers, Use_pool, result_directory, output_directory, n_runs):

    import matplotlib as mpl
    mpl.rcParams['font.serif'] = ['times new roman']
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
    mpl.rc('font', **{'family': 'serif', 'serif': ['Times']})
    # mpl.rc('legend', **{'fontsize': 13})
    mpl.rc('text', usetex=True)


    steps_to_plot = [100, 30000, 50000]
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
                out_file_name_tree = "teacher={}_Use_pool={}_R_designed_expNumber={}_iteration={}_tree.pdf".format(teacher_name, Use_pool, run,step)
                file_to_plot = input_directory_for_specific_iter + file_name.format(step)
                R_shaped = np.loadtxt(file_to_plot)

                plot_reward_action(R_shaped, output_directory.format(run),
                                   out_file_name,
                                   color_ranges=10)
                tree, node_labels, leaf_action_node_to_state = create_tree()
                plot_reward_tree(R_shaped, output_directory.format(run),
                                 out_file_name=out_file_name_tree,
                                 color_ranges=10,
                                 tree=tree,
                                 node_labels=node_labels,
                                 leaf_action_node_to_state=leaf_action_node_to_state)

    pass

### tree
def rgba_to_hex(rgba):
    r, g, b, a = rgba
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))


def create_tree():
    fontsize_of_leaf_label = 40
    fontsize_of_leaf_Yes_NO_label = 32

    tree = pgv.AGraph(strict=False, directed=True)

    tree.add_edge(1, 2, xlabel="No", fontsize=fontsize_of_leaf_Yes_NO_label)
    tree.add_edge(1, 3, label='Yes', fontsize=fontsize_of_leaf_Yes_NO_label, labelloc='b')
    tree.add_edge(2, 4, xlabel='No', fontsize=fontsize_of_leaf_Yes_NO_label)
    tree.add_edge(4, 8, xlabel='No', fontsize=fontsize_of_leaf_Yes_NO_label)
    tree.add_edge(4, 9, label='Yes', fontsize=fontsize_of_leaf_Yes_NO_label)
    tree.add_edge(2, 5, label='Yes', fontsize=fontsize_of_leaf_Yes_NO_label)
    tree.add_edge(3, 6, xlabel='No', fontsize=fontsize_of_leaf_Yes_NO_label)
    tree.add_edge(3, 7, label='Yes', fontsize=fontsize_of_leaf_Yes_NO_label)

    # Leaf 0
    tree.add_edge(8, 10, label=r'<<b>l</b>>', fontsize=fontsize_of_leaf_label)
    tree.add_edge(8, 11, label=r'<<b>r</b>>', fontsize=fontsize_of_leaf_label)
    tree.add_edge(8, 12, label=r'<<b>p</b>>', fontsize=fontsize_of_leaf_label,)
    # Leaf 1
    tree.add_edge(9, 13, label=r'<<b>l</b>>', fontsize=fontsize_of_leaf_label)
    tree.add_edge(9, 14, label=r'<<b>r</b>>', fontsize=fontsize_of_leaf_label)
    tree.add_edge(9, 15, label=r'<<b>p</b>>', fontsize=fontsize_of_leaf_label)
    # Leaf 2
    tree.add_edge(5, 16, label=r'<<b>l</b>>', fontsize=fontsize_of_leaf_label)
    tree.add_edge(5, 17, label=r'<<b>r</b>>', fontsize=fontsize_of_leaf_label)
    tree.add_edge(5, 18, label=r'<<b>p</b>>', fontsize=fontsize_of_leaf_label)
    # Leaf 3
    tree.add_edge(6, 19, label=r'<<b>l</b>>', fontsize=fontsize_of_leaf_label)
    tree.add_edge(6, 20, label=r'<<b>r</b>>', fontsize=fontsize_of_leaf_label)
    tree.add_edge(6, 21, label=r'<<b>p</b>>', fontsize=fontsize_of_leaf_label)

    # Leaf 4
    tree.add_edge(7, 22, label=r'<<b>l</b>>', fontsize=fontsize_of_leaf_label)
    tree.add_edge(7, 23, label=r'<<b>r</b>>', fontsize=fontsize_of_leaf_label)
    tree.add_edge(7, 24, label=r'<<b>p</b>>', fontsize=fontsize_of_leaf_label)

    # Define node labels
    node_labels = {
        # label='<<FONT POINT-SIZE="14">$x^2 + y^2</FONT>>' label="<f0> text | {<f1> text | <f2> text}"
        1: 'has key',
        2: 'key loc',
        3: 'goal loc',
        4: 'goal loc',
        7: r'<<I>z<sub>4</sub></I>>',
        6: r'<<I>z<sub>3</sub></I>>',
        5: r'<<I>z<sub>2</sub></I>>',
        9: r'<<I>z<sub>1</sub></I>>',
        8: r'<<I>z<sub>0</sub></I>>',

        # action leaves
        10: "",
        11: "",
        12: "",

        13: "",
        14: "",
        15: "",

        16: "",
        17: "",
        18: "",

        19: "",
        20: "",
        21: "",

        22: "",
        23: "",
        24: ""

    }
    leaf_action_node_to_state = {

        ## NOT Having Key
        # No Goaal Location, No Key Location, Not having Key
        10: (0, 0),  # zero state, action left
        11: (0, 1),  # zero state, action right
        12: (0, 2),  # zero state, action pick

        # Goaal Location, No Key Location, Not having Key
        13: (9, 0),  # 9th state, action left
        14: (9, 1),  # 9th state, action right
        15: (9, 2),  # 9th state, action pick

        # No Goaal Location, Key Location, Not having Key
        16: (1, 0),  # 1st state, action left
        17: (1, 1),  # 1st state, action right
        18: (1, 2),  # 1st state, action pick

        ## Having Key
        # No Goaal Location, having Key
        19: (10, 0),  # 10th state, action left
        20: (10, 1),  # 10th state, action right
        21: (10, 2),  # 10th state, action pick

        ## Having Key
        #  Goal Location, having Key
        22: (19, 0),  # 19th state, action left
        23: (19, 1),  # 19th state, action right
        24: (19, 2),  # 19th state, action pick

    }

    return tree, node_labels, leaf_action_node_to_state

def plot_reward_tree(reward, output_folder, out_file_name, color_ranges,
        tree, node_labels, leaf_action_node_to_state):

    cmap = cm.get_cmap('RdBu')
    # hm = sns.heatmap(R_shaped, cmap=cmap, vmin=-color_ranges, vmax=color_ranges)

    # Normalize data
    norm = Normalize(vmin=-color_ranges, vmax=color_ranges)
    rgba_values = cmap(norm(reward))

    # Set node and edge attributes for shapes, colors, and labels
    for node in tree.nodes():
        if tree.out_degree(node) == 0:  # Leaf nodes
            node.attr['shape'] = 'square'
            node.attr['style'] = 'filled'
            node.attr['width'] = 0.4
            node.attr['height'] = 0.4
            node.attr['fillcolor'] = \
                rgba_to_hex(rgba_values[leaf_action_node_to_state[int(node)]])

        elif tree.out_degree(node) == 3:  # abstract state
            node.attr['shape'] = 'ellipse'
            node.attr['style'] = 'filled'
            node.attr['fillcolor'] = 'lightblue'
            node.attr['width'] = 0.4
            node.attr['height'] = 0.4
            node.attr["fontsize"] = 32

        else:  # Non-leaf nodes
            node.attr['shape'] = 'box'
            node.attr['style'] = 'filled'
            node.attr['fillcolor'] = 'lightblue'
            node.attr["fontsize"] = 32
            # node.attr["fontweight"] = 'bold'

    # Set edge labels
    for edge in tree.edges():
        tree.get_edge(*edge).attr['labelloc'] = 't'
        edge.attr['lp'] = 't'
        edge.attr['labelloc'] = 't'
        # Set node labels
    for node, label in node_labels.items():
        tree.get_node(node).attr['label'] = label

    # Render the tree to a file
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    output_filename = output_folder + out_file_name
    tree.ratio = "fill"
    tree.draw(output_filename, format='pdf', prog='dot',
              args='-Gsize=5 -Gratio=0.9 -Gsplines=true, -Gnodesep=0.35 -Gedgesep=1.0' )
    tree.close()
#enddef

def plot_convergence(teachers, Use_pool, result_directory, output_directory, n_runs):
    dict = input_from_different_files(result_directory, teachers, n_file=n_runs, t=200001)
    plot_convergence_given_dict(dict, teachers, out_folder_name=output_directory,
                                each_number=50, Use_pool=Use_pool)
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
                         # label=r"$\textsc{Ada-}\text{TL}$",
                         label=r"$\textsc{ExpAdaRD}$",
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
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
               ["0", "", "1", "", "2", "", "3", "", "4", "", "5"])
    plt.yticks([0, 2, 4])

    outFname = os.getcwd() + "/" + out_folder_name + "/convergence_linekey_Use_pool={}.pdf".format(Use_pool)
    plt.savefig(outFname, bbox_inches='tight')
    plt.close()

    pass


if __name__ == "__main__":

  pass