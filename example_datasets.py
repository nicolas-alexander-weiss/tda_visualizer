"""
@author: Nicolas Weiss

@date: 2024-10-29

"""

import numpy as np

import gudhi

import plotly.express as px

import time


from matplotlib import ticker
from matplotlib import colormaps
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# B from points in [0,1], square.

simple_B_2 = np.array([
       [0.35119048, 0.95021645],
       [0.69480519, 0.92857143],
       [0.79761905, 0.78787879],
       [0.8219697 , 0.60660173],
       [0.6271645 , 0.49296537],
       [0.38095238, 0.49025974],
       [0.80032468, 0.36309524],
       [0.79491342, 0.16017316],
       [0.61904762, 0.04112554],
       [0.34307359, 0.05465368],
       [0.34307359, 0.27651515],
       [0.34577922, 0.73376623],
       [0.51352814, 0.95292208],
       [0.1969697 , 0.95292208],
       [0.20508658, 0.06277056]])
 

simple_B = np.array([[0.01298701, 0.97727273],
       [0.19967532, 0.96915584],
       [0.38095238, 0.96374459],
       [0.58658009, 0.97727273],
       [0.73268398, 0.93668831],
       [0.88690476, 0.83658009],
       [0.91396104, 0.75      ],
       [0.9004329 , 0.60119048],
       [0.71645022, 0.53625541],
       [0.56222944, 0.53354978],
       [0.56222944, 0.53354978],
       [0.41071429, 0.54166667],
       [0.23484848, 0.54437229],
       [0.23484848, 0.54437229],
       [0.66504329, 0.47402597],
       [0.81114719, 0.43885281],
       [0.87878788, 0.31439394],
       [0.88690476, 0.14664502],
       [0.74891775, 0.02489177],
       [0.52435065, 0.02489177],
       [0.33495671, 0.0275974 ],
       [0.12391775, 0.04112554],
       [0.26190476, 0.379329  ],
       [0.25919913, 0.28192641],
       [0.22402597, 0.86093074],
       [0.23755411, 0.72564935],
       [0.27002165, 0.12770563]])

centered_B = simple_B_2 - [0.5, 0.5]

simple_A = np.array([[0.48917749, 0.85281385],
       [0.35119048, 0.7012987 ],
       [0.65963203, 0.69859307],
       [0.21049784, 0.53896104],
       [0.4025974 , 0.53625541],
       [0.57034632, 0.54707792],
       [0.76244589, 0.53896104],
       [0.07792208, 0.39556277],
       [0.87337662, 0.40638528],
       [0.00757576, 0.24404762],
       [0.97077922, 0.26569264]])

centered_A = simple_A - [0.5, 0.5]

big_A = centered_A * 8

centered_A_of_Bs =  np.vstack([p + centered_B for p in big_A])

a_of_bs = {
    "data" : centered_A_of_Bs,
    "x_range" : (-4,4),
    "y_range" : (-4,4), 
    "distance_tresholds" : np.linspace(0,0.5, 10),
    "file_prefix" : "img/a_of_bs"
}

two_squares = { "data" : np.array([[0.39718615, 0.20616883],
       [0.60010823, 0.21428571],
       [0.78950216, 0.3982684 ],
       [0.80573593, 0.59577922],
       [0.59199134, 0.80140693],
       [0.39177489, 0.81493506],
       [0.19426407, 0.59848485],
       [0.20238095, 0.3982684 ],
       [0.70292208, 0.80952381],
       [0.81655844, 0.80140693],
       [0.59469697, 0.71482684],
       [0.59199134, 0.62012987],
       [0.69751082, 0.61201299],
       [0.80844156, 0.7012987 ],
       [0.20508658, 0.7987013 ],
       [0.19426407, 0.19534632],
       [0.80032468, 0.2034632 ]]),
       "min_persistence" : 0.05,
       "x_range" : (0,1),
       "y_range" : (0,1),
       "distance_tresholds" : np.array([0.0, 0.1, 0.13, 0.16, 0.2, 0.3, 0.4, 0.5]),
       "file_prefix" : "img/two_squares"
}

small_big_circle = { "data": np.array([[0.25649351, 0.28463203],
       [0.18344156, 0.28463203],
       [0.11850649, 0.25757576],
       [0.09415584, 0.18993506],
       [0.09415584, 0.18993506],
       [0.09415584, 0.10606061],
       [0.16179654, 0.06547619],
       [0.16179654, 0.06547619],
       [0.26190476, 0.06277056],
       [0.30790043, 0.11147186],
       [0.32142857, 0.16287879],
       [0.28896104, 0.23593074],
       [0.18614719, 0.2521645 ],
       [0.13203463, 0.18452381],
       [0.16179654, 0.11417749],
       [0.25649351, 0.13311688],
       [0.31872294, 0.21699134],
       [0.4215368 , 0.8771645 ],
       [0.4215368 , 0.8771645 ],
       [0.23484848, 0.72564935],
       [0.3538961 , 0.58766234],
       [0.31331169, 0.44155844],
       [0.61363636, 0.27922078],
       [0.86255411, 0.46320346],
       [0.90854978, 0.6715368 ],
       [0.78409091, 0.83387446],
       [0.61904762, 0.93398268],
       [0.89231602, 0.89880952],
       [0.43235931, 0.31168831],
       [0.21861472, 0.57954545]]),
       "min_persistence" : 0.05,
        "x_range" : (0,1),
        "y_range" : (0,1), 
        "distance_tresholds" : np.linspace(0,0.5, 10),
        "file_prefix" : "img/small_big_circle"
}  




def scatter_disks(ax, data2d, radius, color, alpha, x_range, y_range):
    patches = [ mpatches.Circle(xy=(x,y), radius=radius, alpha=alpha, facecolor=color) 
        for (x,y) in data2d]
    for patch in patches:
        ax.add_artist(patch)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal", adjustable="box")



def get_bars(persistence, max_len=10):
    # CODE FROM GUDHI
    # Takes the bars in a diagram and outputs the list of (d, (start, len))
    x = np.array([birth for (dim, (birth, death)) in persistence])
    y = np.array([(death - birth) if death != float("inf") else (max_len - birth) for (dim, (birth, death)) in persistence])
    d = np.array([dim for (dim, (birth, death)) in persistence], dtype = np.int32)

    return np.transpose(np.vstack((d,x,y)))



def draw_vertical_line(ax, x0, y_range, color="black"):
    ax.add_artist(
        Line2D([x0, x0], y_range, color=color)
    )



def draw_barcode(ax, bars, max_time):

    # bars = get_bars(diag, max_len=max_len)

    dims = [int(i) for i in bars[:, 0]]
    # colors = [COLOR_PALETTE for i in bars[:,0]]  # assign colors based on dim.
    offset = bars[:, 1]
    length = bars[:, 2]

    x_range = (0, max_time)
    y_range = (0 -1 , len(bars) + 1)



    # For each i, plot one bar
    bar_patches = []

    for i, bar in enumerate(bars):
        dim = int(bar[0])
        color = COLOR_PALETTE[dim]
        offset = bar[1]
        length = bar[2]

        # just draw a line?
        bar_patches.append(
            Line2D(xdata=[offset, offset + length], ydata=[i, i], color=color)
        )

    for patch in bar_patches:
        ax.add_artist(patch)

    # adjust the ratio

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    # ax.set_aspect("equal", adjustable="box")

    # Change y ticker:
    ax.yaxis.set_major_locator(ticker.NullLocator())

    # Legend:
    legend_elements = [mpatches.Patch(facecolor=COLOR_PALETTE[d], label=f'H_{d}')
                        for d in set(dims)]
    
    ax.legend(handles=legend_elements, loc='upper right')


def draw_betti_curve(ax, bars, max_time):
    
    # TODO: Add this.

    # sort by dimension

    # Then for each dimension:
    # Compute the curve: 
    # - by placing a node (x) value for each start and end. --> turn into lambda function.
    # - And sum them up.

    occuring_dimensions = set([int(d) for d in bars[:,0]])

    betti_num_change_by_dim = {d:{0:0, max_time+1:0} for d in occuring_dimensions} # and already add 0 for start and end

    #change_points_by_dim = {d:[] for d in occuring_dimensions} # x-values where the betti curve might change.

    for bar in bars:
        d = bar[0] # dimension
        offset = bar[1]
        length = bar[2]

        start = offset
        end = offset + length
        

        betti_num_change_by_dim[d][start] = 1 if (start not in betti_num_change_by_dim[d].keys()) else 1 + betti_num_change_by_dim[d][start]
        betti_num_change_by_dim[d][end] = -1 if (end not in betti_num_change_by_dim[d].keys()) else -1 + betti_num_change_by_dim[d][end]

    # Now construct the curve, by going from left to the right through the change points.

    betti_curves_by_dim = {d:{"xs":[], "ys":[]} for d in occuring_dimensions}

    for d in occuring_dimensions:

        for (i,x) in enumerate(sorted(betti_num_change_by_dim[d].keys())):
    
            change_at_x = betti_num_change_by_dim[d][x]

            if i == 0:
                # just add current value.
                betti_curves_by_dim[d]["xs"].append(x)
                betti_curves_by_dim[d]["ys"].append(change_at_x)
                continue
            
            # repeat the last value 
            last_val = betti_curves_by_dim[d]["ys"][-1]
            betti_curves_by_dim[d]["xs"].append(x)
            betti_curves_by_dim[d]["ys"].append(last_val) # repeat the last value.

            # Add new value if changed.
            if change_at_x != 0:
                betti_curves_by_dim[d]["xs"].append(x)
                betti_curves_by_dim[d]["ys"].append(last_val + change_at_x)

            #print(f"[dim {d}] Betti number at {x} = {last_val + change_at_x}")

    # For now, plot all curves on top of each other. Though would like to have separate window.
    for d in occuring_dimensions:
        ax.plot(betti_curves_by_dim[d]["xs"], betti_curves_by_dim[d]["ys"], color=COLOR_PALETTE[d], linestyle="-")
    
    # set x_range
    ax.set_xlim((0, max_time))

    # Legend:
    legend_elements = [mpatches.Patch(facecolor=COLOR_PALETTE[d], label=f'H_{d}')
                        for d in occuring_dimensions]
    
    ax.legend(handles=legend_elements, loc='upper right')


def draw_2dsimplicial_complex(ax, data2d, simpl_tree, treshold, x_range, y_range):

    # print([s[0] for s in simpl_tree if len(s[0]) == 2][:10])

    # Get triangles and edges:
    triangles = []
    edges = []
    for s in simpl_tree:
        if (len(s[0]) == 3 and s[1] <= treshold):
            triangles.append(s[0])
        elif (len(s[0]) == 2 and s[1] <= treshold):
            edges.append(s[0])
    
    patches = []
          

    # print triangles
    patches += [
        mpatches.Polygon([data2d[i] for i in twosimplex],closed=True, fill=True, facecolor=COLOR_2_SIMPLEX, alpha=ALPHA_2_SIMPLEX)

        for twosimplex in triangles
    ]
    # print edges
    patches += [
        Line2D([data2d[e[0]][0], data2d[e[1]][0]], [data2d[e[0]][1], data2d[e[1]][1]], linestyle="-", color=COLOR_1_SIMPLEX)

        for e in edges
    ]
    # print point
    patches += [
        mpatches.Circle(xy=(x,y), radius=POINT_RADIUS, alpha=1, facecolor=COLOR_0_SIMPLEX) 
        for (x,y) in data2d
    ]


    # Add all patches:

    for patch in patches:
        ax.add_artist(patch)   # TODO: Keep handle to the artists.


    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal", adjustable="box")





POINT_RADIUS = 0.01
ALPHA_THICKENING = 1
ALPHA_2_SIMPLEX = 0.3
COLOR_2_SIMPLEX = "grey"
COLOR_1_SIMPLEX = "green"
COLOR_0_SIMPLEX = "blue"

COLOR_PALETTE = colormaps["Set1"].colors

def draw_epsilon_thickening(ax, data, radius, x_range, y_range):
    # scatter disks over it:
    scatter_disks(ax, data, radius=radius, color="gray", alpha=ALPHA_THICKENING, x_range=x_range, y_range=y_range)
    # scatter center points over it:
    scatter_disks(ax, data, radius=POINT_RADIUS, color="blue", alpha=1, x_range=x_range, y_range=y_range)


def visualization_barcode(data2d, x_range, y_range, radius, min_persistence=0.05, savename=None):
    """
    Create 1:2 view.  (2x2 grid)

    
    A out of B's with disks | Betti curve 
    ---------------------------------------
    Rips complex            | Barcode


    Other half for showing Betti curve as well as barcode.

    Have a sliding parameter for showing a bar as well as the current state of the 

    """
    
    # plot the points as disks:
    # delta_rel = 1.2
    # xs = data2d[:, 0]
    # ys = data2d[:, 1]

    # x_range = (min(xs)*delta_rel, max(xs)*delta_rel)
    # y_range = (min(ys)*delta_rel, max(ys)*delta_rel)

    # compute Rips complex 

    distances = [0]
    for i in range(0, len(data2d)):
        for j in range(i+1, len(data2d)):
            distances.append(np.sqrt(np.sum(np.square(data2d[i] - data2d[j]))))
    
    diam = max(distances)
    max_len= diam + 0.1

    rips_complex = gudhi.RipsComplex(points=data2d, max_edge_length=max_len)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    two_skeleton = simplex_tree.get_skeleton(2)
    diag = simplex_tree.persistence(min_persistence=min_persistence)

    bars = get_bars(diag, max_len=max_len)

    make_plot(radius, data2d, bars, x_range, y_range, two_skeleton, max_len, savename)
        
    

    
def make_plot(radius, data2d, bars, x_range, y_range, two_skeleton, max_len, savename=None):
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(19.20,10.80)
    #
    # Set Titles for the subplots.
    axs[0][0].title.set_text("Epsilon Thickening of X")
    axs[1][0].title.set_text("Vietoris-Rips-Complex (2-skeleton)")
    axs[0][1].title.set_text("Betti Curves")
    axs[1][1].title.set_text("Barcodes")

    # Draw the epsilon thickening:
    draw_epsilon_thickening(axs[0][0], data2d, radius, x_range, y_range)

    # Draw the simplicial complex in bottom left.
    distance_treshold = 2*radius

    draw_2dsimplicial_complex(axs[1][0], data2d, simpl_tree=two_skeleton, treshold=distance_treshold, x_range=x_range, y_range=y_range)

    # Draw the barcode in the bottom right:

    draw_barcode(axs[1][1], bars, max_time=max_len)

    # Draw the betti curve in the top left:

    draw_betti_curve(axs[0][1], bars, max_time=max_len)

    ## Add vertical line for treshold
    draw_vertical_line(axs[1][1], x0=distance_treshold, y_range=axs[1][1].get_ylim())
    draw_vertical_line(axs[0][1], x0=distance_treshold, y_range=axs[0][1].get_ylim())

    if savename != None:
        fig.savefig(f"{savename}.png", dpi=100)
    else:
        plt.show()
    plt.close()





def show_A_of_Bs():

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set limits, aspect ratio and grid
    # ax.set_xlim(x_range)
    # ax.set_ylim(y_range)
    ax.set_aspect("equal", adjustable="box")
    # ax.grid(True)


    # ax.scatter(big_A[:,0], big_A[:,1])

    ax.scatter(centered_A_of_Bs[:,0], centered_A_of_Bs[:,1])


    plt.show()




if __name__ == "__main__":

    data_sets_to_process = [
        two_squares,
        small_big_circle,
        a_of_bs
    ]

    for dataset in data_sets_to_process:
        data2d = dataset["data"]
        #distance_tresholds_two_squares = [0.0, 0.1, 0.15,0.2, 0.4, 0.5]

        for i, radius in enumerate(dataset["distance_tresholds"]/2):
            file_prefix = dataset["file_prefix"]
            visualization_barcode(data2d, x_range=dataset["x_range"], y_range=dataset["y_range"], radius=radius, 
                                  min_persistence = dataset["min_persistence"], max_len=data_set["max_len"], savename=f"{file_prefix}_{i}")
