from .generalfunctions import *


def plot_surface_3d(Z, lengths, title=None, xy_labels=None, range=None, show=True):
    """This plots the grid of values as a surface in a (Lx,Ly) space

    Args:
        Z (np.array): a 2d array of values to plot
        lengths ([type]): system size tuple
        title ([type], optional): Title of the plot. Defaults to None.
        xy_labels ([type], optional): Labels for x and y axes. Defaults to None.
    """

    x = np.arange(lengths[0])
    y = np.arange(lengths[1])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if range is not None:
        ax.set_zlim(range[0], range[1])
    if title is not None:
        plt.title(title)
    if xy_labels is not None:
        plt.xlabel(xy_labels[0])
        plt.ylabel(xy_labels[1])
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    if show == True:
        plt.show()


def plot_triangulation(x_vals, y_vals, z_vals, title=None, xy_labels=None, range=None, show=True, cmap=None):

    ax = plt.axes(projection='3d')
    if cmap is not None:
        ax.plot_trisurf(x_vals, y_vals, z_vals, cmap=cmap, edgecolor='none')
    else:
        ax.plot_trisurf(x_vals, y_vals, z_vals, cmap='viridis',
                        edgecolor='none')
    if range is not None:
        ax.set_zlim(range[0], range[1])
    if title is not None:
        plt.title(title)
    if xy_labels is not None:
        plt.xlabel(xy_labels[0])
        plt.ylabel(xy_labels[1])
    if show == True:
        plt.show()


def cmap_triangulation(x_vals, y_vals, z_vals, title=None, xy_labels=None, range=None, show=True, cmap=None):
    if cmap is not None:
        plt.tripcolor(x_vals, y_vals, z_vals, cmap=cmap)
    else:
        plt.tripcolor(x_vals, y_vals, z_vals)
    plt.colorbar()
    if range is not None:
        plt.clim(range[0], range[1])
    if title is not None:
        plt.title(title)
    if xy_labels is not None:
        plt.xlabel(xy_labels[0])
        plt.ylabel(xy_labels[1])

    if show == True:
        plt.show()
