import matplotlib.pyplot as plt
import numpy as np


def plot_3D(xyz, filename):
    xyz = xyz.squeeze()
    Nx, Ny, Nz = xyz.shape
    X, Y, Z = np.meshgrid(
        np.linspace(-2.67659784, 4.58195351, Nx), 
        np.linspace(-0.08062697, 3.04455779, Ny), 
        np.linspace(-4.83431278, 8.16458168, Nz)
    )
    data = xyz
    
    kw = {
        'vmin': -2.5,
        'vmax': 2.5,
        'levels': np.linspace(-2.5, 2.5, 100),
    }

    # Create a figure with 3D ax
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    # Plot contour surfaces
    _ = ax.contourf(
        X[:, :, 128], Y[:, :, 128], data[:, :, 128],
        zdir='z', offset=Z.max(), **kw
    )
    _ = ax.contourf(
        X[6, :, :], data[6, :, :], Z[6, :, :],
        zdir='y', offset=Y.min(), **kw
    )
    C = ax.contourf(
        data[:, 95, :], Y[:, 95, :], Z[:, 95, :],
        zdir='x', offset=X.max(), **kw
    )
    # --


    # Set limits of the plot from coord limits
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

    # Plot edges
    edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
    ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
    ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

    # Set labels and zticks
    ax.set(
        xlabel='X [m]',
        ylabel='Y [m]',
        zlabel='Z [m]',
        zticks=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    # Set zoom and angle view
    ax.view_init(40, -30, 0)
    ax.set_box_aspect(None, zoom=0.9)

    # Colorbar
    fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='Name [units]')

    # Show Figure
    # plt.show()
    plt.savefig(filename)