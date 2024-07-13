import math
import numpy as np
import matplotlib.pyplot as plt

def log_likelihood(sigma_x, sigma_y, rho, xgrid, ygrid):
    # Calculate the log likelihood of the 2D Gaussian PDF
    normalization = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))
    exp_component = np.exp(-1 / (2 * (1 - rho**2)) * (
        (xgrid**2 / sigma_x**2) +
        (ygrid**2 / sigma_y**2) -
        (2 * rho * xgrid * ygrid / (sigma_x * sigma_y))
    ))
    like = normalization * exp_component
    return like

# -- create grid --
L = 5  # Adjust the range as needed
N = 500  # Number of grid points
xgrid, ygrid = np.meshgrid(np.linspace(-L, L, N), np.linspace(-L, L, N))

# -- create plot --
ginfo = {"top":0.90,"bottom":0.09,"left":0.05,"right":0.99,'hspace':0.5}
fig, axes = plt.subplots(2, 3, figsize=(15, 10), gridspec_kw=ginfo)
sigma_y = 1.
sigma_x_list = [1, 2]
rho_list = [0., 0.8, -0.8]
cs = None
for ix, sigma_x in enumerate(sigma_x_list):
    for jx, rho in enumerate(rho_list):
        ax = axes[ix][jx]
        log_like = log_likelihood(sigma_x, sigma_y, rho, xgrid, ygrid)
        if cs is None:
            cs = ax.contour(xgrid, ygrid, log_like)
        else:
            ax.contour(xgrid, ygrid, log_like, levels=cs.levels)
        ax.set_xlabel("x")
        if jx == 0:
            ax.set_ylabel("y")
        else:
            ax.set_ylabel("")
        ax.tick_params(axis='both', left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        sigma_label = r"$\sigma_x=" + ("%d" % sigma_x) + "$"
        rho_label = r"$\rho=" + ("%1.1f" % rho) + "$"
        ax.set_title(sigma_label + ", " + rho_label)
plt.savefig("plots/hw5_mvn_pdf.png", dpi=200)
plt.close("all")
