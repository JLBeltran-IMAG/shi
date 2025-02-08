import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def confidence_ellipse(absorption_values, scattering_values, ax, ellipse_color, confidence_level = 0.95, degrees_of_freedom = 2):
    """
    Create a plot of the covariance confidence ellipse of *absorption_values* and *scattering_values*.

    Parameters
    ----------
    absorption_values, scattering_values : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if absorption_values.size != scattering_values.size:
        raise ValueError("absorption_values and scattering_values must be the same size")

    cov = np.cov(absorption_values, scattering_values)
    eingenvalues, eigenvectors = np.linalg.eig(cov)

    k = np.sqrt(chi2.ppf(confidence_level, degrees_of_freedom))

    axes_x = 2 * k * np.sqrt(np.abs(eingenvalues[0]))
    axes_y = 2 * k * np.sqrt(np.abs(eingenvalues[1]))

    mean_x = np.mean(absorption_values)
    mean_y = np.mean(scattering_values)

    scale_x = np.sqrt(cov[0, 0])
    scale_y = np.sqrt(cov[1, 1])

    angle = angle = np.rad2deg(np.arctan2(eigenvectors[:, 0][1], eigenvectors[:, 0][0]))

    ellipse = Ellipse(xy = (mean_x, mean_y), width = axes_x, height = axes_y, angle = angle, ec = ellipse_color, fc = "None", lw = 2)

    return ax.add_patch(ellipse)


def correlation_pearson(cluster_1, cluster_2):
    return np.corrcoef(cluster_1, cluster_2)



# absorption_values = np.arange(0, 101)
# scattering_values = np.power(np.arange(0, 101), 2)

# fig = plt.figure()
# ax = fig.subplots()

# ax.plot(absorption_values, scattering_values)
# confidence_ellipse(absorption_values, scattering_values, ax = ax)

# plt.show()

