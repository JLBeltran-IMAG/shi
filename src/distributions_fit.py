import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import inspect
import csv
import warnings

warnings.filterwarnings('ignore')

class distributions_functions:
	def __dir__(self):
		return ['alpha', 'anglit', 'arcsine', 'argus', 'beta',
		'betaprime', 'bradford', 'burr', 'burr12', 'cauchy',
		'chi', 'chi2', 'cosine', 'crystalball', 'dgamma',
		'dweibull', 'erlang', 'expon', 'exponnorm',	'exponpow',
		'exponweib', 'f', 'fatiguelife', 'fisk', 'foldcauchy',
		'foldnorm', 'gamma', 'gausshyper', 'genexpon', 'genextreme',
		'gengamma', 'genhalflogistic', 'geninvgauss', 'genlogistic', 'gennorm',
		'genpareto', 'gilbrat', 'gompertz', 'gumbel_l',	'gumbel_r',
		'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb',
		'johnsonsu', 'kappa3', 'kappa4', 'ksone', 'laplace',
		'laplace_asymmetric', 'loggamma', 'logistic', 'loglaplace', 'lognorm',
		'loguniform', 'lomax', 'maxwell', 'mielke', 'moyal',
		'nakagami', 'norm', 'norminvgauss', 'pareto', 'pearson3',
		'powerlaw', 'powerlognorm', 'powernorm', 'rayleigh', 'rdist',
		'recipinvgauss', 'reciprocal', 'rice', 'semicircular', 'skewnorm',
		't', 'trapezoid', 'trapz', 'triang', 'truncexpon',
		'truncnorm', 'uniform', 'wald', 'weibull_max', 'weibull_min',
		'wrapcauchy']

	def alpha(self, x, a, loc = 0, scale = 1):
		return stats.alpha.pdf(x, a, loc, scale)

	def anglit(self, x, loc = 0, scale = 1):
		return stats.anglit.pdf(x, loc, scale)

	def arcsine(self, x, loc = 0, scale = 1):
		return stats.arcsine.pdf(x, loc, scale)

	def argus(self, x, chi, loc = 0, scale = 1):
		return stats.argus.pdf(x, chi, loc, scale)

	def beta(self, x, a, b, loc = 0, scale = 1):
		return stats.beta.pdf(x, a, b, loc, scale)

	def betaprime(self, x, a, b, loc = 0, scale = 1):
		return stats.betaprime.pdf(x, a, b, loc, scale)

	def bradford(self, x, c, loc = 0, scale = 1):
		return stats.bradford.pdf(x, c, loc, scale)

	def burr(self, x, c, d, loc = 0, scale = 1):
		return stats.burr.pdf(x, c, d, loc, scale)

	def burr12(self, x, c, d, loc = 0, scale = 1):
		return stats.burr12.pdf(x, c, d, loc, scale)

	def cauchy(self, x, loc = 0, scale = 1):
		return stats.cauchy.pdf(x, loc, scale)

	def chi(self, x, df, loc = 0, scale = 1):
		return stats.chi.pdf(x, df, loc, scale)

	def chi2(self, x, df, loc = 0, scale = 1):
		return stats.chi2.pdf(x, df, loc, scale)

	def cosine(self, x, loc = 0, scale = 1):
		return stats.cosine.pdf(x, loc, scale)

	def crystalball(self, x, beta, m, loc = 0, scale = 1):
		return stats.crystalball.pdf(x, beta, m, loc, scale)

	def dgamma(self, x, a, loc = 0, scale = 1):
		return stats.dgamma.pdf(x, a, loc, scale)

	def dweibull(self, x, c, loc = 0, scale = 1):
		return stats.dweibull.pdf(x, c, loc, scale)

	def erlang(self, x, a, loc = 0, scale = 1):
		return stats.erlang.pdf(x, a, loc, scale)

	def expon(self, x, loc = 0, scale = 1):
		return stats.expon.pdf(x, loc, scale)

	def exponnorm(self, x, K, loc = 0, scale = 1):
		return stats.exponnorm.pdf(x, K, loc, scale)

	def exponpow(self, x, b, loc = 0, scale = 1):
		return stats.exponpow.pdf(x, b, loc, scale)

	def exponweib(self, x, a, loc = 0, scale = 1):
		return stats.exponweib.pdf(x, a, loc, scale)

	def f(self, x, dfn, loc = 0, scale = 1):
		return stats.f.pdf(x, dfn, loc, scale)

	def fatiguelife(self, x, c, loc = 0, scale = 1):
		return stats.fatiguelife.pdf(x, c, loc, scale)

	def fisk(self, x, c, loc = 0, scale = 1):
		return stats.fisk.pdf(x, c, loc, scale)

	def foldcauchy(self, x, c, loc = 0, scale = 1):
		return stats.foldcauchy.pdf(x, c, loc, scale)

	def foldnorm(self, x, c, loc = 0, scale = 1):
		return stats.foldnorm.pdf(x, c, loc, scale)

	def gamma(self, x, a, loc = 0, scale = 1):
		return stats.gamma.pdf(x, a, loc, scale)

	def gausshyper(self, x, a, b, c, z, loc = 0, scale = 1):
		return stats.gausshyper.pdf(x, a, b, c, z, loc, scale)

	def genexpon(self, x, a, b, c, loc = 0, scale = 1):
		return stats.genexpon.pdf(x, a, b, c, loc, scale)

	def genextreme(self, x, c, loc = 0, scale = 1):
		return stats.genextreme.pdf(x, c, loc, scale)

	def gengamma(self, x, a, c, loc = 0, scale = 1):
		return stats.gengamma.pdf(x, a, c, loc, scale)

	def genhalflogistic(self, x, c, loc = 0, scale = 1):
		return stats.genhalflogistic.pdf(x, c, loc, scale)

	def geninvgauss(self, x, p, b, loc = 0, scale = 1):
		return stats.geninvgauss.pdf(x, p, b, loc, scale)

	def genlogistic(self, x, c, loc = 0, scale = 1):
		return stats.genlogistic.pdf(x, c, loc, scale)

	def gennorm(self, x, beta, loc = 0, scale = 1):
		return stats.gennorm.pdf(x, beta, loc, scale)

	def genpareto(self, x, c, loc = 0, scale = 1):
		return stats.genpareto.pdf(x, c, loc, scale)

	def gilbrat(self, x, loc = 0, scale = 1):
		return stats.gilbrat.pdf(x, loc, scale)

	def gompertz(self, x, c, loc = 0, scale = 1):
		return stats.gompertz.pdf(x, c, loc, scale)

	def gumbel_l(self, x, loc = 0, scale = 1):
		return stats.gumbel_l.pdf(x, loc, scale)

	def gumbel_r(self, x, loc = 0, scale = 1):
		return stats.gumbel_r.pdf(x, loc, scale)

	def hypsecant(self, x, loc = 0, scale = 1):
		return stats.hypsecant.pdf(x, loc, scale)

	def invgamma(self, x, a, loc = 0, scale = 1):
		return stats.invgamma.pdf(x, a, loc, scale)

	def invgauss(self, x, mu, loc = 0, scale = 1):
		return stats.invgauss.pdf(x, mu, loc, scale)

	def invweibull(self, x, c, loc = 0, scale = 1):
		return stats.invweibull.pdf(x, c, loc, scale)

	def johnsonsb(self, x, a, b, loc = 0, scale = 1):
		return stats.johnsonsb.pdf(x, a, b, loc, scale)

	def johnsonsu(self, x, a, b, loc = 0, scale = 1):
		return stats.johnsonsu.pdf(x, a, b, loc, scale)

	def kappa3(self, x, a, loc = 0, scale = 1):
		return stats.kappa3.pdf(x, a, loc, scale)

	def kappa4(self, x, h, k, loc = 0, scale = 1):
		return stats.kappa4.pdf(x, h, k, loc, scale)

	def ksone(self, x, n, loc = 0, scale = 1):
		return stats.ksone.pdf(x, n, loc, scale)

	def laplace(self, x, loc = 0, scale = 1):
		return stats.laplace.pdf(x, loc, scale)

	def laplace_asymmetric(self, x, kappa, loc = 0, scale = 1):
		return stats.laplace_asymmetric.pdf(x, kappa, loc, scale)

	def loggamma(self, x, c, loc = 0, scale = 1):
		return stats.loggamma.pdf(x, c, loc, scale)

	def logistic(self, x, loc = 0, scale = 1):
		return stats.logistic.pdf(x, loc, scale)

	def loglaplace(self, x, c, loc = 0, scale = 1):
		return stats.loglaplace.pdf(x, c, loc, scale)

	def lognorm(self, x, s, loc = 0, scale = 1):
		return stats.lognorm.pdf(x, s, loc, scale)

	def loguniform(self, x, a, b, loc = 0, scale = 1):
		return stats.loguniform.pdf(x, a, b, loc, scale)

	def lomax(self, x, c, loc = 0, scale = 1):
		return stats.lomax.pdf(x, c, loc, scale)

	def maxwell(self, x, loc = 0, scale = 1):
		return stats.maxwell.pdf(x, loc, scale)

	def mielke(self, x, k, s, loc = 0, scale = 1):
		return stats.mielke.pdf(x, k, s, loc, scale)

	def moyal(self, x, loc = 0, scale = 1):
		return stats.moyal.pdf(x, loc, scale)

	def nakagami(self, x, nu, loc = 0, scale = 1):
		return stats.nakagami.pdf(x, nu, loc, scale)

	def norm(self, x, loc = 0, scale = 1):
		return stats.norm.pdf(x, loc, scale)

	def norminvgauss(self, x, a, b, loc = 0, scale = 1):
		return stats.norminvgauss.pdf(x, a, b, loc, scale)

	def pareto(self, x, b, loc = 0, scale = 1):
		return stats.pareto.pdf(x, b, loc, scale)

	def pearson3(self, x, skew, loc = 0, scale = 1):
		return stats.pearson3.pdf(x, skew, loc, scale)

	def powerlaw(self, x, a, loc = 0, scale = 1):
		return stats.powerlaw.pdf(x, a, loc, scale)

	def powerlognorm(self, x, c, s, loc = 0, scale = 1):
		return stats.powerlognorm.pdf(x, c, s, loc, scale)

	def powernorm(self, x, c, loc = 0, scale = 1):
		return stats.powernorm.pdf(x, c, loc, scale)

	def rayleigh(self, x, r, loc = 0, scale = 1):
		return stats.rayleigh.pdf(x, r, loc, scale)

	def rdist(self, x, c, loc = 0, scale = 1):
		return stats.rdist.pdf(x, c, loc, scale)

	def recipinvgauss(self, x, mu, loc = 0, scale = 1):
		return stats.recipinvgauss.pdf(x, mu, loc, scale)

	def reciprocal(self, x, a, b, loc = 0, scale = 1):
		return stats.reciprocal.pdf(x, a, b, loc, scale)

	def rice(self, x, b, loc = 0, scale = 1):
		return stats.rice.pdf(x, b, loc, scale)

	def semicircular(self, x, loc = 0, scale = 1):
		return stats.semicircular.pdf(x, loc, scale)

	def skewnorm(self, x, a, loc = 0, scale = 1):
		return stats.skewnorm.pdf(x, a, loc, scale)

	def t(self, x, df, loc = 0, scale = 1):
		return stats.t.pdf(x, df, loc, scale)

	def trapezoid(self, x, c, d, loc = 0, scale = 1):
		return stats.trapezoid.pdf(x, c, d, loc, scale)

	def trapz(self, x, c, d, loc = 0, scale = 1):
		return stats.trapz.pdf(x, c, d, loc, scale)

	def triang(self, x, c, loc = 0, scale = 1):
		return stats.triang.pdf(x, c, loc, scale)

	def truncexpon(self, x, b, loc = 0, scale = 1):
		return stats.truncexpon.pdf(x, b, loc, scale)

	def truncnorm(self, x, a, b, loc = 0, scale = 1):
		return stats.truncnorm.pdf(x, a, b, loc, scale)

	def uniform(self, x, loc = 0, scale = 1):
		return stats.uniform.pdf(x, loc, scale)

	def wald(self, x, loc = 0, scale = 1):
		return stats.wald.pdf(x, loc, scale)

	def weibull_max(self, x, c, loc = 0, scale = 1):
		return stats.weibull_max.pdf(x, c, loc, scale)

	def weibull_min(self, x, c, loc = 0, scale = 1):
		return stats.weibull_min.pdf(x, c, loc, scale)

	def wrapcauchy(self, x, c, loc = 0, scale = 1):
		return stats.wrapcauchy.pdf(x, c, loc, scale)

def distribution_fitting(x_data, y_data, filepath, filename):
	# Choosing the distributions
	distributions = distributions_functions()
	dist = [getattr(distributions, d) for d in dir(distributions)]

	distributions_name = []
	distributions_n_parameters = []
	distributions_parameters = []
	distributions_errors = []
	errors_value_sum = []

	# Fitting data by functional form of PDF from scipy.stats module
	for dist in dist:
		try:
			# optimal parameters by non-linear least square
			parameters, pcov = curve_fit(dist, x_data, y_data)
			parameters_error = np.sqrt(np.diag(pcov))
			parameters_name = [p for p in inspect.signature(dist).parameters if not p == 'x']

			parameters_name_value = dict(zip(parameters_name, parameters))
			parameters_name_error = dict(zip(parameters_name, parameters_error))
			errors_value_sum.append(np.sum(parameters_error))

			distributions_name.append(dist.__name__)
			distributions_n_parameters.append(len(parameters))
			distributions_parameters.append(parameters_name_value)
			distributions_errors.append(parameters_name_error)

		except Exception as e:
			print(f"Error in fitting of distribution {dist.__name__}")

	# Save results in DataFrame
	fitting_results = pd.DataFrame({
		'Name': distributions_name,
		'Number of parameters': distributions_n_parameters,
		'Parameters': distributions_parameters,
		'Error': distributions_errors,
		'Sum of errors': errors_value_sum
		})

	fitting_results = fitting_results.sort_values(by = 'Sum of errors').reset_index(drop = True)
	# Exporting DataFrame to excel file
	fitting_results.to_excel("{}/{}.xlsx".format(filepath, filename))
	return fitting_results


def plotting_fitting_results(x_data, y_data, number_plot):
	# Fitting input data by using distribution_fitting
	fitting_dataframe = distribution_fitting(x_data, y_data)

	# Choosing the best distributions to plot
	distributions = distributions_functions()

	first_best_distributions = []
	for i in range(number_plot):
		first_best_distributions.append(fitting_dataframe['Name'][i])

	dist = [getattr(distributions, d) for d in first_best_distributions]

	# Creating subplots
	fig, ax = plt.subplots()
	ax.stem(x_data, y_data, label = 'Input Data')

	x_axis_from_fit = np.linspace(np.min(x_data), np.max(x_data), len(x_data))
	for i in range(number_plot):
		try:
			parameters = fitting_dataframe['Parameters'][i].values()
			y_axis_from_fit = dist[i](x_axis_from_fit, *parameters)
			ax.plot(x_axis_from_fit, y_axis_from_fit, label = dist[i].__name__)

		except Exception as e:
			print(f"Error attempt to plot distribution {dist[i].__name__}")

	ax.legend()
	plt.savefig('fitting_graph.png')





