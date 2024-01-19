import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special


# 3.2
# mu = int(2)
# sigma = int(3)

# result = integrate.quad(lambda x: np.exp(-((1/(2*sigma**2))*(x-mu)**2)),-np.inf,np.inf)
# print(result)

# mu = 2.0
# sigma = 3.0
# delta = sigma / 100.0
# xx = np.arange(mu - 10*sigma, mu + 10*sigma, delta)
# integrand = np.exp(-0.5 * (xx - mu)**2 / sigma**2)
# Zapprox = np.sum(integrand * delta)
# print("Zapprox = %g" % Zapprox)



# 5
N = int(1e6)
xx = np.random.randn(N)
hist_stuff = plt.hist(xx, bins=100)
# plt.show()

mu = np.mean(xx)
sigma = np.var(xx)**0.5
print(np.mean(xx))
print(np.var(xx))

bin_centres = 0.5*(hist_stuff[1][1:] + hist_stuff[1][:-1])
# Fill in an expression to evaluate the PDF at the bin_centres.
# To square every element of an array, use **2
# pdf = ((np.exp(-((xx-mu)**2)/2*sigma**2)/(sigma*(2*np.pi)**0.5)))
pdf = np.exp(-0.5 * bin_centres**2) / np.sqrt(2 * np.pi)
bin_width = bin_centres[1] - bin_centres[0]
predicted_bin_heights = pdf * N * bin_width
# pdf needs scaling correctly
# Finally, plot the theoretical prediction over the histogram:
plt.plot(bin_centres, predicted_bin_heights, '-r')
plt.show()