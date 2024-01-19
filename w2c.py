import numpy as np
import matplotlib.pyplot as plt

plt.figure()

# xx = []
# for n in range(10**5):
#     xx.append(np.random.uniform(0,1,10).sum()-5)

xx = np.sum(-np.log(np.random.rand(10**5, 50)), 1)    

histogram = plt.hist(xx,500)

mean = np.mean(xx)
var = np.var(xx)

cc, bin_centres = histogram[0], histogram[1]
pdf = np.exp(-0.5 * (bin_centres - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)
bin_width = bin_centres[2] - bin_centres[1]
predicted_bin_heights = pdf * 10**5 * bin_width
plt.plot(bin_centres, predicted_bin_heights, '-r', linewidth=3)

plt.show()
