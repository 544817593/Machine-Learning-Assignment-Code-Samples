import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

xx = 1 * (np.random.rand(100) < 0.3)

sampleMeans = []

for i in range(10):
   sampleMeans.append(np.mean(xx[10*i:10*(i+1)]))

print(xx)

# Estimate population mean by group of tens
print(sampleMeans)

sampleMean = np.sum(sampleMeans)/10

print("sampleMean: "+str(sampleMean))

sampleSigmaSquared = (1/9)*np.sum((sampleMeans-sampleMean)**2)

errorBar = sampleSigmaSquared**0.5 / 10**0.5

spErrorBar = stats.sem(sampleMeans)
print("spErrorBar: "+ str(spErrorBar))

print("errorBar: "+ str(errorBar))