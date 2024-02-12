import numpy as np
import matplotlib.pyplot as plt

approx = []
for n in np.arange(10,10000,10):
    x = np.random.randint(-10000,10000,n).reshape(-1,1)/10000
    y = np.random.randint(-10000,10000,n).reshape(-1,1)/10000

    coords = np.concatenate((x,y),1)

    circ_sum = sum(((coords[:,0]**2)+coords[:,1]**2)<=1)

    approx.append(np.pi - (4*(circ_sum/n)))
    
f, ax = plt.subplots(1, 1, figsize=(8, 4))
plt.plot(np.arange(10,10000,10),approx)
