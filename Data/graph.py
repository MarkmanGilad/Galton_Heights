import torch
import numpy as np
import matplotlib.pyplot as plt


# From Udemy, A deep understanding of deep learning, Mike X Cohen
# the "peaks" function
def peaks(x,y):
  # expand to a 2D mesh
  x,y = np.meshgrid(x,y)

#   z = 3*(1-x)**2 * np.exp(-(x**2) - (y+1)**2) \
#       - 10*(x/5 - x**3 - y**5) * np.exp(-x**2-y**2) \
#       - 1/3*np.exp(-(x+1)**2 - y**2)
  z = x * np.exp(-(x**2+y**2))
  return z

# create the landscape
x = np.linspace(-3,3,201)
y = np.linspace(-3,3,201)

z = peaks(x,y)

# let's have a look!
plt.imshow(z,extent=[x[0],x[-1],y[0],y[-1]],vmin=-0.5,vmax=0.5,origin='lower')
plt.show()
