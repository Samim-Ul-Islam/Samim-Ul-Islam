import numpy as np
import matplotlib.pyplot as plt

# Define parameters
def Yeq(x):
    return 0.145*(x**(3/2))*np.exp(-x)
def weq(x):
    return np.log(0.145*(x**(3/2))*np.exp(-x))
sv=1e-33
lb=1000
f = lambda t, s: (lb/t**2)*(np.exp(2*weq(t)-s)-np.exp(s)) # ODE
h = 0.001 # Step size
t = np.arange(1, 100 + h, h) # Numerical grid
s0 = -2.937 # Initial Condition
# Explicit Euler Method
s = np.zeros(len(t))
s[0] = s0
for i in range(0, len(t) - 1):
    s[i + 1] = s[i] + h*f(t[i], s[i])
Y=np.exp(s)
plt.plot(t, Y, 'g')
plt.plot(t, 0.145*(t**(3/2))*np.exp(-t),'r')
plt.title('Yield for Relic Abundance')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.show()
