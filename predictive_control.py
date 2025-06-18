import numpy as np
import matplotlib.pyplot as plt

# Setpoints for w1 and w2 as provided
w1 = [0,0,0,0,0,0,0,10,10,10,10,10,10,10,0,0,0,0,0,0,0,10,10,10,10,10,10,10]
w2 = [10,10,10,10,10,10,10,0,0,0,0,0,0,0,10,10,10,10,10,10,10,0,0,0,0,0,0,0]

time_steps = len(w1)
assert len(w2) == time_steps

# Parameters for the simple process model
A = 1.0       # state coefficient (integrator)
B = 0.1       # input coefficient
lam = 1.0     # input penalty in cost function
N = 5         # prediction horizon

# Initialize arrays
u1 = np.zeros(time_steps)
u2 = np.zeros(time_steps)
y1 = np.zeros(time_steps+1)
y2 = np.zeros(time_steps+1)

# Precompute matrices for the simple MPC
F = B * np.tril(np.ones((N, N)))
H = F.T @ F + lam * np.eye(N)
invH = np.linalg.inv(H)

for k in range(time_steps):
    # Build reference vectors for the next N steps
    ref1 = np.array([w1[min(k+i, time_steps-1)] for i in range(1, N+1)])
    ref2 = np.array([w2[min(k+i, time_steps-1)] for i in range(1, N+1)])

    delta1 = ref1 - y1[k]
    delta2 = ref2 - y2[k]

    U1 = invH @ (F.T @ delta1)
    U2 = invH @ (F.T @ delta2)

    u1[k] = U1[0]
    u2[k] = U2[0]

    # Update process state
    y1[k+1] = A*y1[k] + B*u1[k]
    y2[k+1] = A*y2[k] + B*u2[k]

# Remove the first element to align with time steps
y1 = y1[1:]
y2 = y2[1:]

# Plotting results
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.step(range(time_steps), w1, label='w1 (setpoint)', where='post')
plt.plot(y1, label='y1 (output)')
plt.ylabel('y1')
plt.legend()

plt.subplot(2,1,2)
plt.step(range(time_steps), w2, label='w2 (setpoint)', where='post')
plt.plot(y2, label='y2 (output)')
plt.ylabel('y2')
plt.xlabel('Time step')
plt.legend()

plt.tight_layout()
plt.show()
