from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from circuitTools import *

radius = 1

num_points = 50

'''
# Uniformly sample points on a sphere using spherical coordinates
phi = np.random.uniform(0, np.pi, size=num_points)  # polar angle
theta = np.random.uniform(0, 2*np.pi, size=num_points)  # azimuthal angle

# Define plane: z = a*x + b*y
a, b = 2.5, 0

X_data = []
y_data = []

for i in range (len(phi)):
    x = radius * np.sin(phi[i]) * np.cos(theta[i])
    y = radius * np.sin(phi[i]) * np.sin(theta[i])
    z = radius * np.cos(phi[i])
    if z > a * x + b * y:
        X_data.append((phi[i], theta[i]))
        y_data.append(1)
    else :
        X_data.append((phi[i], theta[i]))
        y_data.append(-1)
'''

# randomly generated data
X_data = [(0.4461365358947761, 0.25911862607262476), (3.0926821747170816, 4.600862637413524), (1.7851245408203378, 1.6361694074503343), (1.9124693788066913, 3.2051979100912438), (2.5489585134527712, 3.0832701695158193), (1.1119669626793638, 2.2683261271671786), (2.1282136759406027, 3.4812172965020314), (0.9565170037781868, 3.4572669384429444), (2.337536772610973, 5.599862618257412), (2.127219261964461, 1.64552119970525), (0.679489446821518, 4.097071836929742), (2.981766539352059, 6.023620055331134), (1.1561388367356276, 4.267214069771276), (0.34307846126965363, 1.0628606837933972), (2.2192079444411896, 4.537265875814268), (1.4521912423318428, 3.139838869795846), (1.4076355497355924, 5.769788843993014), (2.1205421869376857, 1.8907250654815846), (2.746406212999902, 3.7665343255289545), (0.5690562924819066, 0.49834092685876924), (0.328835647459121, 2.2766384229726477), (0.12654800697106378, 3.9430708705266913), (0.6273734495018801, 4.809648529309832), (1.0698911716319843, 1.0031065627284872), (1.6175620233012, 1.9573472432391696), (2.8325306325805353, 5.241472091321098), (0.22964292821655669, 5.125986032899986), (2.797745037146566, 4.42351335987515), (1.022114811089987, 5.522285700844636), (2.6973311856359445, 4.478932234052846), (2.991137843914753, 0.525828749709142), (2.061831383099918, 1.3232765967303397), (0.0711998006526504, 3.3710498971416873), (2.088610634604043, 1.5219950457537497), (0.7035843582435984, 3.093718337336087), (2.6651670705044617, 1.2895194710945164), (3.127740982780784, 5.924390760732136), (0.39376482519044653, 1.0973395808977064), (1.8117668547890935, 2.0203325402589996), (1.162815174243656, 5.691426270604712), (1.5988532759154377, 0.33286722218657905), (1.908395808769174, 3.1440951930887633), (0.5178303297945664, 3.421588146876333), (2.607168052083064, 2.3107104468253095), (0.7258839139507853, 5.87523297245006), (1.974987086591289, 6.223959331042359), (2.323440224732917, 1.5876659525448513), (2.0540493865509366, 0.4151638980200462), (2.3479117264550635, 4.315955322854676), (0.9380003037572883, 2.4543126598286245)]
y_data = [-1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1]

X_train_data = X_data[:35]
X_test_data = X_data[35:]

y_train_data = y_data[:35]
y_test_data = y_data[35:]

epochs = 30
batch_size = 1

# Training loop
params = { "theta1" : 1.0
         , "phi1": 1.0
         , "theta2" : 1.0
         , "phi2": 1.0
         , "theta3": 1.0
         , "phi3": 1.0
         , "theta4": 1.0
         , "phi4": 1.0
         , "theta5": 1.0
         , "phi5": 1.0
         , "theta6": 1.0 
         , "phi6": 1.0
         }
initial_params = dict(params)

epoch_costs = []

def calculate_ev(countsp):
    if '1' not in countsp: return 1
    elif '0' not in countsp: return -1
    else: return (countsp['0'] - countsp['1'])/1000

def accuracy(parameters):
    accuracy = 0
    for x,y in zip(X_train_data, y_train_data):
        circ = circuit(x[0], x[1], parameters)
        qc_compiled = transpile(circ, simulator)
        result = simulator.run(qc_compiled, shots=1000).result()
        counts = result.get_counts(qc_compiled)
        ev = calculate_ev(counts)
        if ev <= 0 : value = -1
        else : value = 1

        if (value == y):
            accuracy += 1
    return accuracy / len(y_train_data)

def predict(parameters):
    accuracy = 0
    for x,y in zip(X_test_data, y_test_data):
        circ = circuit(x[0], x[1], parameters)
        qc_compiled = transpile(circ, simulator)
        result = simulator.run(qc_compiled, shots=1000).result()
        counts = result.get_counts(qc_compiled)
        ev = calculate_ev(counts)
        if ev <= 0 : value = -1
        else : value = 1
        print(x, y, value)

        if (value == y):
            accuracy += 1
    return accuracy / len(y_test_data)

def make_predictions(parameters, X):
    values = []
    for x in X:
        circ = circuit(x[0], x[1], parameters)
        qc_compiled = transpile(circ, simulator)
        result = simulator.run(qc_compiled, shots=1000).result()
        counts = result.get_counts(qc_compiled)
        ev = calculate_ev(counts)
        values.append(ev)
    return values


print("Starting training...")
for epoch in range(epochs):
    # Create batches
    permutation = np.random.permutation(len(X_train_data))
    print(permutation)
    X_train_perm = np.array(X_train_data)[permutation]
    y_train_perm = np.array(y_train_data)[permutation]
    epoch_cost = 0

    for i in range(0, len(X_train_data), batch_size):
        X_batch = X_train_perm[i : i + batch_size]
        y_batch = y_train_perm[i : i + batch_size]

        delTheta1 = 0
        delPhi1 = 0
        delTheta2 = 0
        delPhi2 = 0
        delTheta3 = 0
        delPhi3 = 0
        delTheta4 = 0
        delPhi4 = 0
        delTheta5 = 0
        delPhi5 = 0
        delTheta6 = 0
        delPhi6 = 0

        current_batch_cost = 0
        current_cost = 0

        for (x,y) in zip(X_batch,y_batch):

            circ = circuit(x[0], x[1], params)
            qc_compiled = transpile(circ, simulator)
            result = simulator.run(qc_compiled, shots=1000).result()
            counts = result.get_counts(qc_compiled)
            ev = calculate_ev(counts)
            current_cost = (ev - y)**2
            multiplier = 2 * (ev - y)
           # print(current_cost)
            current_batch_cost += current_cost

            delTheta1 += multiplier * calculatePartialDerivTheta1(x[0], x[1], params)
            delPhi1 += multiplier * calculatePartialDerivPhi1(x[0], x[1], params)
            delTheta2 += multiplier * calculatePartialDerivTheta2(x[0], x[1], params)
            delPhi2 += multiplier * calculatePartialDerivPhi2(x[0], x[1], params)
            delTheta3 += multiplier * calculatePartialDerivTheta3(x[0], x[1], params)
            delPhi3 += multiplier * calculatePartialDerivPhi3(x[0], x[1], params)
            delTheta4 += multiplier * calculatePartialDerivTheta4(x[0], x[1], params)
            delPhi4 += multiplier * calculatePartialDerivPhi4(x[0], x[1], params)
            delTheta5 += multiplier * calculatePartialDerivTheta5(x[0], x[1], params)
            delPhi5 += multiplier * calculatePartialDerivPhi5(x[0], x[1], params)
            delTheta6 += multiplier * calculatePartialDerivTheta6(x[0], x[1], params)
            delPhi6 += multiplier * calculatePartialDerivPhi6(x[0], x[1], params)

        # update weights after batch
        
        params['theta1'] -= 0.01 * delTheta1 / batch_size
        params['phi1'] -= 0.01 * delPhi1 / batch_size
        params['theta2'] -= 0.01 * delTheta2 / batch_size
        params['phi2'] -= 0.01 * delPhi2 / batch_size
        params['theta3'] -= 0.01 * delTheta3 / batch_size
        params['phi3'] -= 0.01 * delPhi3 / batch_size
        params['theta4'] -= 0.01 * delTheta4 / batch_size
        params['phi4'] -= 0.01 * delPhi4 / batch_size
        params['theta5'] -= 0.01 * delTheta5 / batch_size
        params['phi5'] -= 0.01 * delPhi5 / batch_size
        params['theta6'] -= 0.01 * delTheta6 / batch_size
        params['phi6'] -= 0.01 * delPhi6 / batch_size

      #  print(current_batch_cost)
      #  print(params)
        epoch_cost += current_batch_cost
        acc = accuracy(params)
    print("epoch done with cost")
    print(f"Epoch {epoch+1}/{epochs} - Cost: {epoch_cost:.4f} - Train Accuracy: {acc:.4f}")
    print(epoch_cost)
    epoch_costs.append(epoch_cost)

print(epoch_costs)
print(params)


# parameters after training
trained_params = {'theta1': np.float64(0.3981696875000002), 'phi1': np.float64(1.325726328125), 'theta2': np.float64(1.4431997265625012), 'phi2': np.float64(0.7607752734374985), 'theta3': np.float64(1.2752600000000003), 'phi3': np.float64(0.6592110546874999), 'theta4': np.float64(2.901331484375), 'phi4': np.float64(0.9939541406249996), 'theta5': np.float64(0.9933662109375005), 'phi5': np.float64(0.9997637890624995), 'theta6': np.float64(1.1174424999999988), 'phi6': np.float64(0.9836150000000002)}


accuracy = predict(initial_params)
print(accuracy)
accuracy = predict(trained_params)
print(accuracy)


# make data for decision regions
phi, theta = np.meshgrid(np.linspace(0, np.pi, 20), np.linspace(0, 2* np.pi, 20))
X_grid = [np.array([x, y]) for x, y in zip(phi.flatten(), theta.flatten())]

# Convert spherical to Cartesian coordinates     
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

predictions_grid = make_predictions(trained_params, X_grid)
Z = np.reshape(predictions_grid, phi.shape)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

N = 9
levels = np.linspace(-1, 1, N)

level_colors = plt.cm.get_cmap("RdYlGn", N-1)(np.arange(N-1))
cmap, norm = mcolors.from_levels_and_colors(levels, level_colors)
color_vals = cmap(norm(Z))

ax.plot_surface(x, y, z, facecolors=color_vals, rstride=1, cstride=1)

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # required for older Matplotlib versions
cbar = plt.colorbar(
    sm,
    ax=ax,
    shrink=0.6,
    pad=0.1
)
cbar.set_ticks(levels)
cbar.set_ticklabels([f"{lvl:.2f}" for lvl in levels])
cbar.set_label('Prediction value')
cbar.ax.axhline(0, color='black', linewidth=2)


# Plot the plane: z = a*x + b*y
plane_range = 1.1
xx, yy = np.meshgrid(np.linspace(-plane_range, plane_range, 20),
                     np.linspace(-plane_range, plane_range, 20))
a, b = 2.5, 0
zz = a * xx + b * yy
ax.plot_surface(xx, yy, zz, alpha=0.3, color='red')

# Plot unit sphere (radius = 1)
#u = np.linspace(0, 2*np.pi, 50)
#v = np.linspace(0, np.pi, 50)

#xs = np.outer(np.cos(u), np.sin(v))
#ys = np.outer(np.sin(u), np.sin(v))
#zs = np.outer(np.ones_like(u), np.cos(v))

#ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.3)
ax.set_box_aspect([1, 1, 1])
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_zlim(-1.1, 1.1)
plt.show()

