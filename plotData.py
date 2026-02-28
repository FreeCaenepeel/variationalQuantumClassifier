import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
num_points = 50

X_data = [(0.4461365358947761, 0.25911862607262476), (3.0926821747170816, 4.600862637413524), (1.7851245408203378, 1.6361694074503343), (1.9124693788066913, 3.2051979100912438), (2.5489585134527712, 3.0832701695158193), (1.1119669626793638, 2.2683261271671786), (2.1282136759406027, 3.4812172965020314), (0.9565170037781868, 3.4572669384429444), (2.337536772610973, 5.599862618257412), (2.127219261964461, 1.64552119970525), (0.679489446821518, 4.097071836929742), (2.981766539352059, 6.023620055331134), (1.1561388367356276, 4.267214069771276), (0.34307846126965363, 1.0628606837933972), (2.2192079444411896, 4.537265875814268), (1.4521912423318428, 3.139838869795846), (1.4076355497355924, 5.769788843993014), (2.1205421869376857, 1.8907250654815846), (2.746406212999902, 3.7665343255289545), (0.5690562924819066, 0.49834092685876924), (0.328835647459121, 2.2766384229726477), (0.12654800697106378, 3.9430708705266913), (0.6273734495018801, 4.809648529309832), (1.0698911716319843, 1.0031065627284872), (1.6175620233012, 1.9573472432391696), (2.8325306325805353, 5.241472091321098), (0.22964292821655669, 5.125986032899986), (2.797745037146566, 4.42351335987515), (1.022114811089987, 5.522285700844636), (2.6973311856359445, 4.478932234052846), (2.991137843914753, 0.525828749709142), (2.061831383099918, 1.3232765967303397), (0.0711998006526504, 3.3710498971416873), (2.088610634604043, 1.5219950457537497), (0.7035843582435984, 3.093718337336087), (2.6651670705044617, 1.2895194710945164), (3.127740982780784, 5.924390760732136), (0.39376482519044653, 1.0973395808977064), (1.8117668547890935, 2.0203325402589996), (1.162815174243656, 5.691426270604712), (1.5988532759154377, 0.33286722218657905), (1.908395808769174, 3.1440951930887633), (0.5178303297945664, 3.421588146876333), (2.607168052083064, 2.3107104468253095), (0.7258839139507853, 5.87523297245006), (1.974987086591289, 6.223959331042359), (2.323440224732917, 1.5876659525448513), (2.0540493865509366, 0.4151638980200462), (2.3479117264550635, 4.315955322854676), (0.9380003037572883, 2.4543126598286245)]

# Convert spherical to Cartesian coordinates     
x = [np.sin(p[0]) * np.cos(p[1]) for p in X_data]
y = [np.sin(p[0]) * np.sin(p[1]) for p in X_data]
z = [np.cos(p[0]) for p in X_data]


# Define plane: z = a*x + b*y
a, b = 2.5, 0

# Classify points
above_mask = [zi - (a*xi + b*yi) > 0 for xi, yi, zi in zip(x, y, z)]
below_mask = [zi - (a*xi + b*yi) <= 0 for xi, yi, zi in zip(x, y, z)]

x_above = [xi for xi, m in zip(x, above_mask) if m]
x_below = [xi for xi, m in zip(x, above_mask) if not m]

y_above = [xi for xi, m in zip(y, above_mask) if m]
y_below = [xi for xi, m in zip(y, above_mask) if not m]

z_above = [xi for xi, m in zip(z, above_mask) if m]
z_below = [xi for xi, m in zip(z, above_mask) if not m]

# Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot points
ax.scatter(x_above, y_above, z_above, color='green', s=10, label='Above plane')
ax.scatter(x_below, y_below, z_below, color='blue', s=10, label='Below plane')

# Plot the plane: z = a*x + b*y
plane_range = 1.1
xx, yy = np.meshgrid(np.linspace(-plane_range, plane_range, 20),
                     np.linspace(-plane_range, plane_range, 20))
zz = a * xx + b * yy
ax.plot_surface(xx, yy, zz, alpha=0.3, color='red')

# Plot unit sphere (radius = 1)
u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)

xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))

ax.plot_wireframe(xs, ys, zs, color='black', alpha=0.3, linewidth=0.5)

# Formatting
ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax.set_zlabel('Z')
ax.set_title(f'{num_points} Samples on Sphere Separated by Plane z = {a}x + {b}y')
ax.legend()

plt.tight_layout()
ax.set_box_aspect([1, 1, 1])
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_zlim(-1.1, 1.1)
plt.show()