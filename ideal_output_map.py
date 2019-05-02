import numpy as np
import matplotlib.pyplot as plt

def dist(x1, y1, p2):
    return np.sqrt((x1-p2[0])**2 + (y1-p2[1])**2)

def getVal(i,j):
    x,y = float(i)/res, float(j)/res
    points = { 0:[0,0], 1:[0,1], 2:[1,0], 3:[1,1] }    
    dists = np.array([dist(x,y,points[0]), dist(x,y,points[1]), dist(x,y,points[2]), dist(x,y,points[3])])
    minDist, iDist = min(dists), np.argmin(dists)

    if minDist >= 0.5:
        minDist = 0.5

    if iDist == 1 or iDist == 2:
        return 1 - minDist
    else:
        return 0 + minDist


res = 49
output = [None] * (res+1)

for i in range(res+1):
    output[i] = [None] * (res+1)
    for j in range(res+1):
        x = np.array([float(i)/res, float(j)/res]).reshape(1, -1)
        output[i][j] = getVal(i,j)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
cax = plt.imshow(np.array(output), interpolation='nearest', vmin=0, vmax=1)
cbar = fig.colorbar(cax, ticks=[0, 1])
cbar.ax.tick_params(labelsize=15)
plt.set_cmap('gray')
plt.axis('off')

table = {'0, 0':(-2, -1),
         '0, 1':(-2, res+2),
         '1, 0':(res-2, -1),
         '1, 1':(res-2, res+2)}
for text, corner in table.items():
    ax.annotate(text, xy=corner, size=15, annotation_clip=False)

plt.show()
