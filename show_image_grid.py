import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid

def generate_link(image_index=0):
    return "resized2/00000"[:-len(str(image_index))] + str(image_index) +".jpg"

images = []
x=4
y=10
x_start = 0
for i in range(x_start,x_start+x*y):
    images.append(mpimg.imread(generate_link(i)))


fig = plt.figure(figsize=(10, 10))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(x, y),  # creates 2x2 grid of axes
                 axes_pad=0.001,  # pad between axes in inch.
                 )

for ax, im in zip(grid, images):
    # Iterating over the grid returns the Axes.
    ax.imshow(im)

plt.show()