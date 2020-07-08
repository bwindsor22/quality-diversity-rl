import imageio
from os import listdir
from os.path import isfile, join

filenames = listdir('saved_images')

temp = []

for i in range(len(filenames)):
    fn = 'Steps-' + str(i) + '.png'
    if (fn) in filenames:
        temp.append(fn)

filenames = temp
#print(filenames)
images = []
for filename in filenames:
    images.append(imageio.imread('saved_images/' + filename))
imageio.mimsave('NSGAII-Lvl4-movie.gif', images)