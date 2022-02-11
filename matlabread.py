import scipy.io
import pandas as pd
import subprocess
import os

mat = scipy.io.loadmat('imagelabels.mat')
print(mat['labels'][0])

for i in range(1,103):
    process = subprocess.run(['mkdir', "flowerclasseddataset/" + str(i)])

directory = 'jpg/'

j = 0
for filename in sorted(os.listdir(directory)):
    f = os.path.join(directory, filename)
    # checking if it is a file
    print("Old directory")
    print(f)
    print("New directory")
    print("flowerclasseddataset/" + str(mat['labels'][0][j]) + "/" + filename)
    if os.path.isfile(f):
        os.rename(f, "flowerclasseddataset/" + str(mat['labels'][0][j]) + "/" + filename)
    j += 1

