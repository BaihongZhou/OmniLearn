import matplotlib.pyplot as plt
import numpy as np

import csv

loss_path = '/lustre/collider/zhoubaihong/QE_study/Leptonic_ML/Omni_temp/OmniLearn/scripts/logs/loss.csv'

loss = []
with open(loss_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        loss.append(row)[0]
        
plt.plot(np.arange(len(loss)), loss)
plt.savefig('loss.png')
