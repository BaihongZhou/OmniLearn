import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
# import scienceplots
# plt.style.use(['science','no-latex'])
import csv
# rc('font', **{'family': 'serif'})
# rc('text', usetex=True)
# Load the loss
loss_csv = '/global/homes/b/baihong/workdir/OmniLearn/scripts/logs_pipi_recon/loss.csv'
train_loss = []
val_loss = []
with open(loss_csv, 'r', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        train_loss.append(float(row[0]))
        val_loss.append(float(row[1]))
train_loss = np.array(train_loss)
val_loss = np.array(val_loss)
epochs = np.arange(1, len(train_loss)+1)
plt.plot(epochs, train_loss,label='Training Loss')
plt.plot(epochs, val_loss, linestyle ='-.', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plt.text(0.32, 0.2, r'$\mathcal{L} = - \sum_i y_i \cdot log \hat{y_i}$', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=15)
plt.legend()
plt.savefig('loss.png', dpi=300)
plt.close()
# plt.show()

# # Then plot the accuracy
# score_csv = '/lustre/collider/zhoubaihong/QE_study/Leptonic_ML/Omni_temp/OmniLearn/logs/score.csv'
# score = []
# val_score = []
# with open(score_csv, 'r', newline='') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         score.append(float(row[0]))
#         val_score.append(float(row[1]))
# score = np.array(score)
# val_score = np.array(val_score)
# plt.plot(epochs, score, label='RMSE')
# plt.plot(epochs, val_score, linestyle ='-.', label='Val RMSE')
# plt.xlabel('Epochs')
# plt.ylabel('Score')
# plt.legend()
# plt.savefig('Score.png', dpi=300)
# plt.close()
