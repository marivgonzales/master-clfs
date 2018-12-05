import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

predicted_labels = np.load("./laps_nobg_100/predictions.npy")

real_labels = np.load("./laps_nobg_100/real_labels.npy")

measures = np.load("focus_valid_laps.npy")

data = pd.DataFrame({'real':real_labels, 'predicted':predicted_labels, 'lapv':measures[:,0], 'lapm':measures[:,1], 'teng':measures[:,2]})

f, a = plt.subplots(1,20, sharex=False, sharey=True)

for i in range(20):
    box_true = data[data['predicted'] == i][data['real'] == i]['teng']
    box_false = data[data['predicted'] != i][data['real'] == i]['teng']
    a[i].boxplot([box_true, box_false], widths=0.8, labels=['+', '-'])
    a[i].set_title(i)
    
f.subplots_adjust(hspace=0)
plt.show()