import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

predicted_labels = np.load("./laps_nobg_100/predictions.npy")

real_labels = np.load("./laps_nobg_100/real_labels.npy")

measures = np.load("focus_valid_laps.npy")

data = pd.DataFrame({'real':real_labels, 'predicted':predicted_labels, 'lapv':measures[:,0], 'lapm':measures[:,1], 'teng':measures[:,2]})

f, a = plt.subplots(1,40, sharex=False, sharey=True)

for i in range(40):
    box_true = data[data['predicted'] == i][data['real'] == i]['teng'][:40]
    box_false = data[data['predicted'] != i][data['real'] == i]['teng'][:40]
    a[i].boxplot([box_true, box_false], widths=0.8, labels=['+', '-'])
    a[i].set_title(i)
    
f.subplots_adjust(hspace=0)
plt.show()

f1, a1 = plt.subplots(1,40, sharex=False, sharey=True)

for i in range(20):
    box_true = data[data['predicted'] == i][data['real'] == i]['teng'][40:80]
    box_false = data[data['predicted'] != i][data['real'] == i]['teng'][40:80]
    a1[i].boxplot([box_true, box_false], widths=0.8, labels=['+', '-'])
    a1[i].set_title(i+40)
    
f1.subplots_adjust(hspace=0)
plt.show()

f2, a2 = plt.subplots(1,38, sharex=False, sharey=True)

for i in range(20):
    box_true = data[data['predicted'] == i][data['real'] == i]['teng'][80:118]
    box_false = data[data['predicted'] != i][data['real'] == i]['teng'][80:118]
    a2[i].boxplot([box_true, box_false], widths=0.8, labels=['+', '-'])
    a2[i].set_title(i+80)
    
f2.subplots_adjust(hspace=0)
plt.show()

