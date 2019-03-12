import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

predicted_labels = np.load("./ndsb_dataset_nounk/predicted_labels_079_valid.npy")
real_labels = np.load("./ndsb_dataset_nounk/real_labels_nounk.npy")
measures = np.load("focus_valid_ndsb_nounk.npy")

data = pd.DataFrame({'real':real_labels, 'predicted':predicted_labels, 'lapv':measures[:,0], 'lapm':measures[:,1], 'teng':measures[:,2]})



f, a = plt.subplots(1,80, sharex=False, sharey=True)

for i in range(80):
	box_true = data[data['predicted'] == i][data['real'] == i]['teng']
	box_false = data[data['predicted'] != i][data['real'] == i]['teng']
	a[i].boxplot([box_true, box_false], widths=0.8, labels=['+', '-'])
	a[i].set_title(i)
    
f.subplots_adjust(hspace=0)
plt.show()
"""
f1, a1 = plt.subplots(1,40, sharex=False, sharey=True)

j=0
for i in range(40, 80):
	box_true = data[data['predicted'] == i][data['real'] == i]['teng']
	box_false = data[data['predicted'] != i][data['real'] == i]['teng']
	a1[j].boxplot([box_true, box_false], widths=0.8, labels=['+', '-'])
	a1[j].set_title(i)
	j = j+1
    
f1.subplots_adjust(hspace=0)
plt.show()

f2, a2 = plt.subplots(1,38, sharex=False, sharey=True)

for i in range(80, 119):
    box_true = data[data['predicted'] == i][data['real'] == i]['teng']
    box_false = data[data['predicted'] != i][data['real'] == i]['teng']
    a2[i].boxplot([box_true, box_false], widths=0.8, labels=['+', '-'])
    a2[i].set_title(i)
    
f2.subplots_adjust(hspace=0)
plt.show()
"""