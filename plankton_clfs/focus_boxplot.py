import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

measures = np.load("focus_valid_ndsb_nounk.npy")
predictions= np.load("./ndsb_dataset_nounk/complete_predictions_079_valid.npy")

#grouped
predicted_labels = np.load("./ndsb_dataset_nounk/labels_predicted_079_valid_grouped.npy")
real_labels = np.load("./ndsb_dataset_nounk/labels_real_079_valid_grouped.npy")
scores = np.load("./ndsb_dataset_nounk/max_scores_predicted_079_valid_grouped.npy")
measures = np.log(measures)

data = pd.DataFrame({'max': scores, 'real':real_labels, 'predicted':predicted_labels, 'lapv':measures[:,0], 'lapm':measures[:,1], 'teng':measures[:,2]})
#scores = pd.DataFrame(predictions)
#data['max'] = scores.apply(max, axis=1)


var = 'max'
colors = ['blue', 'red']


#ploting boxplots
f, a = plt.subplots(1,33, sharex=False, sharey=True)

for i in range(33):
	j = i
	box_true = data[data['predicted'] == j][data['real'] == j][var]
	box_false = data[data['predicted'] != j][data['real'] == j][var]
	print("loading", j)
	bplot = a[i].boxplot([box_true, box_false], widths=0.8, labels=['+', '-'],patch_artist=True)
	a[i].set_title(j)

	for patch, color in zip(bplot['boxes'], colors):
		patch.set_facecolor(color)
        
f.subplots_adjust(hspace=0)
plt.show()

"""
f, a = plt.subplots(1,40, sharex=False, sharey=True)

for i in range(40):
	j = i
	box_true = data[data['predicted'] == j][data['real'] == j][var]
	box_false = data[data['predicted'] != j][data['real'] == j][var]
	print("loading", j)
	bplot = a[i].boxplot([box_true, box_false], widths=0.8, labels=['+', '-'],patch_artist=True)
	a[i].set_title(j)

	for patch, color in zip(bplot['boxes'], colors):
		patch.set_facecolor(color)
        
f.subplots_adjust(hspace=0)
plt.show()

f, a = plt.subplots(1,40, sharex=False, sharey=True)

for i in range(40):
	j=i+40
	box_true = data[data['predicted'] == j][data['real'] == j][var]
	box_false = data[data['predicted'] != j][data['real'] == j][var]
	print("loading", j)
	if j != 63:
		bplot = a[i].boxplot([box_true, box_false], widths=0.8, labels=['+', '-'],patch_artist=True)
	a[i].set_title(j)

	for patch, color in zip(bplot['boxes'], colors):
		patch.set_facecolor(color)
        
f.subplots_adjust(hspace=0)
plt.show()

f, a = plt.subplots(1,38, sharex=False, sharey=True)

for i in range(38):
	j=i+80
	box_true = data[data['predicted'] == j][data['real'] == j][var]
	box_false = data[data['predicted'] != j][data['real'] == j][var]
	print("loading", j)
	bplot = a[i].boxplot([box_true, box_false], widths=0.8, labels=['+', '-'],patch_artist=True)
	a[i].set_title(j)

	for patch, color in zip(bplot['boxes'], colors):
		patch.set_facecolor(color)
    
f.subplots_adjust(hspace=0)
plt.show()
"""
