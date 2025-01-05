import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np


ADG_data = {
    'LS-CNN': [0.5745, 0.5620],
    'R-BI-C': [0.5775, 0.5685],
    'BERT-FT':[0.5880, 0.5795],
}


CMS_data = {
    'LS-CNN': [0.5520, 0.5415],
    'R-BI-C': [0.5600, 0.5485],
    'BERT-FT': [0.5810, 0.5665],
}

x = ['LS-CNN', 'R-BI-C', 'BERT-FT']


y_majorLocator = MultipleLocator(0.05)
y_majorFormatter = FormatStrFormatter('%5.2f')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()-0.19, 1.01 * height, '%s' %float(height))

font = {
    # 'family': "New Century Schoolbook",
    'weight': 'normal',
    'size': 12,
}


fig = plt.figure(figsize=(8.5, 5))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel("Steganalysis tools", font)
ax.set_ylabel("Steganalysis accuracy", font)
ax.yaxis.set_major_locator(y_majorLocator)
ax.yaxis.set_major_formatter(y_majorFormatter)

plt.ylim(0.4, 0.65)
y_1 = [value[0] for _, value in ADG_data.items()]
y_2 = [value[1] for _, value in ADG_data.items()]
y_3 = [value[0] for _, value in CMS_data.items()]
y_4 = [value[1] for _, value in CMS_data.items()]

x_axis = np.arange(len(x))
a = plt.bar(x_axis, y_1, width=0.2, color='b', edgecolor='k', label='ADG_Flickr30K')
b = plt.bar(x_axis+0.2, y_3, width=0.2, color='royalblue',  edgecolor='k', label='Our_Flickr30K')
c = plt.bar(x_axis+0.4, y_2, width=0.2, color='orangered', edgecolor='k', label='ADG_MSCOCO')
d = plt.bar(x_axis+0.6, y_4, width=0.2, color='salmon',   edgecolor='k', label='Our_MSCOCO')

autolabel(a)
autolabel(b)
autolabel(c)
autolabel(d)
plt.xticks(x_axis+0.3,  x)
plt.legend(ncol=2)
plt.savefig('ADG.pdf')