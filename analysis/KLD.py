import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np

F_KLD_data = {
    'HC':[8.24, 6.55, 5.29, 3.11],
    'AC':[7.56, 6.11, 4.75, 2.44],
    'CMS':[0.85, 0.94, 1.1, 1.12],
    'ADG': 1.45,
}

M_KLD_data = {
    'HC': [7.45, 5.34, 5.01, 2.88],
    'AC': [6.54, 5.32, 4.11, 2.00],
    'CMS': [0.71, 0.81, 0.89, 0.99],
    'ADG': 1.33,
}

F_bpw = {
    'HC': [1.00, 1.78, 2.95, 3.89],
    "AC": [1.31, 2.05, 2.97, 4.21],
    'CMS': [1.01, 2.06, 3.02, 4.12],
    'ADG': 4.65,
}

M_bpw = {
    'HC': [1.00, 1.78, 2.95, 3.89],
    "AC": [1.31, 2.05, 2.97, 4.21],
    'CMS': [1.03, 2.11, 3.08, 4.15],
    'ADG': 4.81,
}

y_majorLocator = MultipleLocator(1.)
y_majorFormatter = FormatStrFormatter('%5.1f')

font = {
    # 'family': "New Century Schoolbook",
    'weight': 'normal',
    'size': 12,
}

fig = plt.figure(figsize=(8.5, 5))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Embedding capacity (bpw)', font)
ax.set_ylabel('KLD value', font)


ax.yaxis.set_major_locator(y_majorLocator)
ax.yaxis.set_major_formatter(y_majorFormatter)


plt.ylim(0, 9)

plt.plot(F_bpw['HC'], F_KLD_data['HC'], color='r', marker='o', label="Filckr30k_HC")
plt.plot(M_bpw['HC'], M_KLD_data['HC'], color='r', linestyle='-.', marker='*', label="MSMCOCO_HC")
plt.plot(F_bpw['AC'], F_KLD_data['AC'], color='b', marker='o', label="Filckr30k_AC")
plt.plot(M_bpw['AC'], M_KLD_data['AC'], color='b', linestyle='-.', marker='*', label="MSMCOCO_AC")
plt.plot(F_bpw['CMS'], F_KLD_data['CMS'], color='g', marker='o', label="Filckr30k_Ours")
plt.plot(M_bpw['CMS'], M_KLD_data['CMS'], color='g', linestyle='-.', marker='*', label="MSMCOCO_Ours")
plt.scatter(F_bpw['ADG'], F_KLD_data['ADG'], color='violet', marker='*', s=100, label='Filckr30k_ADG')
plt.scatter(M_bpw['ADG'], M_KLD_data['ADG'], color='purple', marker='*', s=100, label='MSCOCO_ADG')
x = np.arange(1,6)
plt.xticks(x)
plt.legend(loc=0, ncol=2)
plt.savefig('KLD.pdf')

