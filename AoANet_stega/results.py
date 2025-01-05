
from fast_bleu import SelfBLEU
import numpy as np



file_path = './log/gen_cap.txt'


with open(file_path, 'r') as f:
    lines = f.readlines()

references = []
for l in lines:
    line = l.strip().split(' ')[1:]
    references.append(line)


weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}

self_bleu = SelfBLEU(references, weights)

score = self_bleu.get_score()

print('self_BLEU2: ', np.mean(score['bigram']))

print('self_BLEU3: ', np.mean(score['trigram']))
