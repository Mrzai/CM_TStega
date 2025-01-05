import json

file_path = '/data/I2T_data/MSCOCO/dataset_coco.json'

f = open(file_path, encoding='utf-8')

result = json.load(f)

print(type(result['images']))
num = len(result['images'])
print(num)
for i in range(num):
    if result['images'][i]['split'] == 'test':
        print(result['images'][i])
        input()
