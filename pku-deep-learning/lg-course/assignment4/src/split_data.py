import numpy as np

data = []
with open('../dataset/raw/cmn.txt','r',encoding='utf8') as f:
    for line in f.readlines():
        data.append(line.strip())

n_samples = len(data)
print('loaded {} samples'.format(n_samples))

val_ratio = 0.1
test_ratio = 0.1

num1 = int(n_samples * (val_ratio + test_ratio))
num2 = int(n_samples * test_ratio)

train_data = []
val_data = []
test_data = []
idxs = np.random.choice(n_samples,n_samples,replace=False)
train_idxs = idxs[:-num1]
val_idxs = idxs[-num1:-num2]
test_idxs = idxs[-num2:]

for idx in train_idxs:
    train_data.append(data[idx])
for idx in val_idxs:
    val_data.append(data[idx])
for idx in test_idxs:
    test_data.append(data[idx])

print('the number of training samples : {}'.format(len(train_data)))
print('the number of validation samples : {}'.format(len(val_data)))
print('the number of test samples : {}'.format(len(test_data)))

def save_data(data,save_path_src,save_path_target):
    def split_han(s):
        r = ''
        for ch in s:
            r += ch
            r += ' '
        r = r[:-1]
        return r
    source_data = []
    target_data = []
    for sample in data:
        source,target = sample.split('\t')
        source = source[:-1] + ' ' + source[-1]
        target = split_han(target)
        source_data.append(source)
        target_data.append(target)
    with open(save_path_src,'w',encoding='utf8') as f:
        for s in source_data:
            f.write(s + '\n')
    with open(save_path_target,'w',encoding='utf8') as f:
        for s in target_data:
            f.write(s + '\n')

save_path_src = '../dataset/train.eng.txt'
save_path_target = '../dataset/train.chn.txt'
save_data(train_data,save_path_src,save_path_target)

save_path_src = '../dataset/val.eng.txt'
save_path_target = '../dataset/val.chn.txt'
save_data(val_data,save_path_src,save_path_target)

save_path_src = '../dataset/test.eng.txt'
save_path_target = '../dataset/test.chn.txt'
save_data(test_data,save_path_src,save_path_target)




