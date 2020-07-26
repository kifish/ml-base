ref = 'dev.txt'
pred = 'dev_pred.txt'

def get_entity(path):
    res = [[], []]
    with open(path, encoding='utf-8') as f:
        sentence_lst = f.read().split('\n\n')[:-1]
    for x in sentence_lst:
        tmp0, tmp1 = [], []
        for token, tag in map(lambda s: s.split()[1:], x.split('\n')):
            tmp0.append(token)
            tmp1.append(tag)
        res[0].append(tmp0)
        res[1].append(tmp1)
    return res
ref_entity = get_entity(ref)
pred_entity = get_entity(pred)

cnt = 0
bad = []
for i in range(len(ref_entity[1])):
    if ref_entity[1][i] == pred_entity[1][i]:
        cnt += 1
    else:
        bad.append(i)
print('acc: %.2f'%(cnt/len(ref_entity[1])*100))

with open('eval_res.txt', 'w', encoding='utf-8') as f:
    for i in bad:
        f.write(' '.join(ref_entity[0][i]) + '\n')
        f.write(' '.join(map(lambda x:x[0], ref_entity[1][i])) + '\n')
        f.write(' '.join(map(lambda x:x[0], pred_entity[1][i])) + '\n\n')

