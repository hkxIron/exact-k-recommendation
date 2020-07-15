user_ids, item_ids = set(), set()

# userid item_ids item_ids
# 1	202,185,222,104,68,34,151,49,157,136	259,4,262,135,136,140,158,143,145,146,149,151,24,260,155,29,30,159,34,164,241,157,49,222,184,185,186,63,66,68,69,202,162,211,215,90,94,226,227,229,103,104,107,112,232,116,249,123,252,125
with open('rerank_data_10c.txt', 'r') as fin:
    for line in fin:
        line = line.strip()
        strs = line.split('\t')
        user_ids.add(int(strs[0]))

        for x in strs[1].split(','):
            item_ids.add(int(x))
        for x in strs[2].split(','):
            item_ids.add(int(x))

print('user_ids len: ', len(user_ids))
print('item_ids len: ', len(item_ids))

with open('user_ids.txt', 'w') as fout:
    user_ids_list = sorted(list(user_ids))
    for x in user_ids_list:
        fout.write(str(x) + '\n')


with open('item_ids.txt', 'w') as fout:
    item_ids_list = sorted(list(item_ids))
    for x in item_ids_list:
        fout.write(str(x) + '\n')