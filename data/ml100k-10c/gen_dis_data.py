import random


with open('rerank_data_10c_train.txt', 'r') as fin:
    with open('dis_data_10c_train.txt', 'w') as fout:
        for line in fin:
            strs = line.strip().split('\t')
            user = strs[0]
            card_pos = strs[1].split(',') # 第一个item_list为最终选中的item
            item_cand = strs[2].split(',') # 第二个item_list为候选的item
            item_pos = card_pos[0] # 选取第一个item

            item_cand_neg = set(item_cand)
            item_cand_neg.remove(item_pos)
            item_cand_neg = list(item_cand_neg)

            card_neg = []
            if random.random() < 0.3: # 0.3的概率用正样本做一下扰动
                while True:
                    card_neg.append(item_pos)
                    card_neg.extend(random.sample(item_cand_neg, k=9)) # 从item_cand_neg中选取9个item
                    if ','.join(sorted(card_neg)) != ','.join(sorted(card_pos)):
                        break
                    else:
                        card_neg = []
            else:
                card_neg.extend(random.sample(item_cand_neg, k=10))
            # 原来的一条样本现在拆分成2条不同的样本:1条正样本,1条负样本
            fout.write(user + '\t' + ','.join(card_pos) + '\t' + '1' + '\n')
            fout.write(user + '\t' + ','.join(card_neg) + '\t' + '0' + '\n') # 对负样本偶尔进行扰动
            fout.flush()