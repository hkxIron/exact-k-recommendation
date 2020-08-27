### Exact-K Recommendation via Maximal Clique Optimization

#### 数据说明
rerank_data_10c.txt :
格式:userid item_ids item_ids, 第一个item_ids为最终选出的item_id,第二个item_ids为候选item_id
card item_id有10个, 候选item_id有50个
1	202,185,222,104,68,34,151,49,157,136	259,4,262,135,136,140,158,143,145,146,149,151,24,260,155,29,30,159,34,164,241,157,49,222,184,185,186,63,66,68,69,202,162,211,215,90,94,226,227,229,103,104,107,112,232,116,249,123,252,125

rerank_data_10c_train.txt :
文件数据格式:user_id card_item_ids candidate_item_ids

rerank_data_10c_test.txt :
文件数据格式:user_id card_item_ids candidate_item_ids

dis_data_10c_test.txt:
格式:userid card_item_ids label, label代表userid对该组item_ids而形的卡片是否感兴趣
666	506,186,924,129,114,258,143,684,974,709	1


使用ml100k-10c的数据, python2运行:
mkdir logdir
python train.py

Please cite:
```
@inproceedings{gong2019exact,
  title={Exact-k recommendation via maximal clique optimization},
  author={Gong, Yu and Zhu, Yu and Duan, Lu and Liu, Qingwen and Guan, Ziyu and Sun, Fei and Ou, Wenwu and Zhu, Kenny Q},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={617--626},
  year={2019}
}
```
