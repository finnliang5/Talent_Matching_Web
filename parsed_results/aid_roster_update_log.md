# AID Roster 更新日志

- round1_updated_count: 3
- round2_updated_count: 10
- updated_total_after_round2: 13
- parsed_files_current_count: 208
- matched_total_current_count: 196
- file_unmatched_after_round2: 11
- roster_unmatched_after_round2: 94

## Second Round Updated People

| file_prefix | matched_roster_eid | file | rule | 城市(old -> new) | 级别(old -> new) |
|---|---|---|---|---|---|
| fanen.liu | vance.fanen.liu | Unmatched_0005_parsed.json | subset_tokens | Dalian -> Dalian | Level 9 -> Level 9 |
| hongwei.liu | hongwei.c.liu | Unmatched_0073_parsed.json | subset_tokens | Dalian -> Dalian | Level 11 -> Level 11 |
| linlin.tang | linlin.a.tang | Unmatched_0015_parsed.json | subset_tokens | Dalian -> Dalian | Level 10 -> Level 10 |
| ma zheng | zheng.a.ma | Unmatched_0019_parsed.json | subset_tokens | Dalian -> Dalian | Level 7 -> Level 7 |
| meng wei | wei.a.meng | Unmatched_0020_parsed.json | subset_tokens | Shanghai -> Shanghai | Level 8 -> Level 8 |
| mengyao.liu | mengyao.a.liu | Unmatched_0018_parsed.json | subset_tokens | Dalian -> Dalian | Level 9 -> Level 9 |
| minghao.wang | minghao.b.wang | Unmatched_0113_parsed.json | subset_tokens | Dalian -> Dalian | Level 9 -> Level 9 |
| song xianpeng | xianpeng.song | Unmatched_0029_parsed.json | subset_tokens | Dalian -> Dalian | Level 11 -> Level 11 |
| xiaoyun.wang | xiaoyun.a.wang | Unmatched_0033_parsed.json | subset_tokens | Dalian -> Dalian | Level 10 -> Level 10 |
| Unmatched_0181_parsed.json | yao.wu | Unmatched_0035_parsed.json | norm_exact | Dalian -> Dalian | Level 9 -> Level 9 |

## Still Unmatched After Round 2

- ericzhang (Unmatched_0003_parsed.json)
- jialing.d.zhang (Unmatched_0083_parsed.json)
- jian.peng (Unmatched_0085_parsed.json)
- kerry.sun (Unmatched_0096_parsed.json)
- leon.liang.he (Unmatched_0102_parsed.json)
- qihui.liu (Unmatched_0127_parsed.json)
- xiaojiao.zhang (Unmatched_0165_parsed.json)
- xinyan.yu (Unmatched_0170_parsed.json)
- xinyue.j.wang (Unmatched_0172_parsed.json)
- xuhongyan (Unmatched_0179_parsed.json)
- yutong.zhu (Unmatched_0036_parsed.json)

## 第三轮建议（人工确认）

- generated_at: 2026-07-23T09:19:29
- unmatched_count: 11
- 评分规则: score = 0.55*token_overlap + 0.35*sequence_ratio + 0.10*first_token_match

| eid_from_filename | file | confidence | candidate_1 | score_1 | candidate_2 | score_2 | candidate_3 | score_3 |
|---|---|---|---|---:|---|---:|---|---:|
| ericzhang | Unmatched_0003_parsed.json | low | eric.y.zhang | 0.3316 | yue.ac.zhang | 0.2579 | jiaxin.c.zhang | 0.2333 |
| jialing.d.zhang | Unmatched_0083_parsed.json | low | jiaxin.c.zhang | 0.4633 | jing.z.zhang | 0.4572 | qianbing.zhang | 0.4526 |
| jian.peng | Unmatched_0085_parsed.json | low | jian.b.zheng | 0.6472 | peng.a.yu | 0.4617 | jialu.zheng | 0.2333 |
| kerry.sun | Unmatched_0096_parsed.json | low | shoufeng.sun | 0.4224 | - | - | - | - |
| leon.liang.he | Unmatched_0102_parsed.json | low | ken.jiajian.liang | 0.3718 | liang.q.wang | 0.3500 | - | - |
| qihui.liu | Unmatched_0127_parsed.json | low | yueqi.liu | 0.4938 | shi.a.liu | 0.4617 | max.xinguang.liu | 0.4341 |
| xiaojiao.zhang | Unmatched_0165_parsed.json | low | jiaxin.c.zhang | 0.4990 | qianbing.zhang | 0.4904 | jing.z.zhang | 0.4880 |
| xinyan.yu | Unmatched_0170_parsed.json | low | chunyang.yu | 0.5083 | hai.a.yu | 0.4750 | peng.a.yu | 0.4617 |
| xinyue.j.wang | Unmatched_0172_parsed.json | low | yue.y.wang | 0.4412 | xinchen.wang | 0.4379 | ximin.wang | 0.4283 |
| xuhongyan | Unmatched_0179_parsed.json | low | zhengyuan.hu | 0.2100 | - | - | - | - |
| yutong.zhu | Unmatched_0036_parsed.json | low | ying.c.zhu | 0.5221 | fangyi.zhu | 0.4694 | dengke.zhu | 0.4694 |
