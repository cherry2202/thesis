# import math

# def get_topk_results(predictions, scores, targets, k, all_items=None):
#     results = []
#     B = len(targets)
#     # predictions = [_.split("Response:")[-1] for _ in predictions]
#     predictions = [_.strip().replace(" ","") for _ in predictions]
#     # print(predictions)##################
#     if all_items is not None:
#         for i, seq in enumerate(predictions):
#             if seq not in all_items:
#                 scores[i] = -1000

#     # print(scores)
#     for b in range(B):
#         batch_seqs = predictions[b * k: (b + 1) * k]
#         batch_scores = scores[b * k: (b + 1) * k]

#         pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
#         # print(pairs)
#         sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
#         target_item = targets[b]
#         one_results = []
#         for sorted_pred in sorted_pairs:
#             if sorted_pred[0] == target_item:
#                 one_results.append(1)
#             else:
#                 one_results.append(0)

#         results.append(one_results)

#     return results
# def get_topk_ranking_results(predictions, targets, k, all_items=None):
#     results = []
#     B = len(targets)

#     for b in range(B):
#         batch_seqs = predictions[b]
#         target_item = targets[b]
#         one_results = []
#         for sorted_pred in predictions:
#             if sorted_pred == target_item:
#                 one_results.append(1)
#             else:
#                 one_results.append(0)

#         results.append(one_results)

#     return results
# def get_metrics_results(topk_results, metrics):
#     res = {}
#     for m in metrics:
#         if m.lower().startswith("hit"):
#             k = int(m.split("@")[1])
#             res[m] = hit_k(topk_results, k)
#         elif m.lower().startswith("ndcg"):
#             k = int(m.split("@")[1])
#             res[m] = ndcg_k(topk_results, k)
#         else:
#             raise NotImplementedError

#     return res


# def ndcg_k(topk_results, k):
#     """
#     Since we apply leave-one-out, each user only have one ground truth item, so the idcg would be 1.0
#     """
#     ndcg = 0.0
#     for row in topk_results:
#         res = row[:k]
#         one_ndcg = 0.0
#         for i in range(len(res)):
#             one_ndcg += res[i] / math.log(i + 2, 2)
#         ndcg += one_ndcg
#     return ndcg


# def hit_k(topk_results, k):
#     hit = 0.0
#     for row in topk_results:
#         res = row[:k]
#         if sum(res) > 0:
#             hit += 1
#     return hit

import math


def get_topk_results(predictions, scores, targets, k, all_items=None):
    """
    predictions: List[str], length = B * k
    scores: List[float], length = B * k
    targets: List[str], length = B
    return: List[List[int]]  shape = [B, k]
    """
    results = []
    B = len(targets)

    # 统一清洗预测字符串
    predictions = [p.strip().replace(" ", "") for p in predictions]

    # 非法 item 直接置极小分（保持 float 稳定）
    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1e9

    for b in range(B):
        batch_seqs = predictions[b * k:(b + 1) * k]
        batch_scores = scores[b * k:(b + 1) * k]

        # 加入 beam_id 作为确定性 tie-breaker
        # (score 相同 → beam_id 小的排前)
        pairs = [
            (beam_id, seq, score)
            for beam_id, (seq, score) in enumerate(zip(batch_seqs, batch_scores))
        ]

        pairs.sort(key=lambda x: (x[2], -x[0]), reverse=True)

        target_item = targets[b]
        one_results = [1 if seq == target_item else 0 for _, seq, _ in pairs]

        results.append(one_results)

    return results


def get_metrics_results(topk_results, metrics):
    """
    返回的是“分子和”，不是均值
    可直接用于 DDP all_reduce
    """
    res = {}
    for m in metrics:
        m_lower = m.lower()
        if m_lower.startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m_lower.startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError(f"Unknown metric: {m}")
    return res


def ndcg_k(topk_results, k):
    """
    Leave-one-out 场景：
    - 每个 sample 只有一个正样本
    - IDCG = 1
    返回 DCG 的“分子和”，不是均值
    """
    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        for i, v in enumerate(res):
            if v > 0:
                ndcg += 1.0 / math.log2(i + 2)
                break
    return ndcg


def hit_k(topk_results, k):
    """
    Hit@K：命中计数
    """
    hit = 0.0
    for row in topk_results:
        if sum(row[:k]) > 0:
            hit += 1.0
    return hit
