# Retrieval Evaluation Metrics

Ranking (IR) metrics over the top-`k` of a ranked list. `@k` = computed over the
first `k` results.

**Hit Rate@k** — did any relevant doc land in the top `k`?

```
hit(q)    = 1 if top-k results for query q contain >= 1 relevant doc, else 0
HitRate@k = (1 / N) * sum_{q=1..N} hit(q)          (N = number of queries)
```

**Recall@k** — share of a query's relevant docs found in the top `k` (averaged).

```
Recall@k = (# relevant docs in top k) / (# relevant docs for the query)
```

**NDCG@k** — rank quality; relevant docs score more when ranked higher.

```
DCG@k  = sum_{i=1..k} rel_i / log2(i + 1)
NDCG@k = DCG@k / IDCG@k          (IDCG = DCG of the ideal ranking)
```

**MAP@k** — mean over queries of Average Precision; rewards ranking hits early.

```
Precision@i      = (# relevant docs in ranks 1..i) / i
AveragePrecision = mean of Precision@i over ranks i<=k holding a relevant doc
MAP@k            = mean of AveragePrecision over all queries
```
