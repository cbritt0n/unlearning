# Formal recall bounds under adversarial deletions

This document summarizes the claims formalized in
[`compliance/recall_bounds.py`](../compliance/recall_bounds.py) and checked by
[`tests/test_recall_bounds.py`](../tests/test_recall_bounds.py).

## Setup

- Base layer graph \(G=(V,E)\) on active (non-deleted) nodes.
- Soft-delete, unhealed hard-delete, and MN-RU healed hard-delete are three
  policies producing graphs \(G_{soft}\), \(G_{unheal}\), \(G_{heal}\).
- Exact recall@\(k\) is measured against linear-scan ground truth on active nodes.

## Theorems (summary)

| # | Claim | Operational meaning |
|---|--------|---------------------|
| **T1** | Soft-delete preserves geometry for traversal | Soft-delete is the **connectivity upper envelope** |
| **T2** | Unhealed: \(E[R_1] \le 1 - \|U\|/\|V_{active}\|\) | Unreachable mass caps recall@1 |
| **T3** | MN-RU does not increase orphan-component count when capacity allows a spanning rewire | Healed \(\ge\) unhealed on local stars |
| **T4** | Adaptive cut adversaries require rebuild | Bulk/adversarial deletes → call `compact()` |

## What we guarantee in software

1. **Theorem 2 bound** is enforced as a testable inequality for synthetic
   fragmented graphs.
2. **Healed vs unhealed dominance** on star-delete gadgets (reachability and
   recall proxy \(1-\mathrm{unreach}\)).
3. **Soft upper envelope** within tolerance (sampling noise allowed in
   large benchmarks).

## What we do *not* guarantee

- Global expansion restoration under adaptive balanced separators without
  `compact()` / full rebuild.
- Exact equality of healed recall with soft-delete for arbitrary delete sets.

## Empirical validation

```bash
pytest tests/test_recall_bounds.py -v
python tests/benchmark.py --quick   # compares A/B/C policies
```
