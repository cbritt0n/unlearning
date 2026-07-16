## Summary

<!-- What does this PR change and why? -->

## Type of change

- [ ] Bug fix
- [ ] Feature / adapter
- [ ] Docs / community
- [ ] Benchmarks / eval
- [ ] Refactor / CI
- [ ] Security-sensitive (delete path, residual, auth)

## Residual / threat impact

<!-- Required if you checked Security-sensitive. Else "N/A". -->

- Surfaces wiped / not wiped:
- Fail-closed behavior unchanged? (yes/no)

## How tested

- [ ] `pytest tests/ -v --ignore=tests/benchmark.py`
- [ ] Optional: adapter tests / example scripts
- [ ] Optional: `python tests/benchmark.py --profile quick --backend hnswlib`

## Checklist

- [ ] Docs updated if user-facing
- [ ] No secrets or live `data/` / `benchmark_results/` artifacts committed
- [ ] Non-claims preserved (no “full GDPR” language)
