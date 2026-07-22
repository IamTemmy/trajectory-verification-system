# Repository working agreement

## Engineering priorities

1. Correctness and traceability before interface polish.
2. Keep the verification core independent from dataset SDKs.
3. Every metric and requirement evaluator needs deterministic tests.
4. Do not claim to evaluate the production Waymo Driver.
5. Record assumptions, units, coordinate frames, and threshold rationale.

## Verification commands

```bash
python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m trajectory_verification.cli examples/following_scenario.json examples/requirements.json
```

## Code style

- Python 3.11+.
- Type hints on public APIs.
- Standard library core; justify new runtime dependencies.
- Small, reviewable commits with tests passing.
