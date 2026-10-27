# Operating instructions

- Work only inside this independent `04-llm-gateway/` repository. Check `pwd`
  and `git rev-parse --show-toplevel` before changes. Do not inspect, import, or
  modify AI Lab, sibling projects, or their project memory.
- Build one milestone at a time. Update `PLAN.md` with evidence, and keep
  `SPEC.md` and `README.md` accurate about implemented versus planned behavior.
  A mock proxy is the foundation, not completion of this gateway.
- Use Python 3.12+, async network I/O, focused modules, strict schemas, type hints,
  and a project-local virtual environment. Keep dependencies small and locked.
  Verify consequential dependency choices against authoritative documentation.
- Run relevant tests and lint checks before advancing. Test concurrency,
  cancellation, isolation, and failure boundaries as those features arrive.
  Avoid tests that only repeat trivial implementation details.
- Never commit credentials or log authorization headers, prompts, or raw backend
  errors. Require explicit credentials; bind development servers to loopback.
  Route only to operator-configured endpoints and never follow redirects.
- Default to conservative, client-isolated caching. Similarity alone never
  establishes answer equivalence. Include model, settings, context, client, and
  policy version in cache partitions. Never claim unmeasured cache safety.
- The target is a 24 GB macOS laptop; record the actual verification hardware.
  Run model downloads, embedding loads, and inference experiments sequentially.
  Do not install large models until the gateway foundation passes its checks.
- Keep one process and in-memory state initially; no distributed consistency
  claims. Do not add Redis, a database, orchestration, UI, or routing complexity
  without evidence of a need.
- Label mock, real inference, cache, and gateway-overhead measurements separately.
  Keep commands, workload, versions, hardware, and raw benchmark results
  reproducible. Never fabricate performance or real-backend success.
