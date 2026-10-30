# M07 preregistered semantic generalization protocol

Baseline: `7850d8a6452a28ceed877488e6dae2a9a2ac3ebb`.
The final holdout is frozen before scoring or implementation of the proposed
unseen-wording gate. A delegated author created it without viewing a development
corpus or gate implementation. The author could inspect prior M05/M06 data and
knows the proposed domain: France/Japan national capitals with city-name-only
answers. This is internal author separation, not external independent review,
double blindness, random sampling, or evidence of model-training novelty.

- Keep the selected pinned L3 encoder, production threshold **0.75**, generation
  settings and source questions fixed. Do not tune thresholds, grammar, response
  validation or approvals from final-case outcomes. Use a separately identified
  development dataset for changes; freeze the candidate implementation and its
  source hashes before first reading/scoring the final cases.
- The final JSON contains 72 complete wire cases: 24 equivalent city-only requests
  and 48 hard/policy negatives. Both countries share 12 paired positive linguistic
  patterns; report this correlation rather than claiming 24 independent forms.
  Some valid paraphrases intentionally use less conventional formulations to
  measure conservative abstention. All final user wording is absent from the
  prior approval registry and M05/M06 evaluations. Context/guard variants reuse
  new positive wording with different real wire inputs.
- Seed only the original canonical France/Japan requests for `client-a`, using
  the same resolved model/revisions/settings as each ordinary case. Use fresh
  cache state per case (or restore only source entries) so a prior final query
  cannot become a source. Do not register any final wording as approved. The
  `client-b` case must remain cold and cannot consume `client-a` responses.
- Compare baseline reviewed-phrase mode with the frozen proposed mode. Pass each
  JSON `request`, `headers` and `client` verbatim through the production HTTP/ASGI
  boundary, including authentication, quota, validation, exact-first lookup,
  eligibility, embedding retrieval, compatibility and final response handling.
  Labels and suitability metadata never enter the gateway. Record disposition,
  status, upstream-attempt delta, selected source family and emitted answer.
- Report true semantic hits, false semantic hits, semantic-hit precision, valid
  equivalent coverage, misses, bypasses, rejections and actual response suitability
  with explicit numerators/denominators. A negative semantic hit is a failure even
  if the city accidentally coincides. Do not count cold inference as cache reuse.
- Use actual ONNX embeddings for complete-pipeline scoring. Deterministic fixture
  answers help isolate decision correctness; separately seed genuine Gemma outputs
  and replay them through all final cases without re-generating every negative.
  Distinguish response suitability from byte preservation: for each equivalent,
  check the returned assistant answer satisfies the factual city and city-only
  output contract. Normalization may trim surrounding whitespace only; expected
  content is exactly `Paris` or `Tokyo`, with correct casing and no punctuation,
  JSON quoting or prose. Record genuine source requests, full successful sample
  answers, model/version pins and independent upstream evidence.
- Zero false hits on this finite set is necessary for enabling the proposed gate,
  not sufficient proof of universal safety. Any false semantic hit or unsuitable
  genuine replay fails the preregistered safety gate: retain reviewed-phrase
  serving and report the failure. Do not repair the gate using final labels and
  rerun the same corpus as if still independently held out. If the proposed mode
  succeeds, retain explicit opt-in, bounded domain and all existing safeguards,
  and describe the resulting language coverage precisely.

The original M05/M06 corpora remain diagnostic regression evidence, not unseen
final evaluation. No performance optimization, approval expansion for Python,
embedding model training, external service or vector database is part of M07.
