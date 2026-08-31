# Test fixtures

## synthetic_genomic.csv / synthetic_clinical.csv

**Synthetic. 30 hand-written rows. Not clinical data, not patient data, not CHAMP.**

Formerly `genomic.csv` and `clinical.csv` at the repository root, where their names
invited them to be mistaken for real inputs. They were the training data for
`ml/artifacts/legacy-synthetic-v0/` — see that directory's `PROVENANCE.md`.

Their label is a deterministic function of `mutation_type` and `severity`, so they
are useless for training but useful as a small, fully-known fixture for testing
preprocessing and inference plumbing.

They are retained for exactly two reasons:

1. Provenance — they document what the legacy artifacts were fitted on.
2. Test fixtures — deterministic, tiny, and safe to assert against.

They must never be loaded by application code. The authoritative dataset is
`ml/data/champ.csv`.
