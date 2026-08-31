# API reference

Base URL `http://localhost:8000`. Interactive docs at `/docs`.

All `/api/*` endpoints except `auth/register` and `auth/login` require
`Authorization: Bearer <token>`. Every record is scoped to the authenticated
user: another account's patient or prediction returns **404**, not 403, so ids
cannot be probed.

## System

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | `status` is `healthy` only when the database answers **and** a model is loaded. Returns `degraded` otherwise, with `detail`. |
| GET | `/` | Service name and the research disclaimer. |

## Authentication

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/auth/register` | `email`, `full_name`, `password` (min 12 chars) | 201 + token |
| POST | `/api/auth/login` | `email`, `password` | 200 + token |
| GET | `/api/auth/me` | — | the current user |

Login returns the same message and status for an unknown email as for a wrong
password, so the endpoint does not enumerate accounts. Passwords are bcrypt
hashed (SHA-256 pre-hashed so long passphrases are not silently truncated at
bcrypt's 72-byte limit).

## Patients

| Method | Path | Notes |
|---|---|---|
| POST | `/api/patients` | `identifier`, `display_name`, optional `notes`. 409 if the identifier is already used by this account. |
| GET | `/api/patients` | `limit`, `offset`. Includes `prediction_count` and the latest estimate. |
| GET | `/api/patients/{id}` | |
| DELETE | `/api/patients/{id}` | 204. Cascades to profiles, predictions and explanations. |

A patient record holds identity only. No clinical variables are stored, because
none of them reach the model — the predictive input is the F8 variant.

## Predictions

### `GET /api/predictions/schema`

The category vocabulary the model was actually fitted on. The frontend builds
its form from this, so the UI cannot offer a value the model has never seen.

```json
{
  "model_version": "champ-v1",
  "categorical": {
    "Variant Type": ["3'UTR", "5'UTR", "Frameshift", "Large structural change (>50 bp)", "..."],
    "Domain": ["3'UTR", "5'UTR", "A1", "A2", "A3", "B", "C1", "C2", "Signal", "a1", "a2", "a3"]
  },
  "numeric": { "exon_number": {...}, "codon_number": {...} },
  "boolean": { "is_intron": {...} },
  "required": ["Variant Type", "Mechanism", "Domain", "Subtype", "In Poly A", "Reported Clinical Severity"]
}
```

`A1` and `a1` are **different categories**: lowercase a1–a3 are the acidic
regions, biologically distinct from the A1–A3 domains.

### `POST /api/patients/{id}/predictions`

```json
{
  "Variant Type": "Large structural change (>50 bp)",
  "Mechanism": "Deletion",
  "Domain": "A2",
  "Subtype": "Heavy chain",
  "In Poly A": "N",
  "Reported Clinical Severity": "Severe",
  "exon_number": 14,
  "codon_number": 1200,
  "is_intron": false
}
```

201 response:

```json
{
  "id": 1,
  "patient_id": 1,
  "probability": 0.246929,
  "risk_category": "Elevated estimated risk",
  "threshold": 0.17497797170639112,
  "model_version": "champ-v1",
  "preprocessing_version": "champ-preprocessing-1",
  "created_at": "2026-08-31 12:23:24",
  "interpretation": "The model estimates a 24.7% probability that this F8 variant has a reported history of inhibitor development, against a decision threshold of 17.5% (elevated estimated risk). This is an estimate attributed to the variant, not a prediction about an individual patient, and it does not indicate any course of treatment.",
  "disclaimer": "Research decision-support prototype. This estimate is not intended for standalone diagnosis or treatment decisions."
}
```

`risk_category` is `Lower estimated risk` or `Elevated estimated risk`, split at
the model's own recorded `threshold` — never a hardcoded 0.5, and never a
verdict. The API does not recommend treatment.

**422 on an unknown category**, with the accepted values, rather than a
prediction about a row the model has never seen:

```json
{
  "detail": {
    "error": "invalid_genomic_input",
    "detail": "'Bogus' is not a value of 'Variant Type' that this model was trained on.",
    "field": "Variant Type",
    "allowed_values": ["3'UTR", "5'UTR", "Frameshift", "..."]
  }
}
```

**503** if no model is loaded. The API never falls back to a hand-written
formula, so a failure can always be told apart from a prediction.

### Other prediction endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/api/predictions/{id}` | |
| GET | `/api/patients/{id}/history` | Newest first. |
| GET | `/api/predictions/{id}/explanation` | SHAP + LIME. Computed on first call, then served from storage. `?refresh=true` recomputes. |
| GET | `/api/explanations/global` | Model-wide feature importance. |

Explanations are a separate endpoint so a slow explainer never delays a
prediction. Contributions are reported against the original CHAMP columns and
never name a feature the caller did not supply:

```json
{
  "shap": {
    "available": true,
    "basis": "uncalibrated ensemble",
    "contributions": [
      { "feature": "Variant Type", "value": "Nonsense", "contribution": 0.0694, "direction": "increases" }
    ]
  },
  "lime": { "available": true, "contributions": [] }
}
```

`direction` is `increases`, `decreases`, or `no effect`. If an explainer fails,
`available` is `false` with a `reason` — never a fabricated attribution.

A prediction made under a different model version returns **409** rather than an
explanation computed by the wrong model.

## Analytics

`GET /api/analytics` — counts computed from this account's stored predictions.
Nothing is synthesised, and `mean_probability` is `null` when there are none.

## Errors

| Status | Meaning |
|---|---|
| 401 | Missing, invalid or expired token |
| 404 | Not found, or belongs to another account |
| 409 | Duplicate identifier, or model-version mismatch |
| 422 | Input failed validation |
| 500 | Generic body only; details are logged, never returned |
| 503 | Model unavailable |
