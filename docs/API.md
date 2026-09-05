# API reference

Base URL `http://localhost:8000`. Interactive docs at `/docs`.

All `/api/*` endpoints except `auth/register` and `auth/login` require
`Authorization: Bearer <token>`. Every record is scoped to the authenticated
user: another account's patient or prediction returns **404**, not 403, so ids
cannot be probed.

## System

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | `status` is `healthy` only when the database answers **and** the default prediction mode's model is loaded. `model_versions` lists every mode that loaded; a mode missing from it is reported in `detail` but does not by itself make the service unhealthy. |
| GET | `/` | Service name, dataset, target and the research disclaimer. |

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
| DELETE | `/api/patients/{id}` | 204. Cascades to case records, predictions and explanations. |

A patient record holds identity only. No clinical variables are stored on it,
because the predictive input — the mutation description and its clinical record —
is submitted with the prediction.

## Predictions

### Prediction modes

Three models are served, matching the feature blocks of the reference analysis:

| `feature_set` | Model version | Input |
|---|---|---|
| `genomic` | `mmc2-genomic-v1` | 14 MMC2 mutation fields |
| `clinical` | `mmc3-clinical-v1` | 6 MMC3 clinical fields, aggregated per mutation |
| `merged` | `mmc2-mmc3-v1` | all 20 fields, fused (**default**) |

The default is set by `ML_DEFAULT_FEATURE_SET`; a request may choose any mode
that loaded.

### `GET /api/predictions/schema`

Everything needed to build a form for one mode, straight from the fitted model.
Optional query parameter `feature_set`.

```json
{
  "model_version": "mmc2-mmc3-v1",
  "feature_set": "merged",
  "available_feature_sets": ["genomic", "clinical", "merged"],
  "default_feature_set": "merged",

  "categorical": {
    "mut_type":   ["Deletion", "Duplication", "Indel", "Insertion", "Point"],
    "mut_effect": ["Frameshift", "In-frame", "Large Deletion", "Missense",
                   "Nonsense", "Silent", "Splice"],
    "location":   ["Exon", "Intron", "UTR"],
    "cli_phe":    ["Mild", "Mild/Moderate", "Moderate", "Moderate/Severe",
                   "Not reported", "Severe", "Unclassified"]
  },
  "numeric": {
    "aa_numb": { "description": "Residue position of the variant", "required": true }
  },

  "labels": { "mut_type": "Mutation type", "cli_phe": "Clinical severity" },
  "groups": { "mut_type": "genomic", "cli_phe": "clinical" },
  "open_vocabulary": ["mut_syn", "nuc_numb", "clotting", "..."],

  "required": ["mut_type", "mut_effect", "location", "e_i_numb", "locnumb",
               "n_bp", "nuc_numb", "mut_syn", "cli_phe", "aa_numb_old", "aa_numb"],
  "optional": ["codon_change", "ntchange", "CpG", "assay", "antigen", "..."]
}
```

- **`labels`** — a human-readable name per field, so a UI never has to show a raw
  column name.
- **`groups`** — `genomic` or `clinical`, for laying the form out in sections.
- **`required`** — the columns present in at least 95% of the training split.
  Everything else is optional; omitting it records the field as *not measured*
  rather than guessing a value for it. Several clinical columns are >99% missing
  in the source data, so most submissions leave most fields blank.
- **`open_vocabulary`** — fields where the listed values are *suggestions*, not
  the only accepted ones. These are identifiers or measurements written as text
  (`mut_syn` holds 1,610 distinct HGVS strings); a new mutation has a notation
  the model has never seen, and it is routed to the encoder's infrequent bucket
  instead of being rejected. Every other categorical field is validated strictly.

### `POST /api/patients/{id}/predictions`

```json
{
  "feature_set": "merged",
  "features": {
    "mut_type": "Point",
    "mut_effect": "Missense",
    "location": "Exon",
    "e_i_numb": "14",
    "locnumb": "14",
    "n_bp": "1",
    "nuc_numb": "1834",
    "mut_syn": "c.1834C>T",
    "cli_phe": "Severe",
    "aa_numb_old": 593,
    "aa_numb": 612
  },
  "mutation_label": "c.1834C>T"
}
```

`features` is keyed by MMC2/MMC3 source column name. The accepted keys are not
fixed in the API layer: they are whatever the served model's schema reports for
the chosen mode, so retraining on a different feature set needs no API change
and the two can never disagree. `mutation_label` is a display convenience and
never reaches the model.

201 response:

```json
{
  "id": 1,
  "patient_id": 1,
  "prediction": 0,
  "risk": "Low",
  "probability": 0.093977,
  "risk_category": "Lower estimated risk",
  "threshold": 0.24532813727855682,
  "model_version": "mmc2-mmc3-v1",
  "feature_set": "merged",
  "preprocessing_version": "mmc-preprocessing-1",
  "created_at": "2026-09-03 10:31:46",
  "features": { "mut_type": "Point", "...": "..." },
  "mutation_label": "c.1834C>T",
  "interpretation": "The model estimates a 9.4% probability that a record with this mutation and clinical description reports inhibitor development, against a decision threshold of 24.5% (lower estimated risk). This is an estimate attributed to the record's features, not a prediction about an individual patient, and it does not indicate any course of treatment.",
  "disclaimer": "Research decision-support prototype. This estimate is not intended for standalone diagnosis or treatment decisions."
}
```

`prediction` is `1` when `probability >= threshold` and `0` otherwise; `risk` is
the same fact as `"High"` / `"Low"`. `risk_category` carries the longer
`Lower estimated risk` / `Elevated estimated risk` wording. The split is always
the model's own recorded `threshold` — never a hardcoded 0.5, and never a
verdict. The API does not recommend treatment.

**422 on a value outside a closed vocabulary**, with the accepted values,
rather than a prediction about a row the model has never seen:

```json
{
  "detail": {
    "error": "invalid_case_input",
    "detail": "'Bogus' is not a value of 'mut_type' that this model was trained on.",
    "field": "mut_type",
    "allowed_values": ["Deletion", "Duplication", "Indel", "Insertion", "Point"]
  }
}
```

422 is also returned for a missing required field, a non-numeric value in a
numeric field, an unknown `feature_set`, and an empty `features` object.

**503** if the requested mode's model is not loaded. The API never falls back to
a hand-written formula, so a failure can always be told apart from a prediction.
Losing one mode does not affect the others.

### Other prediction endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/api/predictions/{id}` | |
| GET | `/api/patients/{id}/history` | Newest first. |
| GET | `/api/predictions/{id}/explanation` | SHAP + LIME. Computed on first call, then served from storage. `?refresh=true` recomputes. |
| GET | `/api/explanations/global` | Model-wide feature importance. Optional `feature_set`. |

Explanations are a separate endpoint so a slow explainer never delays a
prediction. Contributions are reported against the original MMC2/MMC3 columns,
never against post-encoding names, and never outside the feature set the model
was fitted on:

```json
{
  "feature_set": "merged",
  "unit_of_explanation": "One F8 mutation: its MMC2 genomic description together with the aggregate of the MMC3 clinical records reporting it. ...",
  "shap": {
    "available": true,
    "basis": "uncalibrated ensemble",
    "contributions": [
      { "feature": "mut_effect", "label": "Mutation effect", "value": "Missense",
        "supplied": true, "contribution": -0.3735, "direction": "decreases" },
      { "feature": "CpG", "label": "CpG dinucleotide", "value": null,
        "supplied": false, "contribution": 0.2709, "direction": "increases" }
    ]
  },
  "lime": { "available": true, "contributions": [] }
}
```

`direction` is `increases`, `decreases`, or `no effect`. `supplied` is `false`
for a field the caller left blank: its absence really did move the estimate, so
it is reported — but with a null `value`, never as something the caller entered.
If an explainer fails, `available` is `false` with a `reason`, never a
fabricated attribution.

A prediction made under a different model version returns **409** rather than an
explanation computed by the wrong model.

## Analytics

`GET /api/analytics` — counts computed from this account's stored predictions.
Nothing is synthesised, and `mean_probability` is `null` when there are none.

```json
{
  "total_patients": 1,
  "total_predictions": 2,
  "mean_probability": 0.059413,
  "risk_distribution": { "Lower estimated risk": 2 },
  "mutation_type_distribution": { "Point": 1, "Not supplied": 1 },
  "feature_set_distribution": { "merged": 1, "clinical": 1 },
  "model_version": "mmc2-mmc3-v1"
}
```

A clinical-only submission has no `mut_type`, and is counted as `Not supplied`
rather than being assigned a mutation type it never carried.

## Errors

| Status | Meaning |
|---|---|
| 401 | Missing, invalid or expired token |
| 404 | Not found, or belongs to another account |
| 409 | Duplicate identifier, or model-version mismatch |
| 422 | Input failed validation |
| 500 | Generic body only; details are logged, never returned |
| 503 | Model unavailable |
