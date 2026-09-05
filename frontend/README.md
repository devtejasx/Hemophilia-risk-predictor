# Hemophilia A Inhibitor-Risk Frontend

React + TypeScript UI for the Hemophilia A inhibitor-risk research prototype.
It collects a description of an F8 mutation, asks the backend for an estimate,
and shows the SHAP/LIME attribution behind that estimate.

> Research prototype. Nothing here is intended for standalone diagnosis or
> treatment decisions.

## What the system actually models

The unit of analysis is a **mutation**, not a patient and not a single clinical
record.

- **MMC2** is the genomic table: one row per mutation, describing its type,
  effect, location, exon/intron, codon and nucleotide position.
- **MMC3** is the clinical table: one row per clinical record, carrying the
  assay values and the reported inhibitor outcome. Several records may report
  the same mutation.

The two are joined on the mutation id and aggregated into **one row per
mutation**, so an estimate says how often that mutation is reported alongside an
inhibitor in the source literature. It is not an individual patient's
probability of developing an inhibitor, and the UI says so on every screen that
shows a number (`MutationLevelNote` in `src/components/Disclaimer.tsx`).

Mutations whose clinical records disagree about the outcome are excluded
upstream, in the ML pipeline.

## Prediction modes

The mode picker on the prediction form offers whatever
`GET /api/predictions/schema` reports as available:

| Mode | Source | Use it when |
| --- | --- | --- |
| `merged` | MMC2 + MMC3 fused at the mutation level | You have both the mutation description and the clinical findings. This is the served default. |
| `genomic` | MMC2 only | You have only the mutation description, e.g. before any assay result is back. |
| `clinical` | MMC3 only | You have only the reported clinical findings and assay values. |

Which mode is the default is read from the schema response
(`default_feature_set`) rather than hardcoded, and the model artifact version is
read from `model_version`. Neither is duplicated in the UI copy.

## The prediction form is schema-driven

`src/components/PredictionForm.tsx` hardcodes **no field list**. For the chosen
mode the backend reports which fields the fitted model accepts, a human-readable
label for each, whether it belongs to the genomic or clinical block, whether it
is required, and which values it was fitted on. Consequences:

- The UI cannot offer a value the model has never seen.
- A retrained model with different columns needs no frontend change.
- Fields marked `open_vocabulary` (HGVS notation, an activity reading) render as
  free text with the known values as suggestions, because a mutation you are
  describing may carry a notation the model has not seen. Everything else is a
  strict `<select>`.
- A field left blank is omitted from the request rather than sent as `""`, so
  the model records it as not measured.

Keep it that way. Do not reintroduce a static field list.

## Quick start

```bash
npm install

cp .env.example .env
# set VITE_API_URL if the backend is not on http://localhost:8000/api

npm run dev        # http://localhost:3000, /api proxied to :8000
```

Other scripts:

```bash
npm run build       # tsc && vite build -> dist/
npm run preview     # serve the built bundle
npm run lint        # eslint, zero warnings tolerated
npm run type-check  # tsc --noEmit
```

## Project structure

```
src/
├── components/
│   ├── Disclaimer.tsx      # research-prototype banner + MutationLevelNote
│   ├── MetricCard.tsx      # dashboard KPI tile
│   ├── PredictionForm.tsx  # schema-driven input form + mode picker
│   ├── RiskBadge.tsx       # probability, band and the model's own threshold
│   └── Sidebar.tsx         # nav, theme toggle, sign out
├── pages/
│   ├── Login.tsx           # sign in / register
│   ├── Dashboard.tsx       # counts and recent patients
│   ├── Patients.tsx        # patient list
│   ├── AddPatient.tsx      # identity only; no clinical fields are collected here
│   ├── PatientDetail.tsx   # prediction form + estimate history
│   ├── Explanation.tsx     # SHAP and LIME attribution for one estimate
│   └── Analytics.tsx       # global feature importance and your own distributions
├── services/
│   ├── api.ts              # axios instance, auth interceptor, error formatting
│   └── api-client.ts       # typed endpoint wrappers; mirrors backend/schemas.py
├── store/appStore.ts       # zustand: user, theme, patient list
├── styles/index.css        # Tailwind entry
├── App.tsx                 # routes and the auth gate
└── main.tsx
```

`AddPatient` deliberately collects identity only. An earlier version asked for
around twenty clinical fields (blood type, joint damage score, adherence, HLA
typing) that exist nowhere in the dataset and never reached the model.

## Backend endpoints used

All are relative to `VITE_API_URL`. A `401` clears the stored token and bounces
to `/login`.

| Endpoint | Purpose |
| --- | --- |
| `POST /auth/register`, `POST /auth/login`, `GET /auth/me` | Authentication |
| `GET/POST /patients`, `GET/DELETE /patients/{id}` | Patient records |
| `GET /predictions/schema?feature_set=` | Fields, labels and allowed values per mode |
| `POST /patients/{id}/predictions` | Run an estimate |
| `GET /predictions/{id}`, `GET /patients/{id}/history` | Read estimates back |
| `GET /predictions/{id}/explanation` | SHAP and LIME attribution |
| `GET /explanations/global` | Model-wide feature importance |
| `GET /analytics` | Counts and distributions |

`src/services/api-client.ts` mirrors `backend/schemas.py`. Keep the two in step.

## Copy rules

Two rules apply to anything user-visible in this app:

1. **No performance numbers in UI copy.** No ROC-AUC, PR-AUC, precision, recall,
   F1, Brier, accuracy, row counts or feature counts, and no informal stand-in
   for one ("discriminates best", "highly accurate"). Metrics come from a real
   training run and live in the ML documentation, not in a blurb.
2. **Mutation framing, not patient framing.** An estimate describes a mutation
   as the source literature reports it. Never word it as an individual patient's
   future risk.

## Tech stack

React 18, TypeScript, Vite, Tailwind (class-based dark mode), React Router v6,
Axios, Zustand, Recharts, Lucide icons.

## Troubleshooting

- **Requests fail / CORS errors** — check `VITE_API_URL` and that the backend is
  up on port 8000; in dev, `/api` is proxied by Vite.
- **Signed out immediately** — the stored token expired; the interceptor clears
  it and redirects to `/login`.
- **The form says the model is unavailable** — `GET /predictions/schema` failed,
  usually because no model artifact is loaded on the backend.
- **Dark mode not applying** — `App.tsx` toggles `.dark` on `<html>` from the
  stored theme; check `localStorage.theme` and the Tailwind `darkMode` config.
