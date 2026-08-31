# Legacy frontend components

Superseded React components from the pre-consolidation SPA.

| File | Replaced by / reason |
|---|---|
| `Chatbot.tsx`, `ChatBox.tsx` | The chatbot is out of scope; all five backend chatbot implementations are in `archive/legacy-chatbot/`. |
| `PatientCard.tsx` | Rendered fabricated clinical fields (blood type, severity) that no longer exist on a patient record. |
| `FormField.tsx` | Superseded by `GenomicForm.tsx`, whose inputs are driven by the model's own fitted category vocabulary. |
| `Charts.tsx` | Superseded by charts built directly in `pages/Analytics.tsx` against real stored predictions. |
| `Navbar.tsx` | Never rendered; navigation lives in `Sidebar.tsx`. |
| `Predictions.tsx` | Superseded by `components/GenomicForm.tsx` on the patient detail page. The old page posted free-text clinical fields to an endpoint that computed a hand-written formula. |
| `SHAPAnalysis.tsx` | Superseded by `pages/Explanation.tsx`, which renders real per-prediction SHAP and LIME output rather than a standalone chart. |
