# Decisions Log — Case 4: Churn Detective

## Key Assumptions
1. **CMO is non-technical** — brief says "she's allergic to black box answers". All outputs use plain language.
2. **36% base churn** — higher than typical telecom (~1.5% monthly). Treating as natural dataset property, not a data quality issue.
3. **Segment overlap is OK** — real customers can be in multiple segments. Retention plays are designed to be stackable.
4. **Revenue per customer ≈ monthly_charges** — used for rough impact sizing since actual margin data isn't provided.

## Trade-offs
| Choice | Alternative | Why |
|---|---|---|
| GradientBoosting (250 trees) | XGBoost / LightGBM / Neural Net | Scikit-learn native, no extra deps, interpretable feature importance. Brief says "understand WHY", not "maximize AUC". |
| Feature importance + evidence tables | SHAP | SHAP adds a dependency and takes time to explain to the CMO. Evidence tables (churn rate per driver) are more intuitive and directly actionable. |
| 3 business segments (manual) | Clustering (K-Means) | Manual segments are tied to business levers (price, service, engagement). Clusters give arbitrary groups the CMO can't act on. |
| Honest AUC reporting | Threshold optimization | Brief says "don't just chase AUC". Reporting confusion matrix + precision/recall tradeoff instead. |
| Evidence-based drivers | Correlation matrices | Showing "churn rate when X" is 10x more useful than "correlation = 0.3" for a non-technical audience. |

## What I de-scoped and why
- **SHAP waterfall plots** — Adds complexity for marginal interpretability gain over the evidence tables approach.
- **Uplift modeling** — Requires counterfactual estimation (who would be SAVED by offer). Mentioned as stretch goal and acknowledged in Risks tab.
- **Cost-aware thresholds** — Needs actual margin/CAC data we don't have. Mentioned in production recommendations.
- **Notebook EDA** — Dashboard IS the deliverable. All EDA is interactive in the app.

## What I'd do with another day
- Add SHAP for individual prediction explanations (call center use case)
- Build uplift model to predict "persuadables" vs inevitable churners
- Cost-aware threshold tuning with actual margin data
- A/B test power analysis for the 60-day measurement plan
- Survival analysis for time-to-churn modeling
