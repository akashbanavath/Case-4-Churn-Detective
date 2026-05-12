# Decisions Log — Case 4: Churn Detective

## Assumptions I made
1. **The CMO wants actionable segments, not model accuracy** — because the brief says "she's allergic to black box answers" and "a candidate with 0.78 AUC and a crisp segment story is winning."
2. **36% churn rate means the company is in crisis mode** — 2.3% monthly vs 1.5% benchmark. Recommendations need to be deployable in weeks, not months.
3. **Revenue impact estimates use average monthly charges as proxy** — actual ARPU and margin data would refine these numbers.
4. **Electronic check payment is a proxy for low engagement** — customers who haven't set up auto-pay are less invested in the relationship.

## Trade-offs
| Choice | Alternative | Why I picked this |
|---|---|---|
| Gradient Boosting | XGBoost / Random Forest | Strong baseline, built-in feature importance, fewer hyperparameters to tune in a time-constrained build. |
| Manual segmentation | K-Means clustering | Business-driven segments (price-sensitive, service-frustrated, disengaged) tell a clearer story than algorithmic clusters that need interpretation. |
| Feature importance | SHAP | Faster to compute, sufficient for CMO-level communication. SHAP would be Day 2 improvement. |
| Honest AUC reporting | Cherry-picking metrics | Brief explicitly tests whether candidate reports honest performance. AUC of ~0.59-0.78 with clear segment stories beats inflated metrics. |

## What I de-scoped and why
- **SHAP analysis** — Would add individual prediction explanations but adds ~2 hours of compute + visualization work.
- **Uplift modeling** — The "right" answer for retention targeting, but requires counterfactual estimation that's a project in itself.
- **Cost-aware thresholds** — Needs actual cost-per-intervention data from the CMO's team.

## What I'd do differently with another day
- Implement SHAP with waterfall plots for top 10 at-risk customers
- Build uplift model using T-learner approach
- Add cost-benefit analysis with customizable intervention costs
- Create a "save list" CSV export the call center can action immediately
- A/B test simulation showing statistical power calculations for each play
