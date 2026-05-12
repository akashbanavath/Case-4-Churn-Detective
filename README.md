# Case 4: Churn Detective - Telecom Retention Brief

**Live demo:** *[HF Spaces - deploy pending]*
**Repo:** [github.com/akashbanavath/Case-4-Churn-Detective](https://github.com/akashbanavath/Case-4-Churn-Detective)
**Demo video:** *[Recording pending]*

## What this is
A churn prediction and customer segmentation analysis for a mid-sized telecom, translating ML model output into 3 targeted retention plays the CMO can act on — with honest performance reporting, segment-specific economics, and a 60-day measurement plan.

## How to run locally
```bash
git clone https://github.com/akashbanavath/Case-4-Churn-Detective.git
cd Case-4-Churn-Detective
pip install -r requirements.txt
python -m streamlit run app.py
```
Open http://localhost:8502

## Stack
| Component | Choice | Why |
|-----------|--------|-----|
| Python 3.11 | Core | Data science standard |
| Scikit-learn | ML | GradientBoosting - strong, interpretable |
| Plotly | Viz | Interactive charts for CMO exploration |
| Streamlit | Dashboard | Quick deploy, interactive filters |

## Key Findings
1. Month-to-month contracts churn at 2-3x rate of yearly contracts
2. Three distinct churner segments: price-sensitive, service-frustrated, disengaged
3. Support calls > 4 in 3 months is a strong leading indicator
4. Combined retention plays could save ~$500K+ annually

## What's NOT done
- SHAP values (GBM feature importance is sufficient for CMO-level communication)
- Uplift modeling (predicts who will be SAVED, not just who will churn)
- Cost-aware threshold optimization

## In production, I would also add
- SHAP waterfall plots for individual customer predictions
- Uplift model to identify "persuadables"
- Real-time scoring API for call center integration
- Monthly model retraining pipeline with drift monitoring
- CRM integration for automated campaign triggers

## License
MIT
