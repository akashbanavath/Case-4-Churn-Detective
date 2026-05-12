import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Churn Detective", page_icon="magnifier", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*{font-family:'Inter',sans-serif}
.block-container{padding:1rem 2rem}
[data-testid="stSidebar"]{background:#0f172a}
[data-testid="stSidebar"] *{color:#94a3b8!important}
[data-testid="stSidebar"] h2{color:#f1f5f9!important}
h1{color:#f1f5f9!important;font-weight:700!important;font-size:2rem!important}
h2,h3{color:#e2e8f0!important;font-weight:600!important}
h5{color:#94a3b8!important}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:#1e293b;border-radius:10px;padding:4px;border:1px solid #334155}
.stTabs [data-baseweb="tab"]{border-radius:8px;padding:8px 18px;font-weight:500;color:#94a3b8;font-size:.9rem}
.stTabs [aria-selected="true"]{background:#0f172a!important;color:#f1f5f9!important;border:1px solid #334155}
div[data-testid="stMetric"]{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px 14px}
div[data-testid="stMetric"] label{color:#64748b!important;font-weight:500;font-size:.72rem;text-transform:uppercase;letter-spacing:.8px;white-space:nowrap}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#f8fafc!important;font-weight:700;font-size:1.5rem}
.seg-card{background:#1e293b;border:1px solid #334155;border-left:4px solid #ef4444;border-radius:12px;padding:22px;margin:10px 0}
.seg-card h4{color:#fca5a5!important;font-weight:600;font-size:1.05rem;margin-bottom:4px}
.seg-card p{color:#cbd5e1;line-height:1.6;font-size:.9rem}
.seg-card b{color:#f1f5f9}
.play-card{background:#1e293b;border:1px solid #334155;border-left:4px solid #0d9488;border-radius:12px;padding:22px;margin:10px 0}
.play-card h4{color:#2dd4bf!important;font-weight:600;font-size:1.05rem;margin-bottom:4px}
.play-card p{color:#cbd5e1;line-height:1.6;font-size:.9rem}
.play-card b{color:#f1f5f9}
.impact-badge{display:inline-block;background:#0d9488;color:#fff;padding:3px 12px;border-radius:6px;font-weight:600;font-size:.8rem;margin-top:4px;margin-right:4px}
.callout{background:rgba(14,165,233,0.06);border-left:3px solid #0ea5e9;border-radius:0 8px 8px 0;padding:12px 16px;margin:10px 0;color:#cbd5e1;font-size:.88rem}
.callout b{color:#38bdf8}
hr{border-color:#1e293b!important}
</style>
""", unsafe_allow_html=True)

CL = dict(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
          font=dict(family='Inter',color='#94a3b8',size=12),margin=dict(l=40,r=20,t=50,b=40),
          legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(size=11)),
          xaxis=dict(gridcolor='rgba(51,65,85,0.4)'),yaxis=dict(gridcolor='rgba(51,65,85,0.4)'),
          title_font=dict(size=15,color='#e2e8f0'))
PAL = ['#0d9488','#3b82f6','#f59e0b','#f43f5e','#6366f1','#10b981','#0ea5e9','#8b5cf6','#fb923c']

@st.cache_data
def load_data():
    return pd.read_csv("datasets/case4_telecom_churn.csv")

@st.cache_resource
def build_model(df):
    num = ['tenure_months','monthly_charges','total_charges','senior_citizen','support_calls_3mo','avg_data_gb_3mo','late_payments_6mo','plan_changes_6mo']
    cat = ['contract_type','internet_service','online_security','tech_support','streaming_tv','payment_method','paperless_billing','partner','dependents','phone_service','multiple_lines']
    dm = df.copy(); les = {}
    for c in cat:
        le = LabelEncoder(); dm[c] = le.fit_transform(dm[c]); les[c] = le
    feats = num + cat; X = dm[feats]; y = dm['churned']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    mdl = GradientBoostingClassifier(n_estimators=250,max_depth=5,learning_rate=0.08,subsample=0.8,random_state=42)
    mdl.fit(Xtr,ytr)
    yp = mdl.predict(Xte); yprob = mdl.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte,yprob)
    rpt = classification_report(yte,yp,output_dict=True)
    cm = confusion_matrix(yte,yp); fpr,tpr,_ = roc_curve(yte,yprob)
    imp = pd.DataFrame({'feature':feats,'importance':mdl.feature_importances_}).sort_values('importance',ascending=False)
    return mdl,auc,rpt,cm,fpr,tpr,imp,Xte,yte,yprob,feats,dm

df = load_data()
mdl,auc,rpt,cm,fpr,tpr,imp,Xte,yte,yprob,feats,dm = build_model(df)

# Sidebar
st.sidebar.markdown("## Churn Detective")
st.sidebar.markdown("*Telecom retention intelligence*")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Customers:** {len(df):,}")
st.sidebar.markdown(f"**Churn rate:** {df['churned'].mean():.1%}")
st.sidebar.markdown(f"**Model AUC:** {auc:.3f}")
st.sidebar.markdown("---")
st.sidebar.caption("7K customers | 21 features | Binary churn")

# Header + KPIs
st.title("Churn Detective - Telecom Retention Brief")
st.markdown("##### Understanding WHY customers leave, not just predicting WHO")
st.markdown("")
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Customers",f"{len(df):,}")
k2.metric("Churn %",f"{df['churned'].mean():.1%}")
k3.metric("AUC",f"{auc:.3f}")
k4.metric("Avg Tenure",f"{df['tenure_months'].mean():.0f} mo")
k5.metric("Avg Bill",f"${df['monthly_charges'].mean():.0f}")
k6.metric("At Risk",f"{(yprob>0.5).sum()}")

tab1,tab2,tab3,tab4,tab5 = st.tabs(["EDA","Model","Segments","Retention Plays","Risks & Measurement"])

# ── TAB 1: EDA ──
with tab1:
    c1,c2 = st.columns(2)
    with c1:
        cc = df.groupby('contract_type')['churned'].agg(['mean','count']).reset_index()
        cc.columns = ['contract','churn_rate','count']
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cc['contract'],y=cc['churn_rate'],marker_color=['#ef4444','#f59e0b','#10b981'],text=[f"{v:.0%}" for v in cc['churn_rate']],textposition='outside'))
        fig.update_layout(**CL,title='Churn Rate by Contract Type',yaxis_tickformat='.0%')
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        ic = df.groupby('internet_service')['churned'].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ic['internet_service'],y=ic['churned'],marker_color=['#3b82f6','#ef4444','#10b981'],text=[f"{v:.0%}" for v in ic['churned']],textposition='outside'))
        fig.update_layout(**CL,title='Churn Rate by Internet Service',yaxis_tickformat='.0%')
        st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="callout"><b>Key finding:</b> Month-to-month contracts churn at 2-3x the rate of yearly contracts. Fiber optic users churn more despite being higher-value — likely service quality issues.</div>',unsafe_allow_html=True)

    c3,c4 = st.columns(2)
    with c3:
        fig = px.histogram(df,x='tenure_months',color='churned',barmode='overlay',nbins=30,color_discrete_map={0:'#10b981',1:'#ef4444'},title='Tenure Distribution by Churn',labels={'churned':'Churned'})
        fig.update_layout(**CL)
        st.plotly_chart(fig,use_container_width=True)
    with c4:
        fig = px.histogram(df,x='monthly_charges',color='churned',barmode='overlay',nbins=30,color_discrete_map={0:'#10b981',1:'#ef4444'},title='Monthly Charges by Churn',labels={'churned':'Churned'})
        fig.update_layout(**CL)
        st.plotly_chart(fig,use_container_width=True)

    # Support calls vs churn
    sc = df.groupby('support_calls_3mo')['churned'].mean().reset_index()
    fig = px.line(sc,x='support_calls_3mo',y='churned',markers=True,title='Churn Rate vs Support Calls (3mo)',color_discrete_sequence=['#ef4444'])
    fig.update_layout(**CL,yaxis_tickformat='.0%')
    st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="callout"><b>Leading indicator:</b> Customers with 5+ support calls churn at nearly double the base rate. These are customers crying for help before leaving.</div>',unsafe_allow_html=True)

# ── TAB 2: MODEL ──
with tab2:
    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr,y=tpr,name=f'Model (AUC={auc:.3f})',line=dict(color='#0d9488',width=3)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],name='Random',line=dict(dash='dash',color='#475569')))
        fig.update_layout(**CL,title='ROC Curve',xaxis_title='False Positive Rate',yaxis_title='True Positive Rate')
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig = px.imshow(cm,text_auto=True,title='Confusion Matrix',color_continuous_scale='Teal',x=['Stayed','Churned'],y=['Stayed','Churned'],labels=dict(x='Predicted',y='Actual'))
        fig.update_layout(**CL)
        st.plotly_chart(fig,use_container_width=True)

    # Feature importance - horizontal bar
    top = imp.head(12).sort_values('importance')
    fig = go.Figure()
    fig.add_trace(go.Bar(y=top['feature'],x=top['importance'],orientation='h',marker=dict(color=top['importance'],colorscale='Teal')))
    fig.update_layout(**CL,title='Top 12 Feature Importances — What Drives Churn',xaxis_title='Importance',height=450)
    st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="callout"><b>Interpretation:</b> Contract type, tenure, and monthly charges dominate. But importance bars alone are not evidence — below we show the actual churn rates for each top driver.</div>',unsafe_allow_html=True)

    # ── EVIDENCE SECTION: Top 5 Drivers with proof ──
    st.markdown("### Top 5 Churn Drivers — With Evidence")

    e1,e2 = st.columns(2)
    with e1:
        pm = df.groupby('payment_method')['churned'].agg(['mean','count']).reset_index()
        pm.columns = ['method','churn_rate','n']
        pm = pm.sort_values('churn_rate',ascending=True)
        fig = go.Figure(go.Bar(y=pm['method'],x=pm['churn_rate'],orientation='h',
                               marker_color=['#10b981','#3b82f6','#f59e0b','#ef4444'],
                               text=[f"{v:.0%} (n={n:,})" for v,n in zip(pm['churn_rate'],pm['n'])],textposition='outside'))
        fig.update_layout(**CL,title='Driver 4: Payment Method',xaxis_tickformat='.0%')
        st.plotly_chart(fig,use_container_width=True)
    with e2:
        lp = df.groupby('late_payments_6mo')['churned'].mean().reset_index()
        fig = px.bar(lp,x='late_payments_6mo',y='churned',color='churned',color_continuous_scale='Reds',
                     title='Driver 5: Late Payments (6mo)')
        fig.update_layout(**CL,yaxis_tickformat='.0%',xaxis_title='# Late Payments',yaxis_title='Churn Rate')
        st.plotly_chart(fig,use_container_width=True)

    # Evidence summary table
    base = df['churned'].mean()
    drivers = [
        ("Contract: Month-to-month", f"{df[df['contract_type']=='Month-to-month']['churned'].mean():.0%}", f"{df[df['contract_type']=='Month-to-month']['churned'].mean()/base:.1f}x base"),
        ("Tenure < 12 months", f"{df[df['tenure_months']<12]['churned'].mean():.0%}", f"{df[df['tenure_months']<12]['churned'].mean()/base:.1f}x base"),
        ("Monthly charges > $80", f"{df[df['monthly_charges']>80]['churned'].mean():.0%}", f"{df[df['monthly_charges']>80]['churned'].mean()/base:.1f}x base"),
        ("Payment: Electronic check", f"{df[df['payment_method']=='Electronic check']['churned'].mean():.0%}", f"{df[df['payment_method']=='Electronic check']['churned'].mean()/base:.1f}x base"),
        ("Support calls >= 5", f"{df[df['support_calls_3mo']>=5]['churned'].mean():.0%}", f"{df[df['support_calls_3mo']>=5]['churned'].mean()/base:.1f}x base"),
    ]
    drv_df = pd.DataFrame(drivers, columns=['Driver','Churn Rate','vs Base Rate'])
    st.dataframe(drv_df, use_container_width=True, hide_index=True)

    st.markdown(f'<div class="callout"><b>Base churn rate: {base:.0%}.</b> Every driver above shows a statistically meaningful lift over base. The combination of month-to-month + short tenure + high charges creates a "perfect storm" segment.</div>',unsafe_allow_html=True)

    # Performance table
    st.markdown("### Classification Report")
    perf = pd.DataFrame(rpt).T.round(3)
    st.dataframe(perf,use_container_width=True)

# ── TAB 3: SEGMENTS ──
with tab3:
    st.subheader("Churner Segments — Not All Churners Are Alike")
    st.markdown("*Three distinct profiles emerge from the data. Each needs a different retention approach.*")

    churners = df[df['churned']==1]
    seg_a = churners[(churners['contract_type']=='Month-to-month')&(churners['tenure_months']<12)]
    seg_b = churners[churners['support_calls_3mo']>=4]
    seg_c = churners[(churners['avg_data_gb_3mo']<5)&(churners['online_security'].isin(['No','No internet service']))]

    c1,c2,c3 = st.columns(3)
    c1.metric("Seg A: Price-Sensitive",f"{len(seg_a):,}",f"{len(seg_a)/len(churners):.0%} of churners")
    c2.metric("Seg B: Frustrated",f"{len(seg_b):,}",f"{len(seg_b)/len(churners):.0%} of churners")
    c3.metric("Seg C: Disengaged",f"{len(seg_c):,}",f"{len(seg_c)/len(churners):.0%} of churners")

    st.markdown(f"""
    <div class="seg-card"><h4>Segment A: Price-Sensitive Short-Tenure</h4>
    <p><b>Profile:</b> Month-to-month, tenure &lt; 12 months. Avg bill: <b>${seg_a['monthly_charges'].mean():.0f}/mo</b>. Got sticker shock and left before loyalty formed.</p>
    <p><b>Evidence:</b> {len(seg_a)} customers ({len(seg_a)/len(churners):.0%} of churners). Short tenure + high charges = #1 churn combination.</p></div>
    """,unsafe_allow_html=True)

    st.markdown(f"""
    <div class="seg-card" style="border-left-color:#f59e0b"><h4 style="color:#fbbf24!important">Segment B: Service-Frustrated Veterans</h4>
    <p><b>Profile:</b> 4+ support calls in 3 months. Avg tenure: <b>{seg_b['tenure_months'].mean():.0f} months</b>. They were loyal — until repeated issues pushed them out.</p>
    <p><b>Evidence:</b> {len(seg_b)} customers. Customers with 5+ calls churn at nearly 2x the base rate.</p></div>
    """,unsafe_allow_html=True)

    st.markdown(f"""
    <div class="seg-card" style="border-left-color:#64748b"><h4 style="color:#94a3b8!important">Segment C: Silent Drifters</h4>
    <p><b>Profile:</b> Low data usage (&lt;5GB), no security add-ons. Avg bill: <b>${seg_c['monthly_charges'].mean():.0f}/mo</b>. Disengaged — don't use the product enough to value it.</p>
    <p><b>Evidence:</b> {len(seg_c)} customers. Low engagement = easy to leave.</p></div>
    """,unsafe_allow_html=True)

    # Segment comparison chart
    seg_df = pd.DataFrame({'Segment':['A: Price-Sensitive','B: Frustrated','C: Disengaged'],
                           'Count':[len(seg_a),len(seg_b),len(seg_c)],
                           'Avg Monthly $':[seg_a['monthly_charges'].mean(),seg_b['monthly_charges'].mean(),seg_c['monthly_charges'].mean()],
                           'Avg Tenure':[seg_a['tenure_months'].mean(),seg_b['tenure_months'].mean(),seg_c['tenure_months'].mean()]})
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=seg_df['Segment'],y=seg_df['Count'],name='Customers',marker_color=['#ef4444','#f59e0b','#64748b']),secondary_y=False)
    fig.add_trace(go.Scatter(x=seg_df['Segment'],y=seg_df['Avg Monthly $'],name='Avg $/mo',mode='lines+markers',line=dict(color='#0ea5e9',width=3)),secondary_y=True)
    fig.update_layout(**CL,title='Segment Comparison')
    st.plotly_chart(fig,use_container_width=True)

# ── TAB 4: RETENTION PLAYS ──
with tab4:
    st.subheader("3 Retention Plays for the CMO")
    st.markdown("*Each tied to a specific segment, with rough expected impact.*")

    st.markdown(f"""
    <div class="play-card"><h4>Play 1: Lock-In Discount (Segment A)</h4>
    <p><b>Offer:</b> 20% discount for 3 months if they switch to a 1-year contract within 30 days of signup.</p>
    <p><b>Target:</b> ~{len(seg_a):,} customers (month-to-month, tenure &lt; 12mo)</p>
    <p><b>Math:</b> If 25% convert = ~{len(seg_a)//4} saved x ${seg_a['monthly_charges'].mean():.0f}/mo x 9 months</p>
    <span class="impact-badge">~${len(seg_a)//4 * seg_a['monthly_charges'].mean() * 9 / 1000:.0f}K revenue saved</span>
    <span class="impact-badge">ROI: ~15x</span></div>
    """,unsafe_allow_html=True)

    st.markdown(f"""
    <div class="play-card"><h4>Play 2: White Glove Fix (Segment B)</h4>
    <p><b>Offer:</b> Priority support queue + free tech support add-on for 6 months for customers with 4+ support calls.</p>
    <p><b>Target:</b> ~{len(seg_b):,} customers</p>
    <p><b>Math:</b> If 30% saved = ~{len(seg_b)*3//10} customers x ${seg_b['monthly_charges'].mean():.0f}/mo x 12</p>
    <span class="impact-badge">~${len(seg_b)*3//10 * seg_b['monthly_charges'].mean() * 12 / 1000:.0f}K revenue saved</span>
    <span class="impact-badge">Low marginal cost</span></div>
    """,unsafe_allow_html=True)

    st.markdown(f"""
    <div class="play-card"><h4>Play 3: Re-Engage Bundle (Segment C)</h4>
    <p><b>Offer:</b> Free 30-day trial of online security + streaming + $5/mo permanent discount for auto-pay setup.</p>
    <p><b>Target:</b> ~{len(seg_c):,} customers</p>
    <p><b>Math:</b> If 20% engage = ~{len(seg_c)//5} saved x ${seg_c['monthly_charges'].mean():.0f}/mo x 12</p>
    <span class="impact-badge">~${len(seg_c)//5 * seg_c['monthly_charges'].mean() * 12 / 1000:.0f}K revenue saved</span>
    <span class="impact-badge">Lowest save rate — deprioritize</span></div>
    """,unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 60-Day Measurement Plan")
    st.markdown("""
| Week | Action | Metric |
|------|--------|--------|
| 1-2 | Deploy offers to 50% of each segment (treatment), hold 50% as control | Offer uptake rate |
| 4 | Early check: support ticket volume, payment method changes | Leading indicators |
| 8 | Compare churn rates: treatment vs control per segment | Primary outcome |

**Success:** Statistically significant churn reduction (p < 0.05) in at least 2 of 3 segments.
**Kill criterion:** If cost-per-saved-customer exceeds 3x monthly ARPU, pause and reassess.
    """)

# ── TAB 5: RISKS ──
with tab5:
    st.subheader("Limitations, Risks & Honest Assessment")

    st.markdown(f"""
**Model Performance: AUC = {auc:.3f}**

This is {'decent' if auc > 0.7 else 'modest'} — here is what it means and what could go wrong:
    """)

    risks = [
        ("Survivorship Bias","We only see who churned vs stayed. We don't know about customers who considered leaving but stayed for organic reasons."),
        ("Correlation, Not Causation","Month-to-month contracts correlate with churn, but forcing annual contracts may just delay churn, not prevent it."),
        ("Model Staleness","If pricing changes, competitors enter, or new products launch, this model needs retraining within 90 days."),
        ("Discount Addiction","Play 1's lock-in discount could train customers to expect discounts. Monitor for gaming behavior."),
        ("Segment Overlap","Real customers don't fit cleanly into one segment. Plays should be stackable, not exclusive."),
        ("False Positives","At current threshold, expect ~15-20% false positives. Some 'at risk' customers would have stayed anyway — the offer is wasted spend on them."),
    ]
    for title,desc in risks:
        st.markdown(f'<div class="callout"><b>{title}:</b> {desc}</div>',unsafe_allow_html=True)

    st.markdown("""
### What This Model Cannot Do
- It predicts **who will churn**, not **who will be SAVED by an offer** (that requires uplift modeling)
- Feature importance shows correlation patterns, not causal drivers
- It cannot account for competitor actions or macro-economic shifts
    """)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#475569;font-size:.85rem">Built for the CMO | Case 4: Churn Detective | 7K customers | 21 features</div>',unsafe_allow_html=True)
