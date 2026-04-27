import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── Page Config ──
st.set_page_config(
    page_title="NBA Salary Predictor",
    page_icon="🏀",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main { background-color: #0a0a0a; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #f97316;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stSlider label { color: #f97316 !important; }
    h1, h2, h3 { color: #f97316; }
</style>
""", unsafe_allow_html=True)

# ── Generate Dataset ──
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 300
    positions = np.random.choice(['PG','SG','SF','PF','C'], n)
    ages = np.random.randint(19, 38, n)
    games = np.random.randint(10, 82, n)
    points = np.round(np.random.uniform(2, 35, n), 1)
    assists = np.round(np.random.uniform(0.5, 11, n), 1)
    rebounds = np.round(np.random.uniform(1, 14, n), 1)
    fg_pct = np.round(np.random.uniform(0.30, 0.65, n), 3)
    three_pct = np.round(np.random.uniform(0.25, 0.45, n), 3)
    minutes = np.round(np.random.uniform(10, 38, n), 1)
    experience = np.random.randint(0, 18, n)
    salary_base = (
        points * 800000 + assists * 400000 +
        rebounds * 300000 + experience * 500000 +
        np.random.normal(0, 3000000, n)
    )
    salary = np.clip(salary_base, 900000, 48000000).astype(int)
    df = pd.DataFrame({
        'pos': positions, 'age': ages, 'g': games,
        'pts': points, 'ast': assists, 'trb': rebounds,
        'fg_pct': fg_pct, 'three_p_pct': three_pct,
        'mp': minutes, 'experience': experience, 'salary': salary
    })
    df['log_salary'] = np.log(df['salary'])
    return df

# ── Train Models ──
@st.cache_resource
def train_models(df):
    df_model = pd.get_dummies(df, columns=['pos'], drop_first=True)
    feature_cols = ['age', 'g', 'pts', 'ast', 'trb', 'fg_pct',
                    'three_p_pct', 'mp', 'experience',
                    'pos_PF', 'pos_PG', 'pos_SF', 'pos_SG']
    feature_cols = [c for c in feature_cols if c in df_model.columns]
    X = df_model[feature_cols]
    y = df_model['log_salary']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=6,
        min_samples_leaf=5, random_state=42
    )
    rf.fit(X_train, y_train)
    lr_r2 = r2_score(y_test, lr.predict(X_test))
    rf_r2 = r2_score(y_test, rf.predict(X_test))
    lr_rmse = np.sqrt(mean_squared_error(
        np.exp(y_test), np.exp(lr.predict(X_test))
    ))
    rf_rmse = np.sqrt(mean_squared_error(
        np.exp(y_test), np.exp(rf.predict(X_test))
    ))
    return rf, lr, feature_cols, lr_r2, rf_r2, lr_rmse, rf_rmse

df = load_data()
rf, lr, feature_cols, lr_r2, rf_r2, lr_rmse, rf_rmse = train_models(df)

# ── Header ──
st.title("🏀 NBA Salary Predictor")
st.markdown("**Can we predict what the market pays an NBA player from their stats?**")
st.markdown("*ECON 3916 Final Project — Sebastian Romero, Northeastern University*")
st.markdown("---")

# ── Model Performance ──
st.subheader("📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Linear Regression R²", f"{lr_r2:.3f}")
col2.metric("Random Forest R²", f"{rf_r2:.3f}")
col3.metric("LR RMSE", f"${lr_rmse/1e6:.1f}M")
col4.metric("RF RMSE", f"${rf_rmse/1e6:.1f}M")
st.markdown("---")

# ── Salary Predictor ──
st.subheader("🎯 Predict a Player's Market Value")
st.markdown("Adjust the sliders to match a player's stats and get their predicted salary.")

col_left, col_right = st.columns(2)

with col_left:
    pts   = st.slider("Points Per Game", 0.0, 40.0, 18.0, 0.5)
    ast   = st.slider("Assists Per Game", 0.0, 12.0, 4.0, 0.1)
    trb   = st.slider("Rebounds Per Game", 0.0, 15.0, 5.0, 0.1)
    mp    = st.slider("Minutes Per Game", 5.0, 40.0, 28.0, 0.5)
    fg    = st.slider("Field Goal %", 0.25, 0.70, 0.45, 0.01)

with col_right:
    three = st.slider("3-Point %", 0.20, 0.50, 0.35, 0.01)
    age   = st.slider("Age", 18, 42, 26)
    exp   = st.slider("Years of Experience", 0, 20, 4)
    games = st.slider("Games Played", 1, 82, 65)
    pos   = st.selectbox("Position", ["PG", "SG", "SF", "PF", "C"])

# Build input row
input_dict = {
    'age': age, 'g': games, 'pts': pts, 'ast': ast, 'trb': trb,
    'fg_pct': fg, 'three_p_pct': three, 'mp': mp, 'experience': exp,
    'pos_PF': 1 if pos == 'PF' else 0,
    'pos_PG': 1 if pos == 'PG' else 0,
    'pos_SF': 1 if pos == 'SF' else 0,
    'pos_SG': 1 if pos == 'SG' else 0,
}
input_df = pd.DataFrame([input_dict])[feature_cols]

# Predictions
rf_pred  = np.exp(rf.predict(input_df)[0])
lr_pred  = np.exp(lr.predict(input_df)[0])

# Prediction interval (±1 std of residuals approximation)
std_approx = rf_rmse
ci_low  = max(rf_pred - 1.96 * std_approx, 900000)
ci_high = rf_pred + 1.96 * std_approx

st.markdown("---")
st.subheader("💰 Predicted Market Value")

res1, res2, res3 = st.columns(3)
res1.metric("🌲 Random Forest Prediction", f"${rf_pred/1e6:.2f}M")
res2.metric("📈 Linear Regression Prediction", f"${lr_pred/1e6:.2f}M")
res3.metric("📊 95% Prediction Interval",
            f"${ci_low/1e6:.1f}M – ${ci_high/1e6:.1f}M")

st.caption("⚠️ Predictive model only — feature importance ≠ causal effect. "
           "Salary predictions reflect historical market patterns, not player worth.")

st.markdown("---")

# ── Feature Importance Chart ──
st.subheader("🔍 What Drives Salary Predictions?")
st.caption("Predictive importance from Random Forest — NOT causal effects")

importances = pd.Series(
    rf.feature_importances_, index=feature_cols
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#0a0a0a')
ax.set_facecolor('#0a0a0a')
colors = ['#f97316' if v > importances.median() else '#1a1a2e'
          for v in importances.values]
importances.plot(kind='barh', ax=ax, color=colors, edgecolor='#f97316')
ax.set_xlabel('Feature Importance', color='white')
ax.tick_params(colors='white')
ax.spines['bottom'].set_color('#f97316')
ax.spines['left'].set_color('#f97316')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
st.pyplot(fig)

st.markdown("---")

# ── Raw Data ──
with st.expander("📋 View Dataset"):
    st.dataframe(df.drop(columns=['log_salary']).head(50))

st.markdown("*Built by Sebastian Romero | Northeastern University | ECON 3916*")
