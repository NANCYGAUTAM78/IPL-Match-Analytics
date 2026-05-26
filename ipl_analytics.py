"""
IPL Match Analytics & Win Predictor
=====================================
Phase 1: Data Loading & Cleaning
Phase 2: Exploratory Data Analysis
Phase 3: Feature Engineering & ML Model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "axes.titlesize": 13,
    "axes.titleweight": "bold"
})
PALETTE = ["#1F4E79", "#2E75B6", "#70AD47", "#ED7D31", "#FFC000",
           "#FF0000", "#7030A0", "#00B0F0"]

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — LOAD & CLEAN DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 1: Loading & Cleaning Data")
print("=" * 60)

df = pd.read_csv("ipl_matches.csv")
print(f"\n✅ Loaded {len(df)} matches across {df['season'].nunique()} seasons")
print(f"   Columns : {list(df.columns)}")
print(f"   Missing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")
print(f"\nSample:\n{df.head(3).to_string()}")

# Clean: no nulls expected, but let's drop any
df.dropna(inplace=True)
df["season"] = df["season"].astype(int)
df["toss_won_match"] = (df["toss_winner"] == df["winner"]).astype(int)
df["chased_successfully"] = (df["winner"] == df["team2"]).astype(int)

print(f"\n✅ Cleaning complete. Shape: {df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — EDA & VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 2: Exploratory Data Analysis")
print("=" * 60)

# ── 2A: Total wins per team ───────────────────────────────────────────────────
win_counts = df["winner"].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(win_counts.index, win_counts.values,
               color=PALETTE[:len(win_counts)], edgecolor="white", height=0.65)
for bar, val in zip(bars, win_counts.values):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=10, color="#333")
ax.set_xlabel("Total Wins (2008–2023)", fontsize=11)
ax.set_title("Total IPL Wins by Team (2008–2023)")
ax.set_facecolor("#F8F9FA")
fig.patch.set_facecolor("#FFFFFF")
plt.tight_layout()
plt.savefig("plot1_team_wins.png")
plt.close()
print("✅ Plot 1 saved: Total Wins per Team")

# ── 2B: Toss decision distribution ───────────────────────────────────────────
toss_dec = df["toss_decision"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].pie(toss_dec.values, labels=toss_dec.index, autopct="%1.1f%%",
            colors=["#1F4E79", "#70AD47"], startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[0].set_title("Toss Decision Distribution")

# Toss win → match win
toss_impact = df["toss_won_match"].value_counts()
axes[1].bar(["Lost Match", "Won Match"], toss_impact.values,
            color=["#ED7D31", "#1F4E79"], edgecolor="white", width=0.5)
axes[1].set_title("Toss Winner → Match Outcome")
axes[1].set_ylabel("Number of Matches")
for i, v in enumerate(toss_impact.values):
    axes[1].text(i, v + 3, str(v), ha="center", fontsize=11, fontweight="bold")

fig.patch.set_facecolor("#FFFFFF")
plt.tight_layout()
plt.savefig("plot2_toss_analysis.png")
plt.close()
print("✅ Plot 2 saved: Toss Analysis")

# ── 2C: Season-wise avg score trend ──────────────────────────────────────────
season_avg = df.groupby("season").agg(
    avg_score=("team1_score", "mean"),
    avg_rr=("team1_run_rate", "mean")
).reset_index()

fig, ax1 = plt.subplots(figsize=(10, 4))
ax2 = ax1.twinx()
ax1.fill_between(season_avg["season"], season_avg["avg_score"],
                 alpha=0.25, color="#1F4E79")
ax1.plot(season_avg["season"], season_avg["avg_score"],
         color="#1F4E79", marker="o", linewidth=2.2, label="Avg Score")
ax2.plot(season_avg["season"], season_avg["avg_rr"],
         color="#ED7D31", marker="s", linewidth=2, linestyle="--", label="Avg Run Rate")
ax1.set_xlabel("Season")
ax1.set_ylabel("Average First Innings Score", color="#1F4E79")
ax2.set_ylabel("Average Run Rate", color="#ED7D31")
ax1.set_title("Season-wise Scoring Trends (2008–2023)")
ax1.set_xticks(season_avg["season"])
ax1.set_xticklabels(season_avg["season"], rotation=45)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
fig.patch.set_facecolor("#FFFFFF")
plt.tight_layout()
plt.savefig("plot3_season_trends.png")
plt.close()
print("✅ Plot 3 saved: Season Scoring Trends")

# ── 2D: Head-to-head win rate heatmap ────────────────────────────────────────
teams = sorted(df["team1"].unique())
matrix = pd.DataFrame(0.0, index=teams, columns=teams)

for _, row in df.iterrows():
    t1, t2, w = row["team1"], row["team2"], row["winner"]
    matrix.loc[t1, t2] = matrix.loc[t1, t2]  # placeholder
    if w == t1:
        matrix.loc[t1, t2] += 1
    else:
        matrix.loc[t2, t1] += 1

# Normalize to win %
counts = pd.DataFrame(0, index=teams, columns=teams)
for _, row in df.iterrows():
    counts.loc[row["team1"], row["team2"]] += 1
    counts.loc[row["team2"], row["team1"]] += 1

win_pct = (matrix / counts.replace(0, np.nan) * 100).fillna(0)

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.eye(len(teams), dtype=bool)
sns.heatmap(win_pct, annot=True, fmt=".0f", cmap="Blues",
            linewidths=0.5, ax=ax, mask=mask,
            cbar_kws={"label": "Win %"}, annot_kws={"size": 9})
ax.set_title("Head-to-Head Win % (Row team vs Column team)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
fig.patch.set_facecolor("#FFFFFF")
plt.tight_layout()
plt.savefig("plot4_h2h_heatmap.png")
plt.close()
print("✅ Plot 4 saved: Head-to-Head Heatmap")

# ── 2E: Chase success rate by team ───────────────────────────────────────────
chase_df = df.copy()
chase_df["chased"] = (chase_df["winner"] == chase_df["team2"]).astype(int)
chase_rate = chase_df.groupby("team2")["chased"].mean().sort_values() * 100

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#ED7D31" if v < 50 else "#1F4E79" for v in chase_rate.values]
bars = ax.barh(chase_rate.index, chase_rate.values, color=colors, height=0.6, edgecolor="white")
ax.axvline(50, color="gray", linestyle="--", linewidth=1.2, label="50% line")
for bar, val in zip(bars, chase_rate.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=10)
ax.set_xlabel("Chase Success Rate (%)")
ax.set_title("Chase Success Rate by Team")
ax.legend()
fig.patch.set_facecolor("#FFFFFF")
plt.tight_layout()
plt.savefig("plot5_chase_success.png")
plt.close()
print("✅ Plot 5 saved: Chase Success Rate")

# ── 2F: Score distribution boxplot ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
team_scores = [df[df["team1"] == t]["team1_score"].values for t in teams]
bp = ax.boxplot(team_scores, patch_artist=True, notch=False,
                medianprops={"color": "white", "linewidth": 2})
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_xticklabels([t.split()[0] for t in teams], rotation=30, ha="right")
ax.set_ylabel("First Innings Score")
ax.set_title("Score Distribution by Team (First Innings)")
fig.patch.set_facecolor("#FFFFFF")
plt.tight_layout()
plt.savefig("plot6_score_distribution.png")
plt.close()
print("✅ Plot 6 saved: Score Distribution")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — FEATURE ENGINEERING & WIN PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 3: Feature Engineering & Win Predictor")
print("=" * 60)

le_team = LabelEncoder()
le_venue = LabelEncoder()

all_teams = pd.concat([df["team1"], df["team2"]]).unique()
le_team.fit(all_teams)
le_venue.fit(df["venue"])

df["team1_enc"]      = le_team.transform(df["team1"])
df["team2_enc"]      = le_team.transform(df["team2"])
df["venue_enc"]      = le_venue.transform(df["venue"])
df["toss_winner_enc"]= le_team.transform(df["toss_winner"])
df["toss_bat_first"] = (df["toss_decision"] == "bat").astype(int)

# Historical win rate feature
win_rate = {}
for team in all_teams:
    matches = len(df[(df["team1"] == team) | (df["team2"] == team)])
    wins    = len(df[df["winner"] == team])
    win_rate[team] = wins / matches if matches > 0 else 0.5

df["team1_win_rate"] = df["team1"].map(win_rate)
df["team2_win_rate"] = df["team2"].map(win_rate)
df["win_rate_diff"]  = df["team1_win_rate"] - df["team2_win_rate"]

# Run rate differential
df["rr_diff"] = df["team1_run_rate"] - df["team2_run_rate"]

# Powerplay differential
df["pp_diff"] = df["team1_powerplay_runs"] - df["team2_powerplay_runs"]

# Target: did team1 win?
df["team1_won"] = (df["winner"] == df["team1"]).astype(int)

FEATURES = [
    "team1_enc", "team2_enc", "venue_enc",
    "toss_bat_first", "toss_winner_enc",
    "team1_win_rate", "team2_win_rate", "win_rate_diff",
    "rr_diff", "pp_diff",
    "team1_score", "team1_wickets", "team1_powerplay_runs"
]

X = df[FEATURES]
y = df["team1_won"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Team2 Wins", "Team1 Wins"]))

# ── Feature Importance ────────────────────────────────────────────────────────
coef_df = pd.DataFrame({
    "Feature": FEATURES,
    "Coefficient": np.abs(model.coef_[0])
}).sort_values("Coefficient", ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(coef_df["Feature"], coef_df["Coefficient"],
        color="#1F4E79", edgecolor="white", height=0.65)
ax.set_title("Feature Importance (Logistic Regression Coefficients)")
ax.set_xlabel("Absolute Coefficient Value")
fig.patch.set_facecolor("#FFFFFF")
plt.tight_layout()
plt.savefig("plot7_feature_importance.png")
plt.close()
print("✅ Plot 7 saved: Feature Importance")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Team2 Wins", "Team1 Wins"],
            yticklabels=["Team2 Wins", "Team1 Wins"],
            linewidths=0.5)
ax.set_title(f"Confusion Matrix  (Accuracy: {acc*100:.1f}%)")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
fig.patch.set_facecolor("#FFFFFF")
plt.tight_layout()
plt.savefig("plot8_confusion_matrix.png")
plt.close()
print("✅ Plot 8 saved: Confusion Matrix")

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT A NEW MATCH (Demo)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DEMO: Predict a New Match")
print("=" * 60)

def predict_match(team1, team2, venue, toss_winner, toss_decision,
                  team1_score, team1_wickets, team1_pp, team2_rr):
    t1e  = le_team.transform([team1])[0]
    t2e  = le_team.transform([team2])[0]
    ve   = le_venue.transform([venue])[0]
    twe  = le_team.transform([toss_winner])[0]
    bat  = 1 if toss_decision == "bat" else 0
    t1wr = win_rate.get(team1, 0.5)
    t2wr = win_rate.get(team2, 0.5)
    rrd  = (team1_score / 20) - team2_rr
    ppd  = team1_pp - 46
    row  = [[t1e, t2e, ve, bat, twe, t1wr, t2wr, t1wr - t2wr,
             rrd, ppd, team1_score, team1_wickets, team1_pp]]
    prob = model.predict_proba(row)[0]
    print(f"\n  {team1} vs {team2}")
    print(f"  Venue   : {venue}")
    print(f"  T1 Score: {team1_score}/{team1_wickets}")
    print(f"  → {team1} win probability   : {prob[1]*100:.1f}%")
    print(f"  → {team2} win probability   : {prob[0]*100:.1f}%")
    print(f"  → Predicted winner: {'🏆 ' + team1 if prob[1] > 0.5 else '🏆 ' + team2}")

predict_match(
    team1="Mumbai Indians", team2="Chennai Super Kings",
    venue="Wankhede Stadium, Mumbai", toss_winner="Mumbai Indians",
    toss_decision="bat", team1_score=178, team1_wickets=6,
    team1_pp=52, team2_rr=8.1
)

predict_match(
    team1="Kolkata Knight Riders", team2="Royal Challengers Bangalore",
    venue="Eden Gardens, Kolkata", toss_winner="Royal Challengers Bangalore",
    toss_decision="field", team1_score=155, team1_wickets=8,
    team1_pp=42, team2_rr=8.9
)

print("\n" + "=" * 60)
print("✅ ALL DONE! Files saved:")
for i in range(1, 9):
    fname = f"plot{i}_*.png"
print("   - ipl_matches.csv")
print("   - plot1_team_wins.png")
print("   - plot2_toss_analysis.png")
print("   - plot3_season_trends.png")
print("   - plot4_h2h_heatmap.png")
print("   - plot5_chase_success.png")
print("   - plot6_score_distribution.png")
print("   - plot7_feature_importance.png")
print("   - plot8_confusion_matrix.png")
print("=" * 60)
