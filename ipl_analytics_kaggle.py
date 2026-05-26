"""
IPL Match Analytics & Win Predictor — UPGRADED (Real Kaggle Data)
==================================================================
Dataset : Kaggle — IPL Complete Dataset 2008–2020
Files   : matches.csv, deliveries.csv
Author  : Nancy Gautam

How to run:
    pip install pandas numpy matplotlib seaborn scikit-learn
    python ipl_analytics_kaggle.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
import warnings, os
warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
    "axes.titlesize"   : 13,
    "axes.titleweight" : "bold",
    "axes.labelsize"   : 11,
})
PALETTE = ["#1F4E79","#2E75B6","#70AD47","#ED7D31",
           "#FFC000","#FF4C4C","#7030A0","#00B0F0"]
OUTPUT_DIR = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — LOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  PHASE 1 — Loading & Cleaning Real Kaggle IPL Data")
print("=" * 65)

# ── Load ──────────────────────────────────────────────────────────────────────
try:
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")
    print(f"\n✅ matches.csv     : {matches.shape[0]} rows × {matches.shape[1]} cols")
    print(f"✅ deliveries.csv  : {deliveries.shape[0]} rows × {deliveries.shape[1]} cols")
except FileNotFoundError as e:
    print(f"\n❌ File not found: {e}")
    print("   Please put matches.csv and deliveries.csv in this folder.")
    exit()

# ── Standardise team names (Kaggle has renamed teams) ─────────────────────────
rename_map = {
    "Delhi Daredevils"               : "Delhi Capitals",
    "Deccan Chargers"                : "Sunrisers Hyderabad",
    "Rising Pune Supergiant"         : "Rising Pune Supergiants",
    "Kings XI Punjab"                : "Punjab Kings",
    "Pune Warriors"                  : "Pune Warriors India",
}
for col in ["team1", "team2", "toss_winner", "winner",
            "batting_team", "bowling_team"]:
    if col in matches.columns:
        matches[col] = matches[col].replace(rename_map)
for col in ["batting_team", "bowling_team"]:
    if col in deliveries.columns:
        deliveries[col] = deliveries[col].replace(rename_map)

# ── Drop irrelevant / high-null columns ───────────────────────────────────────
drop_cols = ["umpire1","umpire2","umpire3","id"]
matches.drop(columns=[c for c in drop_cols if c in matches.columns],
             inplace=True, errors="ignore")

# ── Fill / fix ─────────────────────────────────────────────────────────────────
matches["winner"].fillna("No Result", inplace=True)
matches = matches[matches["winner"] != "No Result"].copy()
matches["season"] = matches["season"].astype(str).apply(lambda x: int(x.split("/")[0]) if "/" in x else int(x))

print(f"\nAfter cleaning  : {matches.shape[0]} matches | "
      f"{matches['season'].nunique()} seasons "
      f"({matches['season'].min()}–{matches['season'].max()})")
print(f"Teams           : {sorted(matches['team1'].unique())}")
print(f"\nMissing values:\n{matches.isnull().sum()[matches.isnull().sum()>0]}")

# ── Derived columns ────────────────────────────────────────────────────────────
matches["toss_won_match"] = (matches["toss_winner"] == matches["winner"]).astype(int)
matches["chased_win"]     = (matches["winner"] == matches["team2"]).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PHASE 2 — Exploratory Data Analysis")
print("=" * 65)

# Top teams by total wins
top_teams = [t for t in matches["winner"].value_counts().head(8).index]

# ── Plot 1: Total wins per team ───────────────────────────────────────────────
win_counts = (matches[matches["winner"].isin(top_teams)]
              ["winner"].value_counts().sort_values())
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(win_counts.index, win_counts.values,
               color=PALETTE[:len(win_counts)], edgecolor="white", height=0.65)
for bar, val in zip(bars, win_counts.values):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=10)
ax.set_xlabel("Total Wins")
ax.set_title("IPL — Total Wins by Team (2008–2020)")
ax.set_facecolor("#F8F9FA")
save(f"{OUTPUT_DIR}/01_team_wins.png")

# ── Plot 2: Wins per season (stacked) ────────────────────────────────────────
season_wins = (matches[matches["winner"].isin(top_teams)]
               .groupby(["season","winner"])
               .size().unstack(fill_value=0))
season_wins.plot(kind="bar", stacked=True, figsize=(12, 5),
                 color=PALETTE, edgecolor="white", width=0.7)
plt.title("Wins per Season by Top Teams")
plt.xlabel("Season"); plt.ylabel("Wins")
plt.legend(loc="upper right", fontsize=8, ncol=2)
plt.xticks(rotation=45)
save(f"{OUTPUT_DIR}/02_wins_per_season.png")

# ── Plot 3: Toss decision & impact ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
td = matches["toss_decision"].value_counts()
axes[0].pie(td.values, labels=td.index, autopct="%1.1f%%",
            colors=["#1F4E79","#70AD47"], startangle=140,
            wedgeprops={"edgecolor":"white","linewidth":2})
axes[0].set_title("Toss Decision Distribution")

ti = matches["toss_won_match"].value_counts()
axes[1].bar(["Lost Match","Won Match"], ti.values,
            color=["#ED7D31","#1F4E79"], edgecolor="white", width=0.5)
axes[1].set_title("Did Toss Winner Win the Match?")
axes[1].set_ylabel("Number of Matches")
pct = ti[1] / ti.sum() * 100
axes[1].text(0.5, -0.15, f"Toss winner won {pct:.1f}% of matches",
             ha="center", transform=axes[1].transAxes,
             fontsize=10, color="gray")
for i, v in enumerate(ti.values):
    axes[1].text(i, v+2, str(v), ha="center", fontsize=11, fontweight="bold")
save(f"{OUTPUT_DIR}/03_toss_analysis.png")

# ── Plot 4: Venue win % (bat vs field first) ──────────────────────────────────
venue_toss = (matches.groupby(["venue","toss_decision"])
              ["toss_won_match"].mean().unstack(fill_value=0) * 100)
top_venues = matches["venue"].value_counts().head(10).index
venue_toss = venue_toss.loc[venue_toss.index.isin(top_venues)]
venue_toss.plot(kind="barh", figsize=(10, 6), color=["#1F4E79","#ED7D31"],
                edgecolor="white", width=0.65)
plt.title("Win % by Toss Decision at Top Venues")
plt.xlabel("Win %"); plt.ylabel("")
plt.legend(title="Toss Decision")
plt.axvline(50, color="gray", linestyle="--", linewidth=1)
save(f"{OUTPUT_DIR}/04_venue_toss_impact.png")

# ── Plot 5: Chase success rate by team ───────────────────────────────────────
chase_rate = (matches[matches["team2"].isin(top_teams)]
              .groupby("team2")["chased_win"].mean()
              .sort_values() * 100)
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#ED7D31" if v < 50 else "#1F4E79" for v in chase_rate.values]
ax.barh(chase_rate.index, chase_rate.values, color=colors, height=0.6, edgecolor="white")
ax.axvline(50, color="gray", linestyle="--", linewidth=1.2, label="50% line")
for i, v in enumerate(chase_rate.values):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=10)
ax.set_xlabel("Chase Success Rate (%)")
ax.set_title("Chase Success Rate by Team")
ax.legend()
save(f"{OUTPUT_DIR}/05_chase_success.png")

# ── Plot 6: Head-to-head heatmap (top 8 teams) ───────────────────────────────
ht = matches[matches["team1"].isin(top_teams) & matches["team2"].isin(top_teams)]
matrix = pd.DataFrame(0.0, index=top_teams, columns=top_teams)
counts = pd.DataFrame(0,   index=top_teams, columns=top_teams)
for _, r in ht.iterrows():
    counts.loc[r.team1, r.team2] += 1
    counts.loc[r.team2, r.team1] += 1
    if r.winner == r.team1:
        matrix.loc[r.team1, r.team2] += 1
    elif r.winner == r.team2:
        matrix.loc[r.team2, r.team1] += 1
win_pct = (matrix / counts.replace(0, np.nan) * 100).fillna(0).round(0)
fig, ax = plt.subplots(figsize=(10, 8))
short = [t.split()[0] for t in top_teams]
sns.heatmap(win_pct, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, annot_kws={"size":10},
            xticklabels=short, yticklabels=short,
            cbar_kws={"label":"Win %"})
ax.set_title("Head-to-Head Win % (Row beats Column)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
save(f"{OUTPUT_DIR}/06_h2h_heatmap.png")

# ── Plot 7: Powerplay analysis from deliveries ────────────────────────────────
pp = deliveries[deliveries["over"] <= 6].copy()
pp_runs = (pp.groupby(["match_id","batting_team"])["total_runs"]
           .sum().reset_index()
           .rename(columns={"total_runs":"pp_runs"}))
pp_runs = pp_runs[pp_runs["batting_team"].isin(top_teams)]
pp_avg = pp_runs.groupby("batting_team")["pp_runs"].mean().sort_values()
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(pp_avg.index, pp_avg.values,
               color=PALETTE[:len(pp_avg)], edgecolor="white", height=0.65)
for bar, val in zip(bars, pp_avg.values):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}", va="center", fontsize=10)
ax.set_xlabel("Avg Powerplay Runs (Overs 1–6)")
ax.set_title("Average Powerplay Runs by Team")
save(f"{OUTPUT_DIR}/07_powerplay_runs.png")

# ── Plot 8: Top run scorers ───────────────────────────────────────────────────
top_batsmen = (deliveries.groupby("batter")["batsman_runs"]
               .sum().sort_values(ascending=False).head(12))
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(top_batsmen.index[::-1], top_batsmen.values[::-1],
               color="#1F4E79", edgecolor="white", height=0.65)
for bar, val in zip(bars, top_batsmen.values[::-1]):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=9)
ax.set_xlabel("Total Runs")
ax.set_title("Top 12 Run Scorers in IPL History")
save(f"{OUTPUT_DIR}/08_top_batsmen.png")

# ── Plot 9: Top wicket takers ─────────────────────────────────────────────────
wkts = deliveries[deliveries["dismissal_kind"].notna() &
                  ~deliveries["dismissal_kind"].isin(
                      ["run out","retired hurt","obstructing the field"])]
top_bowlers = (wkts.groupby("bowler")["player_dismissed"]
               .count().sort_values(ascending=False).head(12))
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(top_bowlers.index[::-1], top_bowlers.values[::-1],
        color="#ED7D31", edgecolor="white", height=0.65)
for i, (idx, val) in enumerate(zip(top_bowlers.index[::-1],
                                    top_bowlers.values[::-1])):
    ax.text(val + 1, i, str(val), va="center", fontsize=9)
ax.set_xlabel("Total Wickets")
ax.set_title("Top 12 Wicket Takers in IPL History")
save(f"{OUTPUT_DIR}/09_top_bowlers.png")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — FEATURE ENGINEERING & ML MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PHASE 3 — Feature Engineering & ML Models")
print("=" * 65)

# ── 3A: Compute powerplay runs per match per team ─────────────────────────────
pp_lookup = pp_runs.set_index(["match_id","batting_team"])["pp_runs"].to_dict()

# ── 3B: Compute historical win rate (up to that match — no data leakage) ──────
matches_sorted = matches.sort_values(["season","id"] if "id" in matches.columns
                                     else ["season"]).reset_index(drop=True)

win_rate_t1, win_rate_t2 = [], []
team_stats = {}   # {team: [wins, total]}

for _, row in matches_sorted.iterrows():
    t1, t2 = row["team1"], row["team2"]
    for t in [t1, t2]:
        if t not in team_stats:
            team_stats[t] = [0, 0]
    wr1 = team_stats[t1][0]/team_stats[t1][1] if team_stats[t1][1]>0 else 0.5
    wr2 = team_stats[t2][0]/team_stats[t2][1] if team_stats[t2][1]>0 else 0.5
    win_rate_t1.append(wr1)
    win_rate_t2.append(wr2)
    team_stats[t1][1] += 1; team_stats[t2][1] += 1
    if row["winner"] == t1:
        team_stats[t1][0] += 1
    elif row["winner"] == t2:
        team_stats[t2][0] += 1

matches_sorted["t1_win_rate"] = win_rate_t1
matches_sorted["t2_win_rate"] = win_rate_t2
matches_sorted["win_rate_diff"] = (matches_sorted["t1_win_rate"]
                                   - matches_sorted["t2_win_rate"])

# ── 3C: Encode categoricals ───────────────────────────────────────────────────
le = {}
for col in ["team1","team2","toss_winner","venue"]:
    le[col] = LabelEncoder()
    all_vals = pd.concat([matches_sorted["team1"],
                          matches_sorted["team2"]]).unique() if col in ["team1","team2","toss_winner"] \
               else matches_sorted["venue"].unique()
    le[col].fit(all_vals)
    matches_sorted[col+"_enc"] = le[col].transform(matches_sorted[col])

matches_sorted["toss_bat_first"] = (matches_sorted["toss_decision"]=="bat").astype(int)
matches_sorted["toss_winner_enc"] = le["team1"].transform(
    matches_sorted["toss_winner"].where(
        matches_sorted["toss_winner"].isin(le["team1"].classes_), other=matches_sorted["team1"]))

# Powerplay feature
matches_sorted["t1_pp"] = matches_sorted.apply(
    lambda r: pp_lookup.get((r.get("id", r.name), r["team1"]), 48), axis=1)

# Target
matches_sorted["team1_won"] = (matches_sorted["winner"] == matches_sorted["team1"]).astype(int)

FEATURES = ["team1_enc","team2_enc","venue_enc",
            "toss_bat_first","toss_winner_enc",
            "t1_win_rate","t2_win_rate","win_rate_diff","t1_pp"]

df_model = matches_sorted.dropna(subset=FEATURES+["team1_won"])
X = df_model[FEATURES]
y = df_model["team1_won"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── Logistic Regression ────────────────────────────────────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc  = accuracy_score(y_test, lr_pred)

# ── Random Forest ──────────────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)

print(f"\n  Logistic Regression Accuracy : {lr_acc*100:.2f}%")
print(f"  Random Forest Accuracy       : {rf_acc*100:.2f}%")
print(f"\n  Best model: {'Random Forest' if rf_acc > lr_acc else 'Logistic Regression'}")
print(f"\n  Classification Report (Random Forest):")
print(classification_report(y_test, rf_pred,
                             target_names=["Team2 Wins","Team1 Wins"]))

best_model = rf if rf_acc >= lr_acc else lr
best_pred  = rf_pred if rf_acc >= lr_acc else lr_pred

# ── Plot 10: Model comparison ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Confusion matrix
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Team2 Wins","Team1 Wins"],
            yticklabels=["Team2 Wins","Team1 Wins"],
            linewidths=0.5, annot_kws={"size":13})
axes[0].set_title(f"Confusion Matrix\n(Best model acc: {max(lr_acc,rf_acc)*100:.1f}%)")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

# Model comparison bar
models  = ["Logistic\nRegression","Random\nForest"]
accs    = [lr_acc*100, rf_acc*100]
colors  = ["#2E75B6","#70AD47"]
bars = axes[1].bar(models, accs, color=colors, edgecolor="white", width=0.45)
for bar, v in zip(bars, accs):
    axes[1].text(bar.get_x()+bar.get_width()/2, v+0.3,
                 f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")
axes[1].set_ylim(0, 110)
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Model Accuracy Comparison")
axes[1].axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5)
save(f"{OUTPUT_DIR}/10_model_comparison.png")

# ── Plot 11: Feature Importance (Random Forest) ───────────────────────────────
feat_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
feat_imp.plot(kind="barh", ax=ax, color="#1F4E79", edgecolor="white")
ax.set_title("Feature Importance — Random Forest")
ax.set_xlabel("Importance Score")
save(f"{OUTPUT_DIR}/11_feature_importance.png")

# ── Plot 12: ROC Curve ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
for model, name, color in [(lr,"Logistic Regression","#2E75B6"),
                            (rf,"Random Forest","#70AD47")]:
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})", color=color, lw=2)
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Win Predictor")
ax.legend(loc="lower right")
save(f"{OUTPUT_DIR}/12_roc_curve.png")

# ══════════════════════════════════════════════════════════════════════════════
# LIVE MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  DEMO — Live Match Win Predictor")
print("=" * 65)

def predict_match(team1, team2, venue, toss_winner, toss_decision, pp_runs=48):
    try:
        t1e = le["team1"].transform([team1])[0]
        t2e = le["team2"].transform([team2])[0] if team2 in le["team2"].classes_ \
              else le["team1"].transform([team2])[0]
        ve  = le["venue"].transform([venue])[0] if venue in le["venue"].classes_ else 0
        twe = le["team1"].transform([toss_winner])[0] if toss_winner in le["team1"].classes_ else t1e
        bat = 1 if toss_decision == "bat" else 0
        wr1 = team_stats.get(team1,[0,1])
        wr2 = team_stats.get(team2,[0,1])
        w1  = wr1[0]/wr1[1] if wr1[1]>0 else 0.5
        w2  = wr2[0]/wr2[1] if wr2[1]>0 else 0.5
        row = [[t1e, t2e, ve, bat, twe, w1, w2, w1-w2, pp_runs]]
        prob = best_model.predict_proba(row)[0]
        print(f"\n  {team1}  vs  {team2}")
        print(f"  Venue       : {venue}")
        print(f"  Toss        : {toss_winner} chose to {toss_decision}")
        print(f"  T1 Powerplay: {pp_runs} runs")
        print(f"  ──────────────────────────────────")
        print(f"  {team1:<30} {prob[1]*100:.1f}%")
        print(f"  {team2:<30} {prob[0]*100:.1f}%")
        winner = team1 if prob[1] > 0.5 else team2
        print(f"  🏆 Predicted Winner: {winner}")
    except Exception as e:
        print(f"  ⚠️  Prediction error: {e}")

predict_match("Mumbai Indians","Chennai Super Kings",
              "Wankhede Stadium, Mumbai","Mumbai Indians","bat", 55)
predict_match("Kolkata Knight Riders","Royal Challengers Bangalore",
              "Eden Gardens, Kolkata","Royal Challengers Bangalore","field", 42)
predict_match("Delhi Capitals","Rajasthan Royals",
              "Arun Jaitley Stadium, Delhi","Delhi Capitals","field", 48)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"  ✅ PROJECT COMPLETE — 12 plots saved in '{OUTPUT_DIR}/' folder")
print("=" * 65)
