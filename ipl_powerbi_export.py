"""
IPL Power BI Data Exporter
===========================
Yeh script matches.csv aur deliveries.csv se
Power BI ke liye 4 clean tables export karega:

1. ipl_matches_clean.csv      → Main matches table
2. ipl_team_stats.csv         → Team-wise win/loss stats
3. ipl_top_players.csv        → Top batsmen & bowlers
4. ipl_season_summary.csv     → Season-wise summary
"""

import pandas as pd
import numpy as np
import os

print("=" * 55)
print("  IPL — Power BI Data Exporter")
print("=" * 55)

# ── Load ───────────────────────────────────────────────────
matches     = pd.read_csv("matches.csv")
deliveries  = pd.read_csv("deliveries.csv")

# ── Fix season format ──────────────────────────────────────
matches["season"] = matches["season"].astype(str).apply(
    lambda x: int(x.split("/")[0]) if "/" in x else int(x))

# ── Standardise team names ─────────────────────────────────
rename_map = {
    "Delhi Daredevils"          : "Delhi Capitals",
    "Deccan Chargers"           : "Sunrisers Hyderabad",
    "Rising Pune Supergiant"    : "Rising Pune Supergiants",
    "Kings XI Punjab"           : "Punjab Kings",
    "Pune Warriors"             : "Pune Warriors India",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
}
for col in ["team1","team2","toss_winner","winner"]:
    matches[col] = matches[col].replace(rename_map)
for col in ["batting_team","bowling_team"]:
    if col in deliveries.columns:
        deliveries[col] = deliveries[col].replace(rename_map)

matches = matches[matches["winner"].notna() &
                  (matches["winner"] != "")].copy()

# ══════════════════════════════════════════════════════════
# TABLE 1 — Clean Matches
# ══════════════════════════════════════════════════════════
matches_clean = matches[[
    "id","season","date","venue","city",
    "team1","team2","toss_winner","toss_decision",
    "winner","result_margin","player_of_match"
]].copy()

matches_clean.rename(columns={
    "id"            : "match_id",
    "result_margin" : "win_margin",
}, inplace=True)

matches_clean["toss_won_match"] = (
    matches_clean["toss_winner"] == matches_clean["winner"]).astype(int)
matches_clean["chased_successfully"] = (
    matches_clean["winner"] == matches_clean["team2"]).astype(int)
matches_clean["date"] = pd.to_datetime(matches_clean["date"], errors="coerce")

matches_clean.to_csv("ipl_matches_clean.csv", index=False)
print(f"\n✅ Table 1 saved: ipl_matches_clean.csv  ({len(matches_clean)} rows)")

# ══════════════════════════════════════════════════════════
# TABLE 2 — Team Stats
# ══════════════════════════════════════════════════════════
all_teams = pd.concat([matches["team1"], matches["team2"]]).unique()
rows = []
for team in sorted(all_teams):
    played = len(matches[(matches["team1"]==team)|(matches["team2"]==team)])
    wins   = len(matches[matches["winner"]==team])
    losses = played - wins
    win_pct= round(wins/played*100, 1) if played > 0 else 0

    # Toss stats
    toss_won   = len(matches[matches["toss_winner"]==team])
    toss_match = len(matches[(matches["toss_winner"]==team) &
                              (matches["winner"]==team)])

    # Chase stats
    chases     = len(matches[matches["team2"]==team])
    chase_wins = len(matches[(matches["team2"]==team) &
                              (matches["winner"]==team)])
    chase_pct  = round(chase_wins/chases*100,1) if chases>0 else 0

    rows.append({
        "team"          : team,
        "matches_played": played,
        "wins"          : wins,
        "losses"        : losses,
        "win_pct"       : win_pct,
        "toss_wins"     : toss_won,
        "toss_then_match_wins": toss_match,
        "chases_played" : chases,
        "chase_wins"    : chase_wins,
        "chase_win_pct" : chase_pct,
    })

team_stats = pd.DataFrame(rows).sort_values("wins", ascending=False)
team_stats.to_csv("ipl_team_stats.csv", index=False)
print(f"✅ Table 2 saved: ipl_team_stats.csv      ({len(team_stats)} rows)")

# ══════════════════════════════════════════════════════════
# TABLE 3 — Top Players
# ══════════════════════════════════════════════════════════
# Batsmen
batter_col = "batter" if "batter" in deliveries.columns else "batsman"
top_batsmen = (deliveries.groupby(batter_col)
               .agg(
                   total_runs   =("batsman_runs","sum"),
                   innings      =(batter_col,"count"),
                   fours        =("batsman_runs", lambda x: (x==4).sum()),
                   sixes        =("batsman_runs", lambda x: (x==6).sum()),
               )
               .reset_index()
               .rename(columns={batter_col:"player"}))
top_batsmen["role"] = "Batsman"
top_batsmen = top_batsmen.sort_values("total_runs", ascending=False).head(20)

# Bowlers
wkts = deliveries[
    deliveries["dismissal_kind"].notna() &
    ~deliveries["dismissal_kind"].isin(
        ["run out","retired hurt","obstructing the field"])
]
top_bowlers = (wkts.groupby("bowler")
               .agg(wickets=("player_dismissed","count"))
               .reset_index()
               .rename(columns={"bowler":"player"}))
top_bowlers["role"]       = "Bowler"
top_bowlers["total_runs"] = 0
top_bowlers["innings"]    = 0
top_bowlers["fours"]      = 0
top_bowlers["sixes"]      = 0
top_bowlers = top_bowlers.sort_values("wickets", ascending=False).head(20)

top_players = pd.concat([top_batsmen, top_bowlers], ignore_index=True)
top_players.to_csv("ipl_top_players.csv", index=False)
print(f"✅ Table 3 saved: ipl_top_players.csv     ({len(top_players)} rows)")

# ══════════════════════════════════════════════════════════
# TABLE 4 — Season Summary
# ══════════════════════════════════════════════════════════
season_summary = (matches.groupby("season")
                  .agg(
                      total_matches = ("id","count"),
                      unique_venues = ("venue","nunique"),
                      champion      = ("winner", lambda x: x.value_counts().index[0]),
                  )
                  .reset_index())

# Avg margin
margin = matches.groupby("season")["result_margin"].mean().reset_index()
margin.columns = ["season","avg_win_margin"]
season_summary = season_summary.merge(margin, on="season")
season_summary["avg_win_margin"] = season_summary["avg_win_margin"].round(1)

season_summary.to_csv("ipl_season_summary.csv", index=False)
print(f"✅ Table 4 saved: ipl_season_summary.csv  ({len(season_summary)} rows)")

print(f"""
{'='*55}
  ✅ ALL 4 FILES READY FOR POWER BI!

  Files created in D:\\IPL_Project\\:
  → ipl_matches_clean.csv
  → ipl_team_stats.csv
  → ipl_top_players.csv
  → ipl_season_summary.csv
{'='*55}
""")
