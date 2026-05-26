import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

teams = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Sunrisers Hyderabad",
    "Rajasthan Royals", "Punjab Kings"
]

venues = [
    "Wankhede Stadium, Mumbai", "M. A. Chidambaram Stadium, Chennai",
    "M. Chinnaswamy Stadium, Bangalore", "Eden Gardens, Kolkata",
    "Arun Jaitley Stadium, Delhi", "Rajiv Gandhi Intl. Cricket Stadium, Hyderabad",
    "Sawai Mansingh Stadium, Jaipur", "Punjab Cricket Association Stadium, Mohali"
]

team_venue_home = {
    "Mumbai Indians": "Wankhede Stadium, Mumbai",
    "Chennai Super Kings": "M. A. Chidambaram Stadium, Chennai",
    "Royal Challengers Bangalore": "M. Chinnaswamy Stadium, Bangalore",
    "Kolkata Knight Riders": "Eden Gardens, Kolkata",
    "Delhi Capitals": "Arun Jaitley Stadium, Delhi",
    "Sunrisers Hyderabad": "Rajiv Gandhi Intl. Cricket Stadium, Hyderabad",
    "Rajasthan Royals": "Sawai Mansingh Stadium, Jaipur",
    "Punjab Kings": "Punjab Cricket Association Stadium, Mohali"
}

# Team strength weights (higher = stronger team historically)
team_strength = {
    "Mumbai Indians": 0.62,
    "Chennai Super Kings": 0.60,
    "Kolkata Knight Riders": 0.52,
    "Royal Challengers Bangalore": 0.48,
    "Sunrisers Hyderabad": 0.50,
    "Delhi Capitals": 0.47,
    "Rajasthan Royals": 0.49,
    "Punjab Kings": 0.44
}

rows = []
match_id = 1

for year in range(2008, 2024):
    season_teams = random.sample(teams, 8)
    num_matches = 60

    for _ in range(num_matches):
        team1, team2 = random.sample(season_teams, 2)
        venue = random.choice([team_venue_home[team1], team_venue_home[team2], random.choice(venues)])
        toss_winner = random.choice([team1, team2])
        toss_decision = random.choice(["bat", "field"])

        # First innings score
        team1_score = int(np.random.normal(165, 18))
        team1_wickets = random.randint(4, 10)
        team1_powerplay = int(np.random.normal(48, 6))
        team1_rr = round(team1_score / 20, 2)

        # Second innings — biased by team strength & home advantage
        home_boost = 0.04 if venue == team_venue_home[team2] else 0
        base_prob = team_strength[team2] + home_boost - team_strength[team1] * 0.3
        chase_success = random.random() < (0.48 + base_prob * 0.3)

        if chase_success:
            team2_score = team1_score + random.randint(1, 15)
            team2_wickets = random.randint(3, 9)
            winner = team2
        else:
            team2_score = team1_score - random.randint(1, 25)
            team2_wickets = random.randint(6, 10)
            winner = team1

        team2_powerplay = int(np.random.normal(46, 7))
        team2_rr = round(team2_score / 20, 2)

        rows.append({
            "match_id": match_id,
            "season": year,
            "venue": venue,
            "team1": team1,
            "team2": team2,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "team1_score": team1_score,
            "team1_wickets": team1_wickets,
            "team1_powerplay_runs": team1_powerplay,
            "team1_run_rate": team1_rr,
            "team2_score": team2_score,
            "team2_wickets": team2_wickets,
            "team2_powerplay_runs": team2_powerplay,
            "team2_run_rate": team2_rr,
            "winner": winner,
            "win_by_runs": max(0, team1_score - team2_score) if winner == team1 else 0,
            "win_by_wickets": max(0, 10 - team2_wickets) if winner == team2 else 0,
        })
        match_id += 1

df = pd.DataFrame(rows)
df.to_csv("/home/claude/ipl_matches.csv", index=False)
print(f"Dataset created: {len(df)} matches, {df['season'].nunique()} seasons")
print(df.head(3).to_string())
