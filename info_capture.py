# The specific features crucial for your GAT & Tactical Suite
from statsbombpy import sb
target_features = [
    'play_pattern',   # Critical for "Offensive Style"
    'position',       # Critical for Node Features (Who has the ball?)
    'pass_height',    # Critical for Edge Features (Ground vs High pass)
    'duel_type',      # Critical for "Defensive Intensity" details
    'type',           # Overview of all event types
    'under_pressure', # Critical for Graph Weights
    'shot_outcome'    # To verify we have Goals vs Saves vs Misses
]
match_id = 8658

events = sb.events(match_id)

print(f"Inspecting features for Match ID: {match_id}\n")

for col in target_features:
    if col in events.columns:
        print(f"--- {col.upper()} ---")
        # dropna=False ensures we see how many missing values exist (crucial for filtering)
        print(events[col].value_counts(dropna=False)) 
        print("\n" + "="*40 + "\n")
    else:
        print(f"(!) Column '{col}' not found in this match.\n")