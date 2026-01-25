import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Import your pipeline modules
from window_slicer import get_rolling_windows
from data_pipeline import fetch_match_data
from utils import encode_features

# --- CONFIGURATION ---
MATCH_ID = 8658  # World Cup Final 2018
WINDOW_SIZE = 5  # 5 Minute Windows (High Fidelity)
STRIDE = 1       # 1 Minute Steps

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

def draw_pitch(ax):
    """Draws a basic football pitch layout on the given axes."""
    # Pitch Outline & Centre Line
    plt.plot([0, 0], [0, 80], color="black")
    plt.plot([0, 120], [80, 80], color="black")
    plt.plot([120, 120], [80, 0], color="black")
    plt.plot([120, 0], [0, 0], color="black")
    plt.plot([60, 60], [0, 80], color="black")
    
    # Centre Circle
    circle = plt.Circle((60, 40), 9.15, color="black", fill=False)
    ax.add_patch(circle)
    
    # Penalty Areas (Optional simple boxes)
    plt.plot([0, 18], [18, 18], color="black")
    plt.plot([18, 18], [18, 62], color="black")
    plt.plot([18, 0], [62, 62], color="black")
    
    plt.plot([120, 102], [18, 18], color="black")
    plt.plot([102, 102], [18, 62], color="black")
    plt.plot([102, 120], [62, 62], color="black")

def plot_goal_chain(passes_df, chain_id, goal_info):
    """Visualizes the specific possession chain leading to a goal."""
    # Filter for the specific chain
    if 'possession' not in passes_df.columns:
        print("Error: 'possession' column missing from passes. Cannot link chain.")
        return

    chain = passes_df[passes_df['possession'] == chain_id].sort_values('time_min')
    
    if chain.empty:
        print(f"No passing chain found for Possession ID {chain_id} (Direct FK/Penalty/Steal?)")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    draw_pitch(ax)
    
    # Plot Arrows for the chain
    for i, p in chain.iterrows():
        # Arrow logic
        ax.arrow(p['x'], p['y'], p['end_x']-p['x'], p['end_y']-p['y'],
                 head_width=2, head_length=2, fc='blue', ec='blue', 
                 length_includes_head=True, alpha=0.8)
        
        # Label player name at start of pass
        ax.text(p['x'], p['y'], f"{p['player']}", fontsize=9, color='red', fontweight='bold')

    # Add Goal Info Title
    scorer = goal_info['player']
    time_str = f"{int(goal_info['minute'])}:{int(goal_info['second']):02d}"
    plt.title(f"GOAL! {scorer} ({time_str}) - Possession Chain {chain_id}\nTeam: {goal_info['team']}")
    
    plt.xlim(-5, 125)
    plt.ylim(-5, 85)
    plt.show()

def print_chain_commentary(passes_df, chain_id):
    """Prints text logs for a specific possession chain."""
    chain = passes_df[passes_df['possession'] == chain_id].sort_values('time_min')
    
    if chain.empty:
        print("   (No passes recorded in this possession chain)")
        return

    print(f"   --- Build-up Play (Chain {chain_id}) ---")
    for _, p in chain.iterrows():
        sender = p['player']
        receiver = p['pass_recipient'] if pd.notna(p['pass_recipient']) else "Space"
        time_s = f"{int(p['minute'])}:{int(p['second']):02d}"
        
        # Context (Outcome)
        outcome = "" 
        if 'pass_outcome' in p and pd.notna(p['pass_outcome']):
            outcome = f"({p['pass_outcome']})"
            
        print(f"   [{time_s}] {sender} -> {receiver} {outcome}")
    print("\n")

def find_goal_chain_id(goal_row, passes_df):
    """
    Finds the possession ID associated with a goal.
    Logic: Looks for the last pass by the scoring team immediately before the goal.
    """
    goal_time = goal_row['time_min']
    scoring_team = goal_row['team']
    
    # 1. Filter passes by scoring team that happened BEFORE the goal
    relevant_passes = passes_df[
        (passes_df['team'] == scoring_team) & 
        (passes_df['time_min'] <= goal_time)
    ].sort_values('time_min')
    
    if relevant_passes.empty:
        return None
    
    # 2. Return the possession ID of the very last pass
    return relevant_passes.iloc[-1]['possession']

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"1. Fetching Match {MATCH_ID} Data...")
    raw = fetch_match_data(MATCH_ID)
    
    # Encode features (Good practice, ensuring structure matches training)
    if not raw['passes'].empty:
        raw['passes'] = encode_features(raw['passes'])

    print(f"2. Slicing Windows (Size: {WINDOW_SIZE}m, Stride: {STRIDE}m)...")
    # Note: We pass match_id as discussed in previous steps
    windows = get_rolling_windows(raw, MATCH_ID, window_size=WINDOW_SIZE, stride=STRIDE)
    
    print(f"3. Scanning {len(windows)} windows for Goals...")
    
    # Track printed goals to avoid spamming exactly the same visual 5 times
    # (Optional: Remove this set if you WANT to see every overlapping window's perspective)
    processed_goals = set() 

    for w in windows:
        # Check if any goals happened in this window
        shots = w['shots']
        goals = shots[shots['is_goal'] == True]
        
        if not goals.empty:
            for _, goal in goals.iterrows():
                
                # Unique ID for this goal event (Minute + Scorer) to handle duplicates
                goal_sig = (goal['minute'], goal['player'])
                
                # NOTE: Comment out this 'if' block if you strictly want MULTIPLE graphs per goal
                # as requested ("multiple graph for same goal is okay"). 
                # I left it enabled to show it once per unique goal for clarity, 
                # but you can delete these 2 lines to see it in every slice.
                if goal_sig in processed_goals: continue
                processed_goals.add(goal_sig)
                
                print("="*60)
                print(f"⚽ GOAL DETECTED: {goal['player']} ({goal['team']})")
                print(f"   Time: {int(goal['minute'])}:{int(goal['second']):02d}")
                print(f"   Window ID: {w['window_id']} ({w['start_time']} - {w['end_time']} min)")
                
                # 1. Identify the Chain
                chain_id = find_goal_chain_id(goal, w['passes'])
                
                if chain_id:
                    # 2. Print Commentary
                    print_chain_commentary(w['passes'], chain_id)
                    
                    # 3. Visualize Graph
                    plot_goal_chain(w['passes'], chain_id, goal)
                else:
                    print("   [!] Could not link goal to a passing chain (Direct Set Piece or Steal).")
                
                print("="*60 + "\n")