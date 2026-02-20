from statsbombpy import sb
events = sb.events(match_id=8658)

print("--- pass_body_part ---")
print(events['pass_body_part'].value_counts(dropna=False))

print("\n--- pass_type ---")
print(events['pass_type'].value_counts(dropna=False))

print("\n--- pass_cross ---")
print(events['pass_cross'].value_counts(dropna=False))

print("\n--- pass_cut_back ---")
print(events['pass_cut_back'].value_counts(dropna=False))

print("\n--- pass_switch ---")
print(events['pass_switch'].value_counts(dropna=False))

print("\n--- pass_shot_assist ---")
print(events['pass_shot_assist'].value_counts(dropna=False))

print("\n--- pass_goal_assist ---")
print(events['pass_goal_assist'].value_counts(dropna=False))

print("\n--- shot_technique ---")
print(events['shot_technique'].value_counts(dropna=False))

print("\n--- shot_type ---")
print(events['shot_type'].value_counts(dropna=False))

print("\n--- shot_first_time ---")
print(events['shot_first_time'].value_counts(dropna=False))

print("\n--- duel_type ---")
print(events['duel_type'].value_counts(dropna=False))

print("\n--- dribble_outcome ---")
print(events['dribble_outcome'].value_counts(dropna=False))

print("\n--- interception_outcome ---")
print(events['interception_outcome'].value_counts(dropna=False))