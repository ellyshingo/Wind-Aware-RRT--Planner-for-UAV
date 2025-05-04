import numpy as np
import pandas as pd
from single_run import WindAwareRRTStar

def run_multiple_times(planner, start, goal, num_runs, consider_wind, consider_tke):
    metrics = []
    for run_id in range(1, num_runs + 1):
        print(f"Run {run_id} with wind={consider_wind}, tke={consider_tke}")
        result = planner.plan(start, goal, consider_wind=consider_wind, consider_tke=consider_tke)
        metrics.append({
            "Run_ID": run_id,
            "Start_X": start[0],
            "Start_Y": start[1],
            "Goal_X": goal[0],
            "Goal_Y": goal[1],
            "Consider_Wind": consider_wind,
            "Consider_TKE": consider_tke,
            "Path_Length_m": result["length"],
            "Total_Cost": result["cost"],
            "Computation_Time_s": result["time"]
        })
    
    df = pd.DataFrame(metrics)
    mean_length, std_length = df["Path_Length_m"].mean(), df["Path_Length_m"].std()
    mean_cost, std_cost = df["Total_Cost"].mean(), df["Total_Cost"].std()
    mean_time, std_time = df["Computation_Time_s"].mean(), df["Computation_Time_s"].std()
    
    print(f"Stats (wind={consider_wind}, tke={consider_tke}):")
    print(f"  Length: mean={mean_length:.2f}, std={std_length:.2f}")
    print(f"  Cost: mean={mean_cost:.2f}, std={std_cost:.2f}")
    print(f"  Time: mean={mean_time:.2f}, std={std_time:.2f}")
    
    return df

if __name__ == "__main__":
    planner = WindAwareRRTStar(
        map_size=(1300, 1000),
        obstacle_data_file="input_data/flipped_obstacle_map.json",
        wind_data="input_data/wind_50m.csv",
        tke_data="input_data/tke.csv",
        UAV_speed=8.0,
        tke_lim=0.7
    )
    start, goal = (400, 600), (800, 800)
    num_runs = 100

    # Run for different conditions
    df_wind_tke = run_multiple_times(planner, start, goal, num_runs, True, True)
    df_wind_tke.to_excel("metrics_wind_tke.xlsx", index=False)

    df_wind_only = run_multiple_times(planner, start, goal, num_runs, True, False)
    df_wind_only.to_excel("metrics_wind_only.xlsx", index=False)

    df_no_wind = run_multiple_times(planner, start, goal, num_runs, False, False)
    df_no_wind.to_excel("metrics_no_wind.xlsx", index=False)
    