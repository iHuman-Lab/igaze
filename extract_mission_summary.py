import os

import pandas as pd
import pyxdf


def extract_mission_endpoints(xdf_path):
    streams, _ = pyxdf.load_xdf(xdf_path)
    # Find the stream with behavioral data (assuming it's the first non-marker stream)
    for stream in streams:
        if "saved_victims" in str(stream) and "steps_counts" in str(stream):
            data = pd.DataFrame(stream["time_series"], columns=[ch["label"][0] for ch in stream["info"]["desc"][0]["channels"][0]["channel"]])
            break
    else:
        return []
    # Identify mission change points
    changes = (data[["llm_model", "llm_provider", "prompt_type"]].shift() != data[["llm_model", "provider", "prompt_type"]]).any(axis=1)
    mission_starts = data.index[changes].tolist()
    mission_starts.append(len(data))  # Add end of last mission
    results = []
    for i in range(len(mission_starts)-1):
        end_idx = mission_starts[i+1]-1
        row = data.iloc[end_idx]
        results.append({
            "llm_model": row["llm_model"],
            "provider": row["provider"],
            "prompt_type": row["prompt_type"],
            "saved_victims": row["saved_victims"],
            "steps_counts": row["steps_counts"],
        })
    return results

def process_all_participants(data_dir):
    summary = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".xdf"):
            participant = fname.split("_")[0]
            missions = extract_mission_endpoints(os.path.join(data_dir, fname))
            for m in missions:
                m["participant"] = participant
                summary.append(m)
    df = pd.DataFrame(summary)
    df.to_csv("mission_summary.csv", index=False)
    # Aggregate across all participants
    agg = df.groupby(["llm_model", "provider", "prompt_type"]).agg({"saved_victims":"sum", "steps_counts":"sum"}).reset_index()
    agg.to_csv("mission_summary_aggregate.csv", index=False)

if __name__ == "__main__":
    process_all_participants("data")
    print("Summary files written: mission_summary.csv, mission_summary_aggregate.csv")
