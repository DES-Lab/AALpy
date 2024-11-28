import pickle
import os

resultsdir = "results"

if not os.path.exists(resultsdir):
    print(f"No {resultsdir} directory. Run mealy_conformance_testing.py first.")

with open(resultsdir + '/results-pd.pkl', 'rb') as f:
    # this is a pandas dataframe
    results_df = pickle.load(f)

print(results_df)

for index, row in results_df.iterrows():
    truths = row.value_counts()[True]
    falses = len(row) - truths
    print(f"{index} learned correctly {truths} times and failed {falses} times")
