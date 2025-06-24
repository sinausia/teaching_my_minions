import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def load_and_preprocess(xlsx_path, sheet_name="First9_Points_Fits"):
    """
    Load the specified Excel sheet and ensure 'Parameter_b_Corrected' exists.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if 'Parameter_b_Corrected' not in df.columns:
        df['Parameter_b_Corrected'] = df.apply(
            lambda row: -row['Parameter_b'] if row.get('Data_Inverted', False) else row['Parameter_b'], axis=1)
    return df

def classify_charge_discharge_pattern(b_values, threshold=0.0005):
    """
    Classify sequence of b_corrected values as charging/discharging based on sign alternation.
    Returns list of True/False for whether each pair alternates with sufficient magnitude.
    """
    pattern_flags = []
    for i in range(1, len(b_values)):
        prev = b_values[i - 1]
        curr = b_values[i]
        if abs(prev) >= threshold and abs(curr) >= threshold and (np.sign(prev) != np.sign(curr)):
            pattern_flags.append(True)
        else:
            pattern_flags.append(False)
    return pattern_flags

def evaluate_experiment_patterns(df, threshold=0.0005, min_consecutive_alternations=3):
    """
    Evaluate each experiment for charging/discharging patterns based on corrected b parameter.
    Returns summary dataframe with per-experiment metrics.
    """
    summary = []
    for experiment_id in df['Experiment_ID'].unique():
        exp_df = df[df['Experiment_ID'] == experiment_id].sort_values(by='Pulse_Number')
        b_values = exp_df['Parameter_b_Corrected'].values
        pattern_flags = classify_charge_discharge_pattern(b_values, threshold)

        alternating_count = sum(pattern_flags)
        total_transitions = len(pattern_flags)
        fraction_alternating = alternating_count / total_transitions if total_transitions > 0 else 0
        qualifies = alternating_count >= min_consecutive_alternations

        summary.append({
            'Experiment_ID': experiment_id,
            'Total_Pulses': len(b_values),
            'Alternating_Transitions': alternating_count,
            'Fraction_Alternating': fraction_alternating,
            'Qualifies_Charging_Discharging': qualifies
        })

    return pd.DataFrame(summary)

def plot_alternation_summary(summary_df, output_dir):
    """
    Creates a bar plot of the number of alternating transitions per experiment.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    summary_df_sorted = summary_df.sort_values(by='Alternating_Transitions', ascending=False)

    ax.bar(summary_df_sorted['Experiment_ID'], summary_df_sorted['Alternating_Transitions'], color='skyblue')
    ax.set_title("Number of Alternating Transitions per Experiment", fontsize=14)
    ax.set_xlabel("Experiment ID")
    ax.set_ylabel("Alternating Transitions")
    ax.set_xticklabels(summary_df_sorted['Experiment_ID'], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "alternating_transitions_summary.png"), dpi=300)
    plt.close()

def plot_sign_sequence_per_experiment(df, output_dir, threshold=0.0005):
    """
    Creates individual plots per experiment showing + / - sequences of b_corrected.
    """
    os.makedirs(output_dir, exist_ok=True)
    for experiment_id in df['Experiment_ID'].unique():
        exp_df = df[df['Experiment_ID'] == experiment_id].sort_values(by='Pulse_Number')
        b_vals = exp_df['Parameter_b_Corrected'].values

        signs = ['+' if b >= threshold else '-' if b <= -threshold else '0' for b in b_vals]
        x_vals = list(range(1, len(signs) + 1))
        colors = ['green' if s == '+' else 'red' if s == '-' else 'gray' for s in signs]

        fig, ax = plt.subplots(figsize=(8, 1.5))
        ax.scatter(x_vals, [1]*len(x_vals), c=colors, s=100, edgecolor='black')

        for i, s in enumerate(signs):
            ax.text(x_vals[i], 1.05, s, ha='center', va='bottom', fontsize=10)

        ax.set_yticks([])
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_vals)
        ax.set_xlim(0.5, len(x_vals) + 0.5)
        ax.set_title(f"Charge/Discharge Sign Sequence - Experiment {experiment_id}")
        ax.grid(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sequence_{experiment_id}.png"), dpi=300)
        plt.close()
        
def plot_sign_sequence_overview(df, output_path, threshold=0.0005):
    """
    Creates a combined sign sequence figure for all experiments.
    """
    experiments = df['Experiment_ID'].unique()
    num_experiments = len(experiments)

    fig, ax = plt.subplots(figsize=(10, 1.5 * num_experiments))

    for idx, experiment_id in enumerate(experiments):
        exp_df = df[df['Experiment_ID'] == experiment_id].sort_values(by='Pulse_Number')
        b_vals = exp_df['Parameter_b_Corrected'].values
        signs = ['+' if b >= threshold else '-' if b <= -threshold else '0' for b in b_vals]
        colors = ['green' if s == '+' else 'red' if s == '-' else 'gray' for s in signs]
        x_vals = list(range(1, len(signs) + 1))
        y_vals = [num_experiments - idx] * len(signs)

        ax.scatter(x_vals, y_vals, c=colors, s=80, edgecolor='black')
        ax.text(0.5, num_experiments - idx + 0.2, f"{experiment_id}", ha='left', va='center', fontsize=9)

    ax.set_yticks([])
    ax.set_xlabel("Pulse Number")
    ax.set_title("Charge/Discharge Sign Sequence Overview (All Experiments)")
    ax.set_xlim(left=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_b_corrected_bars(df, output_dir):
    """
    Create bar plots of b_corrected values per pulse number for each experiment.
    """
    os.makedirs(output_dir, exist_ok=True)
    for experiment_id in df['Experiment_ID'].unique():
        exp_df = df[df['Experiment_ID'] == experiment_id].sort_values(by='Pulse_Number')
        x_vals = exp_df['Pulse_Number'].values
        b_vals = exp_df['Parameter_b_Corrected'].values

        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.bar(x_vals, b_vals, color=['green' if b > 0 else 'red' for b in b_vals])
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f"b_corrected per Pulse - Experiment {experiment_id}")
        ax.set_xlabel("Pulse Number")
        ax.set_ylabel("b_corrected")
        ax.grid(axis='y', linestyle=':', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"b_corrected_{experiment_id}.png"), dpi=300)
        plt.close()
        

# === RUN LOCALLY ===
input_path = "/Users/danielsinausia/Downloads/Cation Sizes/Data Analysis/Exp Fits Charging Discharging/pulse_exponential_fits_results.xlsx"
output_path = "/Users/danielsinausia/Downloads/Cation Sizes/Data Analysis/Exp Fits Charging Discharging/pulse_exponential_fits_results-charging_discharging_summary.csv"

df = load_and_preprocess(input_path, sheet_name="First9_Points_Fits")
summary_df = evaluate_experiment_patterns(df, threshold=0.0005, min_consecutive_alternations=3)

summary_df.to_csv(output_path, index=False)
print(summary_df)

plot_alternation_summary(summary_df, output_dir="plots/alternation_summary")

plot_sign_sequence_per_experiment(df, output_dir="plots/sign_sequences", threshold=0.0005)

plot_sign_sequence_overview(
    df,
    output_path="plots/sign_sequences/overview_all_experiments.png",
    threshold=0.0005
)

plot_b_corrected_bars(
    df,
    output_dir="plots/b_corrected_bars"
)


