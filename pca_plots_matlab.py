'''

I added PCA plots for cyclic voltammetries in the second half, but check that. specifically,
check the intersections and the xlim

'''





# This script scans the subfolders (the experiment names) inside the mother folder and looks for the PCA files, i.e., scores, cve and eigen.
# It then creates a new folder called PCA plots. Inside it, it creates new folders with the same name as the subfolders where the txt files are
# It plots everything as png, eps and csv



import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import itertools
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import seaborn as sns
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'  # Do not convert fonts to paths

folder_path = r"/Users/danielsinausia/Documents/Experiments/DS_00133"
base_plot_directory = folder_path
pca_plots_directory = os.path.join(base_plot_directory, "PCA plots")
if not os.path.exists(pca_plots_directory):
    os.mkdir(pca_plots_directory)


files_in_folder = os.listdir(folder_path)
csv_files = [file for file in files_in_folder if file.startswith("DS_") or file.startswith("CZ_") and file.endswith(".csv")]

if csv_files:
    csv_file = csv_files[0]
    experiment_classification = os.path.splitext(csv_file)[0][-3:]
    print("CSV File:", csv_file)
    print("Experiment Classification:", experiment_classification)
else:
    print("No CSV file starting with 'DS_' found in the specified folder.")




cmap = viridis
num_colors = 15  # Number of colors needed
colors = cmap(np.linspace(0, 1, num_colors))

color_map = {f'PC {i}': colors[i - 1] for i in range(1, num_colors + 1)}
pc_labels = [f'PC{i}' for i in range(1, num_colors + 1)]


def create_variance_plot(df, title, subsubfolder_path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    df.iloc[:16].plot(y="% Variance", legend=False, ax=ax, marker='o', linestyle='-', markersize=4)
    ax.set_xlabel("PCs")
    ax.set_ylabel("% Variance")
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    plt.xlim(0, 15)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    ax.grid(False)
    
    # Add an inset
    inset_ax = inset_axes(ax, width="70%", height="60%", loc='center right', borderpad=0.5)
    inset_ax.plot(df.iloc[1:16].index, df.iloc[1:16]["% Variance"], marker='o', linestyle='-', markersize=4)
    inset_ax.set_xlim(1, 15)
    inset_ax.grid(False)

    save_plots(title, subsubfolder_path)

def create_score_plot(df, title, subsubfolder_path, range_start, range_end, start_index):
    columns_to_plot = df.columns[range_start:range_end]
    legend_labels = [f"PC {i}" for i in range(start_index, start_index + len(columns_to_plot))]

    fig, ax = plt.subplots(figsize=(8.3, 11.7))
    y_axis_separation = 0
    text_labels = []

    if experiment_classification == '_07':
        text_labels = ["-0.05 V", "-0.4 V"]
    elif experiment_classification == '_08':
        text_labels = ["-0.4 V", "-0.8 V"]
    elif experiment_classification == '_09':
        text_labels = ["-0.8 V", "-1.1 V"]

    for i, (column, label) in enumerate(zip(columns_to_plot, legend_labels)):
        plt.plot(df.index * 1.1, df[column] - (i * y_axis_separation), color=color_map[label], label=label)
    #plt.yticks([])
    plt.xlabel("Time (s)")
    plt.ylabel("Score (a.u.)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xlim(0, 1000)
    ax.grid(False)
    
    intersections = [0,100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for intersection in intersections:
        plt.axvline(x=intersection, linestyle='--', color='black', alpha=0.1)
        
    for i in range(len(intersections) - 1):
        x_start = intersections[i]
        x_end = intersections[i + 1]
        text_label = text_labels[i % 2] if text_labels else None  # Use text_labels if it's not empty

        if text_label:  # Only add text if it's not None
            text_x = (x_start + x_end) / 2  # Calculate the x-coordinate for the text label
            if text_label:  # Only add text if it's not None
                text_x = (x_start + x_end) / 2  # Calculate the x-coordinate for the text label
                plt.text(text_x, plt.ylim()[1], text_label, rotation=45, va='bottom', ha='center', fontsize=16, color='black', alpha=0.15)


    #plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # Adjust plot margins
    save_plots(title, subsubfolder_path)
    
def create_stacked_score_plot(df, title, subsubfolder_path, range_start, range_end, start_index):
    columns_to_plot = df.columns[range_start:range_end]
    legend_labels = [f"PC {i}" for i in range(start_index, start_index + len(columns_to_plot))]

    fig, ax = plt.subplots(figsize=(8.3, 11.7))
    y_axis_separation = 10
    text_labels = [] 

    if experiment_classification == '_07':
        text_labels = ["-0.05 V", "-0.4 V"]
    elif experiment_classification == '_08':
        text_labels = ["-0.4 V", "-0.8 V"]
    elif experiment_classification == '_09':
        text_labels = ["-0.8 V", "-1.1 V"]
    for i, (column, label) in enumerate(zip(columns_to_plot, legend_labels)):
        plt.plot(df.index * 1.1, df[column] - (i * y_axis_separation), color=color_map[label], label=label)
    plt.yticks([])
    plt.xlabel("Time (s)")
    plt.ylabel("Score (a.u.)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xlim(0, 1000)
    ax.grid(False)
    intersections = [0,100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for intersection in intersections:
        plt.axvline(x=intersection, linestyle='--', color='black', alpha=0.1)
        
    for i in range(len(intersections) - 1):
        x_start = intersections[i]
        x_end = intersections[i + 1]
        text_label = text_labels[i % 2] if text_labels else None  # Use text_labels if it's not empty

        if text_label:  # Only add text if it's not None
            text_x = (x_start + x_end) / 2  # Calculate the x-coordinate for the text label
            plt.text(text_x, plt.ylim()[1], text_label, rotation=45, va='bottom', ha='center', fontsize=16, color='black', alpha=0.15)

    #plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # Adjust plot margins
    save_plots(title, subsubfolder_path)
    
def create_eigenspectra_plot(df, title, subsubfolder_path, range_start, range_end, start_index):
    columns_to_plot = df.columns[range_start:range_end]
    legend_labels = [f"PC {i}" for i in range(start_index, start_index + len(columns_to_plot))]

    fig, ax = plt.subplots(figsize=(8.3, 11.7))
    for i, (column, label) in enumerate(zip(columns_to_plot, legend_labels)):
        plt.plot(df.iloc[:, 0], df[column] - (i * 0.1), color=color_map[label], label=label)
    plt.yticks([])
    plt.xlabel("Wavenumbers (cm$^{-1}$)")
    plt.ylabel("Loading (a.u.)")
    plt.gca().invert_xaxis()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
    plt.xlim(4000, 1100)
    ax.grid(False)
    save_plots(title, subsubfolder_path)
    

def save_plots(title, subsubfolder_path):
    figure_path = os.path.join(subsubfolder_path, title + ".png")
    #eps_path = os.path.join(subsubfolder_path, title + ".eps")
    svg_path = os.path.join(subsubfolder_path, title + ".svg")

    plt.savefig(figure_path, dpi=300)
    #plt.savefig(eps_path, format='eps')
    plt.savefig(svg_path, format='svg', transparent=True)
    plt.close()

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file == "PCA_CVE.txt":
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path, header=None, names=["% Variance"])
            zero_row = pd.DataFrame([[0]], columns=["% Variance"])
            df = pd.concat([zero_row, df], ignore_index=True)
            title = os.path.splitext(file)[0]
            subsubfolder = os.path.basename(root)
            subsubfolder_path = os.path.join(pca_plots_directory, subsubfolder)
            if not os.path.exists(subsubfolder_path):
                os.mkdir(subsubfolder_path)
            create_variance_plot(df, title + f"1_to_{num_colors}", subsubfolder_path)
    for file in files:
        if file == "PCA_scores.txt":
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path, delimiter="\t")
            title = os.path.splitext(file)[0]
            subsubfolder = os.path.basename(root)
            subsubfolder_path = os.path.join(pca_plots_directory, subsubfolder)
            if not os.path.exists(subsubfolder_path):
                os.mkdir(subsubfolder_path)
            #create_score_plot(df, title + "1_to_3", subsubfolder_path, 1, 4, 1)
            #create_score_plot(df, title + "4_to_20", subsubfolder_path, 5, 22, 4)
            create_score_plot(df, title + f"1_to_{num_colors}", subsubfolder_path, 1, 16, 1)
            
            pc_columns = df.columns[1:16].tolist()  # Convert Index to list
            scores_corr_folder = os.path.join(subsubfolder_path, "Scores correlations")
            if not os.path.exists(scores_corr_folder):
                os.mkdir(scores_corr_folder)

            combinations = list(itertools.product(pc_columns, repeat=2))[:225]  # itertools.product() creates the matrix with (PC1, PC1)(PC1, PC2)(...) and list makes them one single line [(PC1, PC1),(PC1,PC2), (...)]
            fig, axes = plt.subplots(nrows=15, ncols=15, figsize=(100, 100))
            for i, (pc1, pc2) in enumerate(combinations): # Having the list above, enumerate() gives each pair (e.g., (PC1,PC1)) an index,so that then you can pass through them with "for i, ..."
                ax = axes[i // 15, i % 15]  # It assigns the indices of the pairs to the axes. For the rows (i//15), it says "if i = 0, then this index occupies the row 0//15 = 0. If i = 25, it occupies the row 25//15 = 1". For the columns, "if i = 0, it occupies the column 0. if i = 25, it occupies the column 25 % 15 = 1"
                ax.scatter(df[pc1], df[pc2], c='blue', alpha=0.7)
                ax.set_xlabel(pc_labels[pc_columns.index(pc1)])  # Set x-axis label
                ax.set_ylabel(pc_labels[pc_columns.index(pc2)])  # Set y-axis label
                ax.set_title(f"{pc_labels[pc_columns.index(pc2)]} vs {pc_labels[pc_columns.index(pc1)]}")
                corr_coef, p_value = pearsonr(df[pc1], df[pc2])
                r_squared = corr_coef ** 2
                spearman_corr, p_value = spearmanr(df[pc1], df[pc2])
                ax.annotate(fr"$R^2$: {r_squared:.6f}, Spearman's $\rho$: {spearman_corr:.6f}", xy=(0.5, 0.9), xycoords='axes fraction',
                            ha='center', va='center', fontsize=10)
                

                

            plt.tight_layout()
            plt.savefig(os.path.join(scores_corr_folder, "PC_Scatter_Matrix.png"))
            #plt.savefig(os.path.join(scores_corr_folder, "PC_Scatter_Matrix.svg")) # commented out because each figure is like 32 mb
            plt.close()
            
            correlation_matrix = np.zeros((len(pc_columns), len(pc_columns)))
            for i, pc1 in enumerate(pc_columns):
                for j, pc2 in enumerate(pc_columns):
                    spearman_corr, _ = spearmanr(df[pc1], df[pc2])
                    correlation_matrix[i, j] = spearman_corr
    
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='magma', xticklabels=pc_labels, yticklabels=pc_labels, vmin=-0.25, vmax=0.25)
            plt.xlabel("PCs")
            plt.ylabel("PCs")
            plt.title("Spearman Correlation Coefficients Matrix")
            plt.savefig(os.path.join(scores_corr_folder, "Spearman_Correlation_Matrix.png"))
            plt.close()
    
                
    for file in files:
        if file == "PCA_scores.txt":
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path, delimiter="\t")
            title = os.path.splitext(file)[0] + '_stacked'
            subsubfolder = os.path.basename(root)
            subsubfolder_path = os.path.join(pca_plots_directory, subsubfolder)
            if not os.path.exists(subsubfolder_path):
                os.mkdir(subsubfolder_path)
            #create_score_plot(df, title + "1_to_3", subsubfolder_path, 1, 4, 1)
            #create_score_plot(df, title + "4_to_20", subsubfolder_path, 5, 22, 4)
            create_stacked_score_plot(df, title + f"1_to_{num_colors}", subsubfolder_path, 1, 16, 1)
            
    for file in files:
        if file == "PCA_eigenspectra.txt":
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path, delimiter="\t")
            title = os.path.splitext(file)[0]
            subsubfolder = os.path.basename(root)
            subsubfolder_path = os.path.join(pca_plots_directory, subsubfolder)
            if not os.path.exists(subsubfolder_path):
                os.mkdir(subsubfolder_path)
            #create_eigenspectra_plot(df, title + "1_to_3", subsubfolder_path, 1, 4, 1)
            #create_eigenspectra_plot(df, title + "4_to_20", subsubfolder_path, 5, 22, 4)
            create_eigenspectra_plot(df, title + f"1_to_{num_colors}", subsubfolder_path, 1, 16, 1)

