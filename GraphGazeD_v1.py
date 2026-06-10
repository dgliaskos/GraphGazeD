#     GraphGazeD
#     Copyright (C) 2026 Dimitrios Liaskos (University of West Attica)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see .
#
#     For further information, please email me: dliaskos[at]uniwa[dot]gr or dgliaskos[at]gmail[dot]com

import os
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

##============================================================================##
## MATHEMATICAL MODEL DEFINITIONS
##============================================================================##

def poly_6(x, a, b, c, d, e, f, g):
    return a * x**6 + b * x**5 + c * x**4 + d * x**3 + e * x**2 + f * x + g


def logistic(x, a, b, c):
    x_safe = np.clip(b * (x - c), -500, 500)
    return a / (1 + np.exp(-x_safe))

##============================================================================##
## STEP 1: DIFFERENCE CALCULATION
##============================================================================##

def dif_calc(dir_path, output_csv_file):
    
    print("Starting difference calculation...")
    
    files = [file for file in os.listdir(dir_path) if file.endswith('.png')]
    
    if not files:
        print(f"ERROR: No PNG files found in '{dir_path}'")
        return
    
    print(f"Found {len(files)} image files")

    with open(output_csv_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Pair", "Difference", "Threshold"])

        files_dict = {}
        
        for file in files:
            parts = file.split('_')
            name = parts[0]
            source = parts[-1].split('.')[0]
            pair_key = f"{name}_{source}"
            
            if pair_key in files_dict:
                files_dict[pair_key].append(file)
            else:
                files_dict[pair_key] = [file]

        pair_count = 0
        
        for pair_key, files_source in files_dict.items():
            for i in range(0, min(5, len(files_source))):
                for j in range(i + 1, min(i + 5, len(files_source))):
                    
                    img_path_1 = os.path.join(dir_path, files_source[i])
                    img_path_2 = os.path.join(dir_path, files_source[j])
                    
                    image1 = cv2.imread(img_path_1, 0).astype("float64")
                    image2 = cv2.imread(img_path_2, 0).astype("float64")

                    if image1 is None or image2 is None:
                        print(f"  WARNING: Failed to load {files_source[i]} or {files_source[j]}")
                        continue

                    pair_name = f"{files_source[i].split('.')[0]} - {files_source[j].split('.')[0]}"

                    diff = image1 - image2
                    value = np.absolute(diff)
                    total_pixels = value.shape[0] * value.shape[1]

                    for threshold in np.linspace(0, 1, num=256):
                        threshold_value = threshold * 255
                        counter = np.sum(value <= threshold_value)
                        heat_dif = counter / total_pixels

                        csv_writer.writerow([pair_name, heat_dif, threshold])
                    
                    pair_count += 1
                    print(f"  Processed pair {pair_count}: {pair_name}")

    print(f"Difference calculation complete!")
    print(f"Results saved to: {output_csv_file}\n")

##============================================================================##
## STEP 2: DIFFERENCE PLOTTING
##============================================================================##

def dif_plot(file_path, output_folder):
    
    print("Generating difference plots...")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    data = []
    x = []
    titles = []

    with open(file_path, 'r') as f:
        csvreader = csv.reader(f)
        next(csvreader)

        for row in csvreader:
            if len(row) >= 3:
                data.append(float(row[-2]))
                x.append(float(row[-1]))
                titles.append(row[0])
    
    print(f"Read {len(data)} data points")

    titles_chunks = [titles[i:i+256] for i in range(0, len(titles), 256)]
    plot_count = 0

    for i in range(len(titles_chunks)):
        start_index = i * 256
        end_index = min((i + 1) * 256, len(data))

        plt.figure(figsize=(11, 7))
        plt.plot(x[start_index:end_index], data[start_index:end_index], 
                color='blue', linewidth=2.5, label='GraphGazeD')

        plt.xlabel('Threshold', fontsize=13)
        plt.ylabel('Heatmap Difference', fontsize=13)
        plt.title(titles_chunks[i][0], fontsize=14, fontweight='bold')

        yticks = np.linspace(0, 100, 11)
        plt.yticks(yticks, fontsize=11)
        plt.xticks(fontsize=11)

        plt.xlim(0, 1)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.4, linestyle='--')
        plt.legend(fontsize=11, loc='best')
        plt.tight_layout()

        output_filename = os.path.join(output_folder, f'{titles_chunks[i][0]}.png')
        plt.savefig(output_filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        plot_count += 1

    print(f"Generated {plot_count} plots")
    print(f"Plots saved to: {output_folder}\n")

##============================================================================##
## STEP 3: CURVE FITTING (POLYNOMIAL OR LOGISTIC)
##============================================================================##

def curve_fitting(file_path, output_folder, model_type='poly'):
    
    if model_type not in ['poly', 'logistic']:
        raise ValueError(f"model_type must be 'poly' or 'logistic', got '{model_type}'")
    
    print(f"Starting curve fitting with {model_type} model...")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    data = []
    x = []
    titles = []

    with open(file_path, 'r') as f:
        csvreader = csv.reader(f)
        next(csvreader)

        for row in csvreader:
            if len(row) >= 3:
                data.append(float(row[-2]))
                x.append(float(row[-1]))
                titles.append(row[0])

    x_array = np.array(x)
    data_array = np.array(data)

    titles_chunks = [titles[i:i+256] for i in range(0, len(titles), 256)]
    plot_count = 0

    for i in range(len(titles_chunks)):
        start_index = i * 256
        end_index = min((i + 1) * 256, len(data))

        x_chunk = x_array[start_index:end_index]
        y_chunk = data_array[start_index:end_index]
        pair_name = titles_chunks[i][0]

        if model_type == 'poly':
            _fit_polynomial(x_chunk, y_chunk, pair_name, output_folder)
        else:  # logistic
            _fit_logistic(x_chunk, y_chunk, pair_name, output_folder)

        plot_count += 1

    print(f"Generated {plot_count} plots")
    print(f"Plots saved to: {output_folder}\n")


def _fit_polynomial(x, y, title, output_folder):
    """Fit 6th degree polynomial and generate plot"""
    
    model_label = 'Polynomial (6th degree)'
    
    plt.figure(figsize=(11, 7))
    plt.plot(x, y, 'b-', linewidth=2.5, label='Actual Data', marker='o', 
            markersize=3, markerfacecolor='lightblue', markeredgecolor='blue')

    try:
        popt, _ = curve_fit(poly_6, x, y, maxfev=5000)
        fitted_curve = poly_6(x, *popt)
        r_squared = r2_score(y, fitted_curve)

        plt.plot(x, fitted_curve, 'r-', linewidth=3, 
                label=f'{model_label} (R²={r_squared:.4f})')

    except Exception as e:
        print(f"  WARNING: Polynomial fitting failed for '{title}': {e}")
        plt.plot(x, y, 'r-', linewidth=2.5, label='Fit failed')

    plt.xlabel('Threshold', fontsize=13)
    plt.ylabel('Heatmap Difference (%)', fontsize=13)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()

    output_filename = os.path.join(output_folder, f'{title}.png')
    plt.savefig(output_filename, dpi=100, bbox_inches='tight')
    plt.close()


def _fit_logistic(x, y, title, output_folder):
    """Fit logistic/sigmoid function and generate plot"""
    
    model_label = 'Logistic (Sigmoid)'
    
    plt.figure(figsize=(11, 7))
    plt.plot(x, y, 'b-', linewidth=2.5, label='Actual Data', marker='o',
            markersize=3, markerfacecolor='lightblue', markeredgecolor='blue')

    try:
        popt, _ = curve_fit(logistic, x, y, p0=[100, 5, 0.5], maxfev=5000)
        fitted_curve = logistic(x, *popt)
        r_squared = r2_score(y, fitted_curve)

        plt.plot(x, fitted_curve, 'r-', linewidth=3,
                label=f'{model_label} (R²={r_squared:.4f})')

    except Exception as e:
        print(f"  WARNING: Logistic fitting failed for '{title}': {e}")
        plt.plot(x, y, 'r-', linewidth=2.5, label='Fit failed')

    plt.xlabel('Threshold', fontsize=13)
    plt.ylabel('Heatmap Difference (%)', fontsize=13)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()

    output_filename = os.path.join(output_folder, f'{title}.png')
    plt.savefig(output_filename, dpi=100, bbox_inches='tight')
    plt.close()


##============================================================================##
## MAIN EXECUTION
##============================================================================##

if __name__ == "__main__":
    
    print("="*80)
    print("GraphGazeD")
    print("="*80 + "\n")
    
    heatmap_dir = 'heatmaps'
    csv_output = 'dif_calc.csv'
    plots_dir = 'plots'
    fitting_dir = 'curve_fitting_results'
    
    print("Configuration:")
    print(f"  Input directory: {heatmap_dir}")
    print(f"  CSV output: {csv_output}")
    print(f"  Plots directory: {plots_dir}")
    print(f"  Fitting directory: {fitting_dir}\n")
    
    print("Step 1: Calculate differences")
    print("-" * 80)
    dif_calc(heatmap_dir, csv_output)
    
    print("Step 2: Generate difference plots")
    print("-" * 80)
    dif_plot(csv_output, plots_dir)
    
    print("Step 3: Fit curves to data")
    print("-" * 80)
    print("Choose model type:")
    print("  'poly'     - 6th degree polynomial (7 parameters)")
    print("  'logistic' - Logistic/sigmoid function (3 parameters)")
    print()
    
    model_choice = input("Enter model type (poly or logistic): ").strip().lower()
    
    if model_choice not in ['poly', 'logistic']:
        print(f"Error: Invalid choice '{model_choice}'. Must be 'poly' or 'logistic'")
    else:
        curve_fitting(csv_output, fitting_dir, model_type=model_choice)
    
    print("="*80)
    print("Pipeline complete!")
    print("="*80)
