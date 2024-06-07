#     GraphGazeD tool
#     Copyright (C) 2024 Dimitrios Liaskos (University of West Attica)
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

##----------------------------------------------------------------------------##
## DIFFERENCE CALCULATION FUNCTION
##----------------------------------------------------------------------------##

def dif_calc(dir_path, output_csv_file):
    # Get the list of image files
    files = [file for file in os.listdir(dir_path) if file.endswith('.png')]

    # Open the CSV file in write mode
    with open(output_csv_file, 'w') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the header row to the CSV file
        csv_writer.writerow(["Pair", "Difference", "Threshold"])

        # Dictionary to store pairs and corresponding image files
        files_dict = {}

        # Iterate over image files
        for file in files:
            # Split the file name
            parts = file.split('_')

            # Extract the name and source
            name = parts[0]
            source = parts[-1].split('.')[0]

            # Append the file to the corresponding name-source pair in the dictionary
            pair_key = f"{name}_{source}"
            if pair_key in files_dict:
                files_dict[pair_key].append(file)
            else:
                files_dict[pair_key] = [file]

        # Iterate over unique name-source pairs
        for pair_key, files_source in files_dict.items():
            # Iterate over images in the set
            for i in range(0, min(5, len(files_source))):
                for j in range(i + 1, min(i + 5, len(files_source))):
                    # Read images
                    image1 = cv2.imread(os.path.join(dir_path, files_source[i]), 0).astype("int8")
                    image2 = cv2.imread(os.path.join(dir_path, files_source[j]), 0).astype("int8")

                    # Pair name
                    pair_name = f"{files_source[i].split('.')[0]} - {files_source[j].split('.')[0]}"

                    # Iterate over all threshold values from 0 to 1
                    for threshold in np.linspace(0, 1, num=256):
                        # Calculate difference
                        diff = image1 - image2

                        # Turn difference to table
                        value = np.absolute(diff)

                        # Accumulate counter for all pixels
                        counter = 0
                        for ii in range(value.shape[0]):
                            for jj in range(value.shape[1]):
                                if value[ii,jj] <= threshold * 255:
                                    counter = counter + 1
                                    heat_dif = (counter/((ii+1) * (jj+1)))

                        # Write the data to the CSV file
                        csv_writer.writerow([pair_name, heat_dif, threshold])

    print("Job done!")


# call function
# dif_calc('heatmaps', 'dif_calc.csv')

##----------------------------------------------------------------------------##
## DIFFERENCE PLOT FUNCTION
##----------------------------------------------------------------------------##
    
def dif_plot(file_path, output_folder):
    data = []
    x = []
    titles = []  # Store titles

    # Open the CSV file
    with open(file_path, 'r') as f:
        csvreader = csv.reader(f)
        next(csvreader)  # Skip the header row

        for row in csvreader:
            if len(row) >= 3:
                # Extract the last two columns (assuming they are the last two columns)
                data.append(float(row[-2]))  # Append second-to-last column value as float to 'data' list
                x.append(float(row[-1]))     # Append last column value as float to 'x' list
                titles.append(row[0])        # Append title from the first column
    
    # Split titles into chunks of 256 rows
    titles_chunks = [titles[i:i+256] for i in range(0, len(titles), 256)]

    # Plot the data in chunks of 256 values
    for i in range(len(titles_chunks)):
        start_index = i * 256 # Calculate start index
        end_index = min((i + 1) * 256, len(data))  # Calculate end index

        plt.plot(x[start_index:end_index], data[start_index:end_index], color='blue')  # Plot chunk

        plt.xlabel('Threshold', fontsize=13)
        plt.ylabel('Heatmap Difference', fontsize=13)
        plt.title(titles_chunks[i][0])  # Use the corresponding title for the plot

        yticks = np.linspace(0, 1, 11)
        yticks_rounded = [round(y, 1) for y in yticks]
        plt.yticks(yticks, yticks_rounded)

        # Set axis limits
        plt.xlim(0, max(x))
        plt.ylim(0, max(data))

        output_filename = os.path.join(output_folder, f'{titles_chunks[i][0]}')
        # Save the plot
        plt.savefig(output_filename)
        plt.close()  # Close the plot to release memory
        

    print("Plots generated successfully!")

# call function
# dif_plot('dif_calc.csv', 'plots')

##----------------------------------------------------------------------------##
## CURVE FITTING FUNCTION
##----------------------------------------------------------------------------##

def curve_fitting(file_path, output_folder):
    # Define the function to fit
    
    # 6th degree polynomial function
    def function(x, a, b, c, d, e, f, g):
        return a * x**6 + b * x**5 + c * x**4 + d * x**3 + e * x**2 + f * x + g
    
    # rectangular hyperbola
    #def function(x, a, b, c):
        #return a * x / (b + x) + c

    # logistic function (sigmoid)
    #def function(x, a, b, c):
        #return a / (1 + np.exp(-b * (x - c)))
    
    data = []
    x = []
    titles = []  # Store titles

    # Open the CSV file
    with open(file_path, 'r') as f:
        csvreader = csv.reader(f)
        next(csvreader)  # Skip the header row

        for row in csvreader:
            if len(row) >= 3:
                # Extract the last two columns (assuming they are the last two columns)
                data.append(float(row[-2]))  # Append second-to-last column value as float to 'data' list
                x.append(float(row[-1]))     # Append last column value as float to 'x' list
                titles.append(row[0])        # Append title from the first column

    # Split titles into chunks of 256 rows
    titles_chunks = [titles[i:i+256] for i in range(0, len(titles), 256)]

    # Plot the data in chunks of 256 values
    for i in range(len(titles_chunks)):
        start_index = i * 256 # Calculate start index
        end_index = min((i + 1) * 256, len(data))  # Calculate end index

        plt.plot(x[start_index:end_index], data[start_index:end_index], color='blue')  # Plot chunk

        # Perform the curve fitting
        popt, pcov = curve_fit(function, x[start_index:end_index], data[start_index:end_index])

        # Generate the fitted curve using the optimized parameters
        fitted_curve = function(np.array(x[start_index:end_index]), *popt)

        # Calculate R^2 value
        r_squared = r2_score(data[start_index:end_index], fitted_curve)

        # Plot the fitted curve
        plt.plot(x[start_index:end_index], fitted_curve, 'r-', label=f'Fitted curve (R²={r_squared:.2f})')

        plt.xlabel('Threshold', fontsize=13)
        plt.ylabel('Heatmap Difference', fontsize=13)
        plt.title(titles_chunks[i][0])

        plt.legend()
        plt.grid(True)

        # Set axis limits
        plt.xlim(0, max(x))
        plt.ylim(0, max(data))

        # Increase font size of axis numbers
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        output_filename = os.path.join(output_folder, f'{titles_chunks[i][0]}')
        # Save the plot
        plt.savefig(output_filename)
        plt.close()  # Close the plot to release memory

    print("Plots with fitted curves generated successfully!")

# call function
# curve_fitting('dif_calc.csv', 'curve_fitting_6_degree_polynomial')
