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

        # Iterate over all combinations of pairs of images
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                # Read images
                image1 = cv2.imread(os.path.join(dir_path, files[i]), 0).astype("int8")
                image2 = cv2.imread(os.path.join(dir_path, files[j]), 0).astype("int8")

                # Pair name
                pair_name = f"{files_source[i].split('.')[0]} - {files_source[j].split('.')[0]}"

                # Iterate over all threshold values from 0 to 1
                for threshold in np.linspace(0, 1, num=256):

                    # Calculate difference
                    diff = image1 - image2

                    # Turn difference to table
                    value = np.absolute(diff)

                    # Counter is the number of pixels
                    # It is calculated as i * j
                    # i and j are the image dimensions
                    # Thershold is normalized within the loop
                    counter = 0
                    for ii in range(value.shape[0]):
                        for jj in range(value.shape[1]):
                            if value[ii,jj] <= threshold * 255:
                                counter = counter + 1
                                heat_dif = (counter/((ii+1) * (jj+1)))

                    csv_writer.writerow([pair_name, heat_dif, threshold])

    print("Job done!")

#dif_calc('heatmaps', 'heat_dif_calc.csv')

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

        plt.xlabel('Threshold')
        plt.ylabel('Heatmap Difference')
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

#dif_plot('heat_dif_calc.csv', 'dif_plots')

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

        plt.xlabel('Threshold', fontsize=14)
        plt.ylabel('Heatcomp', fontsize=14)
        plt.title(titles_chunks[i][0])

        plt.legend(fontsize=12)
        plt.grid(True)

        # Set axis limits
        plt.xlim(0, max(x))
        plt.ylim(0, max(data))

        output_filename = os.path.join(output_folder, f'{titles_chunks[i][0]}')
        # Save the plot
        plt.savefig(output_filename)
        plt.close()  # Close the plot to release memory

    print("Plots with fitted curves generated successfully!")

#curve_fitting('heat_dif_calc.csv', 'curve_fitting')