import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_name = "CompB1.csv"
data = pd.read_csv(file_name)

# Extract columns
cpp_distance = data['CPP Distance']
dm = data['dm']
dm_error = data['dm   _error']
cpp_loga = data['CPP Log(Age)']
loga = data['loga']
loga_error = data['loga _error']
cpp_met = data['CPP Metallicity']
met = data['met']
met_error = data['met  _error']
cpp_ebv = data['CPP E(B-V)']
av = data['Av']
av_error = data['Av   _error']

# Calculate the best-fit line for dm data
slope, intercept = np.polyfit(dm, cpp_distance, 1)  # Linear fit (degree 1)
fit_line = slope * np.array(dm) + intercept
# Distance Plot
plt.figure(figsize=(10, 6))
plt.errorbar( dm, cpp_distance, xerr=dm_error, fmt='o', ecolor='red', capsize=3, label='Data with error bars')
plt.ylabel('CPP Data')
plt.xlabel('My data')
plt.title('Distance')
#min_val = min(min(cpp_distance), min(dm))  # Get the minimum value for the line
#max_val = max(max(cpp_distance), max(dm))  # Get the maximum value for the line
min_val = min(dm)
max_val = max(dm)
plt.plot(dm, fit_line, 'g-', label='Best fit line')  # green line
plt.plot([min_val, max_val], [min_val, max_val], 'b--', label='My Data = CPP Data')  # Dashed blue line
plt.legend()
plt.grid()
plt.savefig('distance.png')
plt.show()


# Calculate the best-fit line for loga data
slope, intercept = np.polyfit(loga,cpp_loga, 1)  # Linear fit (degree 1)
fit_line = slope * np.array(loga) + intercept
# Distance Plot
plt.figure(figsize=(10, 6))
plt.errorbar( loga, cpp_loga, xerr=loga_error, fmt='o', ecolor='red', capsize=3, label='Data with error bars')
plt.ylabel('CPP Data')
plt.xlabel('My data')
plt.title('Log(Age)')
#min_val = min(min(cpp_distance), min(dm))  # Get the minimum value for the line
#max_val = max(max(cpp_distance), max(dm))  # Get the maximum value for the line
min_val = min(loga)
max_val = max(loga)
plt.plot(loga, fit_line, 'g-', label='Best fit line')  # green line
plt.plot([min_val, max_val], [min_val, max_val], 'b--', label='My Data = CPP Data')  # Dashed blue line
plt.legend()
plt.grid()
plt.savefig('loga.png')
plt.show()


# Calculate the best-fit line for dm data
slope, intercept = np.polyfit(met, cpp_met, 1)  # Linear fit (degree 1)
fit_line = slope * np.array(met) + intercept
# Distance Plot
plt.figure(figsize=(10, 6))
plt.errorbar( met, cpp_met, xerr=met_error, fmt='o', ecolor='red', capsize=3, label='Data with error bars')
plt.ylabel('CPP Data')
plt.xlabel('My data')
plt.title('Metallicity')
#min_val = min(min(cpp_distance), min(dm))  # Get the minimum value for the line
#max_val = max(max(cpp_distance), max(dm))  # Get the maximum value for the line
min_val = min(met)
max_val = max(met)
plt.plot(met, fit_line, 'g-', label='Best fit line')  # green line
plt.plot([min_val, max_val], [min_val, max_val], 'b--', label='My Data = CPP Data')  # Dashed blue line
plt.legend()
plt.grid()
plt.savefig('metallicity.png')
plt.show()

# Calculate the best-fit line for dm data
slope, intercept = np.polyfit(av, cpp_ebv, 1)  # Linear fit (degree 1)
fit_line = slope * np.array(av) + intercept
# Distance Plot
plt.figure(figsize=(10, 6))
plt.errorbar( av, cpp_ebv, xerr=av_error, fmt='o', ecolor='red', capsize=3, label='Data with error bars')
plt.ylabel('CPP Data')
plt.xlabel('My data')
plt.title('Extinction')
#min_val = min(min(cpp_distance), min(dm))  # Get the minimum value for the line
#max_val = max(max(cpp_distance), max(dm))  # Get the maximum value for the line
min_val = min(av)
max_val = max(av)
plt.plot(av, fit_line, 'g-', label='Best fit line')  # green line
plt.plot([min_val, max_val], [min_val, max_val], 'b--', label='My Data = CPP Data')  # Dashed blue line
plt.legend()
plt.grid()
plt.savefig('extinction.png')
plt.show()