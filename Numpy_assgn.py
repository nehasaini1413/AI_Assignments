# %%
# # 1. Import NumPy and print its version.
import numpy as np
import pandas as pd
df = pd.read_csv("it_services_tickets_raw.csv")
print("NumPy version:", np.__version__)

# %%

# 2. Make an array from the hours logged column.
hours = np.array(df["hours_logged"])
print("Hours array created.")

# %%

# 3. Print the shape, the size and the data type of that array.
print("Shape:", hours.shape, "Size:", hours.size, "Data type:", hours.dtype)

# %%

# 4. Make an array of the numbers 1 to 50 using arange.
arr_1_to_50 = np.arange(1, 51)
print("Array from 1 to 50:", arr_1_to_50)

# %%

# 5. Make 10 evenly spaced numbers between 0 and 100 using linspace.
arr_linspace = np.linspace(0, 100, 10)
print("10 evenly spaced numbers:", arr_linspace)

# %%

# 6. Make an array of 10 zeros and an array of 10 ones.
zeros = np.zeros(10)
ones = np.ones(10)
print("Zeros:", zeros, "Ones:", ones)

# %%

# 7. Make a 3 by 3 array of random numbers using seed 42.
np.random.seed(42)
random_array = np.random.rand(3, 3)
print("3 by 3 random array:\n", random_array)

# %%

# 8. Make a 2-D array from the hours logged and billing rate columns.
df.columns = df.columns.str.lower().str.replace(" ", "_")
print(df.columns.tolist())
df["billing_rate_(usd)"] = pd.to_numeric(
    df["billing_rate_(usd)"].astype(str).str.replace("$", "").str.replace(",", ""),
    errors="coerce"
)

print(df["billing_rate_(usd)"].head())

hours = np.array(df["hours_logged"])
billing_rate = np.array(df["billing_rate_(usd)"])
two_d_array = np.column_stack((hours, billing_rate))
print("2-D array:", two_d_array)
# %%


# 9. Print how many dimensions that 2-D array has.
print("Number of dimensions:", two_d_array.ndim)

# %%

# 10. Print the first 5 values of the hours array.
print("First 5 hours values:", hours[:5])

# %%

# 11. Print the last 5 values of the hours array.
print("Last 5 hours values:", hours[-5:])

# %%

# 12. Print every second value from the first 20 values.
print("Every second value from first 20:", hours[:20:2])

# %%

# 13. Reverse the array.
reversed_hours = hours[::-1]
print("Reversed hours array:", reversed_hours)

# %%

# 14. Find the sum of the hours array.
# Convert hours_logged to numbers and handle invalid/missing values.
df["hours_logged"] = pd.to_numeric(df["hours_logged"], errors="coerce")
hours = np.array(df["hours_logged"], dtype=float)
print("Hours data type:", hours.dtype)


# %%

# 15. Find the mean, the minimum and the maximum of the hours array.
print("Mean:", np.nanmean(hours), "Minimum:", np.nanmin(hours), "Maximum:", np.nanmax(hours))

# %%

# 16. Find the standard deviation.
print("Standard deviation:", np.nanstd(hours))

# %%

# 17. Find the median.
print("Median:", np.nanmedian(hours))

# %%

# 18. Count how many values are missing in the array.
print("Missing values:", np.isnan(hours).sum())

# %%

# 19. Find the mean while ignoring the missing values.
print("Mean ignoring missing values:", np.nanmean(hours))

# %%

# 20. Replace all the missing values with 0.
hours[np.isnan(hours)] = 0
print("Missing values replaced with 0.")

# %%

# 21. Count how many values are greater than 100.
print("Values greater than 100:", np.sum(hours > 100))

# %%

# 22. Show only the values that are greater than 100.
print("Values greater than 100:", hours[hours > 100])

# %%

# 23. Show the values that are between 10 and 50.
print("Values between 10 and 50:", hours[(hours >= 10) & (hours <= 50)])

# %%

# 24. Count how many values are negative.
print("Negative values:", np.sum(hours < 0))

# %%

# 25. Replace all the negative values with 0.
hours[hours < 0] = 0
print("Negative values replaced with 0.")

# %%

# 26. Sort the array from smallest to largest.
sorted_hours = np.sort(hours)
print("Sorted hours:", sorted_hours)

# %%

# 27. Find the position of the largest value.
print("Position of largest value:", np.argmax(hours))

# %%

# 28. Find the position of the smallest value.
print("Position of smallest value:", np.argmin(hours))

# %%

# 29. Find the 5 largest values.
five_largest = np.sort(hours)[-5:][::-1]
print("5 largest values:", five_largest)

# %%

# 30. Round every value to 1 decimal place.
rounded_hours = np.round(hours, 1)
print("Rounded hours:", rounded_hours)

# %%

# 31. Multiply every value by 2.
hours_times_2 = hours * 2
print("Hours multiplied by 2:", hours_times_2)

# %%

# 32. Add 10 to every value.
hours_plus_10 = hours + 10
print("10 added to every value:", hours_plus_10)

# %%

# 33. Multiply the hours array by the billing rate array.
revenue = hours * billing_rate
print("Hours multiplied by billing rate:", revenue)

# %%

# 34. Check whether the two arrays have the same length.
print("Same length:", len(hours) == len(billing_rate))

# %%

# 35. Use np.where to mark values above 24 as slow and the rest as fast.
speed = np.where(hours > 24, "slow", "fast")
print("Speed classification:", speed)

# %%

# 36. Make an array of the unique priority values.
priority = df["priority"].fillna("Missing").astype(str).to_numpy(dtype=str)
unique_priorities = np.unique(priority)
print("Unique priority values:", unique_priorities)
# %%

# 37. Count how many times each unique priority value appears.
unique_priorities, priority_counts = np.unique(priority, return_counts=True)
print("Priority counts:", dict(zip(unique_priorities, priority_counts)))

# %%

# 38. Take the first 100 values and reshape them into a 10 by 10 array.
first_100 = hours[:100].reshape(10, 10)
print("10 by 10 array:\n", first_100)

# %%

# 39. Transpose that 10 by 10 array.
transposed_array = first_100.T
print("Transposed array:\n", transposed_array)

# %%

# 40. Flatten it back into a 1-D array.
flattened_array = transposed_array.flatten()
print("Flattened array:", flattened_array)

# %%

# 41. Join two arrays together using np.concatenate.
array_1 = np.array([1, 2, 3])
array_2 = np.array([4, 5, 6])
joined_array = np.concatenate((array_1, array_2))
print("Joined array:", joined_array)

# %%

# 42. Split an array into 5 equal parts.
split_array = np.array_split(hours[:100], 5)
print("Array split into 5 parts:", split_array)

# %%

# 43. Find the 25th, 50th and 75th percentile.
print("25th percentile:", np.percentile(hours, 25), "50th percentile:", np.percentile(hours, 50), "75th percentile:", np.percentile(hours, 75))

# %%

# 44. Find the cumulative sum of the first 20 values.
cumulative_sum = np.cumsum(hours[:20])
print("Cumulative sum of first 20 values:", cumulative_sum)

# %%

# 45. Find the difference between each value and the one before it using np.diff.
difference = np.diff(hours)
print("Differences:", difference)

# %%

# 46. Keep every value inside the range 0 to 100 using np.clip.
clipped_hours = np.clip(hours, 0, 100)
print("Values clipped between 0 and 100:", clipped_hours)

# %%

# 47. Convert the array to whole numbers.
whole_hours = hours.astype(int)
print("Whole number array:", whole_hours)

# %%

# 48. Copy the array, change one value in the copy, and show that the original did not change.
hours_copy = hours.copy()
hours_copy[0] = 999
print("Original first value:", hours[0], "Copied array first value:", hours_copy[0])

# %%

# 49. Save the array to a .npy file and load it back.
np.save("hours_array.npy", hours)
loaded_hours = np.load("hours_array.npy")
print("Loaded array:", loaded_hours)

# %%

# 50. Find the total revenue using np.dot on the hours array and the rate array.
total_revenue = np.dot(hours, billing_rate)
print("Total revenue:", total_revenue)

# %%