
# %%
# 1. Import NumPy and print its version.

import numpy as np
import pandas as pd

df = pd.read_csv("it_services_tickets_raw.csv")

# Clean column names: lowercase, remove extra spaces, replace spaces with underscores
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("NumPy version:", np.__version__)


# %%
# 2. Make an array from the hours logged column.

df["hours_logged"] = pd.to_numeric(df["hours_logged"], errors="coerce")

hours = df["hours_logged"].to_numpy(dtype=float)

print("Hours array created:", hours)


# %%
# 3. Print the shape, size, and data type of the array.

print("Shape:", hours.shape, "Size:", hours.size, "Data type:", hours.dtype)


# %%
# 4. Make an array of numbers 1 to 50 using arange.

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
# 8. Make a 2-D array from hours logged and billing rate columns.

# Remove currency symbols and commas, then convert to numbers.
df["billing_rate_(usd)"] = pd.to_numeric(
    df["billing_rate_(usd)"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False),
    errors="coerce"
)

hours = df["hours_logged"].to_numpy(dtype=float)
billing_rate = df["billing_rate_(usd)"].to_numpy(dtype=float)

two_d_array = np.column_stack((hours, billing_rate))

print("2-D array:\n", two_d_array)


# %%
# 9. Print the number of dimensions of the 2-D array.

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
# 13. Reverse the hours array.

reversed_hours = hours[::-1]

print("Reversed hours array:", reversed_hours)


# %%
# 14. Find the sum of the hours array, ignoring missing values.

print("Total hours:", np.nansum(hours))


# %%
# 15. Find the mean, minimum, and maximum.

print(
    "Mean:", np.nanmean(hours),
    "Minimum:", np.nanmin(hours),
    "Maximum:", np.nanmax(hours)
)


# %%
# 16. Find the standard deviation.

print("Standard deviation:", np.nanstd(hours))


# %%
# 17. Find the median.

print("Median:", np.nanmedian(hours))


# %%
# 18. Count missing values in the array.

print("Missing values:", np.isnan(hours).sum())


# %%
# 19. Find the mean while ignoring missing values.

print("Mean ignoring missing values:", np.nanmean(hours))


# %%
# 20. Replace missing values with 0.

hours[np.isnan(hours)] = 0

print("Missing values replaced with 0.")
print("Updated hours array:", hours)


# %%
# 21. Count values greater than 100.

print("Values greater than 100:", np.sum(hours > 100))


# %%
# 22. Show only values greater than 100.

print("Values greater than 100:", hours[hours > 100])


# %%
# 23. Show values between 10 and 50, including both boundaries.

print("Values between 10 and 50:", hours[(hours >= 10) & (hours <= 50)])


# %%
# 24. Count negative values.

print("Negative values:", np.sum(hours < 0))


# %%
# 25. Replace negative values with 0.

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
# 33. Multiply hours by the billing rate.

revenue = hours * billing_rate

print("Hours multiplied by billing rate:", revenue)


# %%
# 34. Check whether the two arrays have the same length.

print("Same length:", len(hours) == len(billing_rate))


# %%
# 35. Use np.where to classify values above 24 as slow and the rest as fast.

speed = np.where(hours > 24, "slow", "fast")

print("Speed classification:", speed)


# %%
# 36. Make an array of unique priority values.

priority = df["priority"].fillna("Missing").astype(str).to_numpy()

unique_priorities = np.unique(priority)

print("Unique priority values:", unique_priorities)


# %%
# 37. Count how many times each unique priority value appears.

unique_priorities, priority_counts = np.unique(
    priority, return_counts=True
)

print("Priority counts:", dict(zip(unique_priorities, priority_counts)))


# %%
# 38. Take the first 100 values and reshape into a 10 by 10 array.

if hours.size >= 100:
    first_100 = hours[:100].reshape(10, 10)
    print("10 by 10 array:\n", first_100)
else:
    first_100 = None
    print("Not enough values to create a 10 by 10 array.")


# %%
# 39. Transpose the 10 by 10 array.

if first_100 is not None:
    transposed_array = first_100.T
    print("Transposed array:\n", transposed_array)
else:
    transposed_array = None
    print("Cannot transpose: the 10 by 10 array was not created.")


# %%
# 40. Flatten the transposed array into a 1-D array.

if transposed_array is not None:
    flattened_array = transposed_array.flatten()
    print("Flattened array:", flattened_array)
else:
    flattened_array = None
    print("Cannot flatten: the transposed array was not created.")


    
# %%
# 41. Join two arrays using np.concatenate.

array_1 = np.array([1, 2, 3])
array_2 = np.array([4, 5, 6])

joined_array = np.concatenate((array_1, array_2))

print("Joined array:", joined_array)


# %%
# 42. Split the hours array into 5 parts.

split_array = np.array_split(hours, 5)

print("Array split into 5 parts:", split_array)


# %%
# 43. Find the 25th, 50th, and 75th percentiles.

print(
    "25th percentile:", np.nanpercentile(hours, 25),
    "50th percentile:", np.nanpercentile(hours, 50),
    "75th percentile:", np.nanpercentile(hours, 75)
)


# %%
# 44. Find the cumulative sum of the first 20 values.

cumulative_sum = np.cumsum(hours[:20])

print("Cumulative sum of first 20 values:", cumulative_sum)


# %%
# 45. Find the difference between consecutive values using np.diff.

difference = np.diff(hours)

print("Differences:", difference)


# %%
# 46. Keep values within the range 0 to 100 using np.clip.

clipped_hours = np.clip(hours, 0, 100)

print("Values clipped between 0 and 100:", clipped_hours)


# %%
# 47. Convert the array to whole numbers.

whole_hours = hours.astype(int)

print("Whole number array:", whole_hours)


# %%
# 48. Copy the array, change one value in the copy,
# and show that the original did not change.

hours_copy = hours.copy()

if hours_copy.size > 0:
    original_first_value = hours[0]
    hours_copy[0] = 999

    print(
        "Original first value:", hours[0],
        "Copied array first value:", hours_copy[0]
    )
else:
    print("The array is empty.")


# %%
# 49. Save the array to a .npy file and load it back.

np.save("hours_array.npy", hours)

loaded_hours = np.load("hours_array.npy")

print("Loaded array:", loaded_hours)


# %%
# 50. Find total revenue using np.dot().

# Use only rows where both hours and billing rates are available.
valid_revenue = np.isfinite(hours) & np.isfinite(billing_rate)

total_revenue = np.dot(
    hours[valid_revenue],
    billing_rate[valid_revenue]
)

print("Total revenue:", total_revenue)
print("Rows included in revenue:", np.sum(valid_revenue))
print("Rows excluded due to missing or invalid data:", np.sum(~valid_revenue))