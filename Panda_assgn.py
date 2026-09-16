# %%
import pandas as pd

# %%
#1. Read the file it_services_tickets_raw.csv into a DataFrame called df.
df = pd.read_csv("it_services_tickets_raw.csv")

# %% 
# 2. Show the first 10 rows.
df = pd.read_csv("it_services_tickets_raw.csv")
print("First 10 rows: ", df.head(10));

# %% 
# 3. Show the last 10 rows.
print("Last 10 records: ", df.tail(10))

# %% 
# 4. Print how many rows and how many columns the table has.
print("\n Number of rows: ", df.shape[0],"\n Number of columns: ",df.shape[1])

# %% 
# 5. Print the names of all the columns.
print("\n Column names: ",df.columns)

# %% 
# 6. Print the data type of each column.
print("Data type of each column: ",df.dtypes)

# %%
# 7. Run df.info() and write one line about what it tells you.
print("\n df information: ")
df.info()
# It shows the number of rows, columns, column names, non-null values, data types, and memory usage.

# %%
# 8. Run df.describe() and look at the numeric columns.
print("\nNumeric columns: ",df.describe())

# %%
# 9. Change all column names to lowercase, and replace spaces with underscores.
df.columns = df.columns.str.lower().str.replace(" ", "_")
print("\nUpdated column names: ",df.columns)

# %%
# 10. Print how many missing values each column has.
print("\nMissing values in each column: ",df.isnull().sum())

# %%
# 11. Print the total number of missing values in the whole table.
print("Total missing values: ", df.isnull().sum().sum())

# %%
# 12. Replace the text values NA, N/A, null and - with proper missing values.
df = df.replace(["NA","N/A","null","null","-"], pd.NA)
print("Missing values after replacement: ", df.isnull().sum())

# %%
# 13. Count how many duplicate rows there are.
print("\n Number of duplicate rows: ", df.duplicated().sum())

# %%
# 14. Remove the duplicate rows and print the new number of rows.
df = df.drop_duplicates()
print("New number of rows: ",df.shape[0])

# %%
# 15. Show only three columns: ticket id, client name and status.
print("\n")



# %%
# 16. Show rows 5 to 15 using iloc.

print(df.iloc[5:16])


# %%
# 17. Show the row at index 100 using loc.
print(df.loc[100])

# %%
# 18. Show all the tickets where priority is P1.
print(df[df["priority"] == "P1"])

# %%
# 19. Show all the tickets where status is Open.
print(df[df["status"] == "Open"])

# %%
# 20. Count how many tickets there are for each status.
print("Tickets by status:\n", df["status"].value_counts())

# %%
# 21. Count how many tickets there are for each priority.
print("Tickets by priority:\n", df["priority"].value_counts())

# %%
# 22. Count how many tickets there are for each service line.
print("Tickets by service line:\n", df["serviceline"].value_counts())

# %%
# 23. Find how many different clients there are.
print("Different clients:", df["clientid"].nunique())

# %%
# 24. Print the list of all countries in the data.
print("Countries:", df["client_country"].unique())

# %%
# 25. Convert the hours logged column to a number.
df["hours_logged"] = pd.to_numeric(df["hours_logged"], errors="coerce")
print("Hours logged data type:", df["hours_logged"].dtype)

# %%
# 26. Convert the billing rate column to a number. Remove the $ sign first.
df["billing_rate_(usd)"] = pd.to_numeric(df["billing_rate_(usd)"].astype("string").str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
print("Billing rate data type:", df["billing_rate_(usd)"].dtype)

# %%
# 27. Convert the CSAT score column to a number.
df["csat_score"] = pd.to_numeric(df["csat_score"], errors="coerce")
print("CSAT score data type:", df["csat_score"].dtype)

# %%
# 28. Print the average, the smallest and the largest value of hours logged.
print("Average:", df["hours_logged"].mean(), "Smallest:", df["hours_logged"].min(), "Largest:", df["hours_logged"].max())

# %%
# 29. Sort the table by hours logged from highest to lowest and show the top 10 rows.
print(df.sort_values("hours_logged", ascending=False).head(10))

# %%
# 30. Show the 5 tickets with the highest billable amount.
print(df.sort_values("billable_amount_usd", ascending=False).head(5))

# %%
# 31. Remove the extra spaces from the client name column.
# df["client_name"] = df["client_name"].str.strip()
# print("Extra spaces removed from client names.")

# %%
# 32. Change the client name column to title case.
# df["client_name"] = df["client_name"].str.title()
# print("Client names converted to title case.")

# %%
# 33. Convert the ticket created column to a date.
df["ticket_created_at"] = pd.to_datetime(df["ticket_created_at"], errors="coerce")
print("Ticket created column converted to date.")

# %%
# 34. Add a new column that holds the year of the ticket.
df["year"] = df["ticket_created_at"].dt.year
print("Year column added.")

# %%
# 35. Add a new column that holds the month of the ticket.
df["month"] = df["ticket_created_at"].dt.month
print("Month column added.")

# %%
# 36. Count how many tickets were created in each year.
print("Tickets by year:\n", df["year"].value_counts().sort_index())

# %%
# 37. Add a new column called total equal to billing rate multiplied by hours logged.
df["total"] = df["billing_rate_(usd)"] * df["hours_logged"]
print("Total column added.")

# %%
# 38. Fill the missing values in the CSAT score column with the average CSAT score.
df["csat_score"] = df["csat_score"].fillna(df["csat_score"].mean())
print("Missing CSAT scores filled with the average.")

# %%
# # 39. Drop the rows where the client name is missing.
# df = df.dropna(subset=["client_name"])
# print("Rows with missing client names removed. Remaining rows:", df.shape[0])

# %%
# 40. Group by service line and find the average hours logged.
print("Average hours by service line:\n", df.groupby("serviceline")["hours_logged"].mean())

# %%
# 41. Group by priority and count the tickets.
print("Ticket count by priority:\n", df.groupby("priority")["ticket_id"].count())

# %%
# # 42. Group by client name and find the total billable amount.
# print("Total billable amount by client:\n", df.groupby("client_name")["billable_amount_usd"].sum())

# %%
# 43. Find the top 5 clients by number of tickets.
# print("Top 5 clients by ticket count:\n", df["client_name"].value_counts().head(5))

# %%
# 44. Find the average CSAT score for each priority.
print("Average CSAT by priority:\n", df.groupby("priority")["csat_score"].mean())

# %%
# 45. Draw a bar chart of ticket count by service line.
import matplotlib.pyplot as plt
df["serviceline"].value_counts().plot(kind="bar")
plt.xlabel("Service Line")
plt.ylabel("Number of Tickets")
plt.title("Ticket Count by Service Line")
plt.show()

# %%
# 46. Draw a histogram of hours logged.
df["hours_logged"].plot(kind="hist", bins=10)
plt.xlabel("Hours Logged")
plt.ylabel("Number of Tickets")
plt.title("Distribution of Hours Logged")
plt.show()

# %%
# 47. Show only the tickets where hours logged is more than 50.
print(df[df["hours_logged"] > 50])

# %%
# 48. Show the tickets where status is Closed and priority is P1.
print(df[(df["status"] == "Closed") & (df["priority"] == "P1")])

# %%
# 49. In the SLA breached column, change Yes, Y, TRUE and 1 to True, and everything else to False.
df["sla_breached"] = df["sla_breached"].astype("string").str.strip().str.upper().isin(["YES", "Y", "TRUE", "1"])
print("SLA breached values converted to True/False.")

# %%
# 50. Save your cleaned table to a new CSV file.
df.to_csv("it_services_tickets_cleaned.csv", index=False)
print("Cleaned file saved as it_services_tickets_cleaned.csv")