
# %%
# 1. Read the CSV file into a DataFrame called df.

import pandas as pd

df = pd.read_csv("it_services_tickets_raw.csv")


# %%
# 2. Show the first 10 rows.

print("First 10 rows:\n", df.head(10))


# %%
# 3. Show the last 10 rows.

print("Last 10 rows:\n", df.tail(10))


# %%
# 4. Print the number of rows and columns.

print("Number of rows:", df.shape[0], "Number of columns:", df.shape[1])


# %%
# 5. Print all column names.

print("Column names:", df.columns.tolist())


# %%
# 6. Print the data type of each column.

print("Data types:\n", df.dtypes)


# %%
# 7. Run df.info() and explain what it tells you.

df.info()

# It shows the number of rows, column names, non-null counts,
# data types, and memory usage.


# %%
# 8. Display descriptive statistics for numeric columns.

print("Numeric column statistics:\n", df.describe())


# %%
# 9. Convert column names to lowercase and replace spaces with underscores.

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("Updated column names:", df.columns.tolist())


# %%
# 10. Count missing values in each column.

print("Missing values by column:\n", df.isnull().sum())




# %%
# 11. Count the total missing values in the whole DataFrame.

print("Total missing values:", df.isnull().sum().sum())


# %%
# 12. Replace NA, N/A, null, and - with proper missing values.

df = df.replace(
    to_replace=r"(?i)^\s*(NA|N/A|null|-)\s*$",
    value=pd.NA,
    regex=True
)

print("Missing values after replacement:\n", df.isnull().sum())


# %%
# 13. Count duplicate rows.

print("Number of duplicate rows:", df.duplicated().sum())


# %%
# 14. Remove duplicate rows and print the new row count.

df = df.drop_duplicates()

print("Rows after removing duplicates:", df.shape[0])


# %%
# 15. Show only ticket_id, client_name, and status.

print(df[["ticket_id", "client_name", "status"]])


# %%
# 16. Show rows 5 to 15 using iloc (both positions included).

print(df.iloc[5:16])


# %%
# 17. Show the row whose index label is 100 using loc.

if 100 in df.index:
    print(df.loc[100])
else:
    print("Index label 100 does not exist.")


# %%
# 18. Show all tickets with priority P1.

print(df[df["priority"].astype("string").str.strip().str.upper() == "P1"])


# %%
# 19. Show all tickets with status Open.

print(df[df["status"].astype("string").str.strip().str.lower() == "open"])


# %%
# 20. Count tickets for each status.

print("Tickets by status:\n", df["status"].value_counts())


# %%
# 21. Count tickets for each priority.

print("Tickets by priority:\n", df["priority"].value_counts())


# %%
# 22. Count tickets for each service line.

print("Tickets by service line:\n", df["serviceline"].value_counts())


# %%
# 23. Count the number of different clients.

print("Different clients:", df["clientid"].nunique())


# %%
# 24. Print the unique countries.

print("Countries:", df["client_country"].dropna().unique())


# %%
# 25. Convert hours_logged to numeric.

df["hours_logged"] = pd.to_numeric(df["hours_logged"], errors="coerce")

print("Hours logged data type:", df["hours_logged"].dtype)


# %%
# 26. Convert billing_rate_(usd) to numeric after removing $ and commas.

df["billing_rate_(usd)"] = pd.to_numeric(
    df["billing_rate_(usd)"].astype("string")
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False),
    errors="coerce"
)

print("Billing rate data type:", df["billing_rate_(usd)"].dtype)


# %%
# 27. Convert csat_score to numeric.

df["csat_score"] = pd.to_numeric(df["csat_score"], errors="coerce")

print("CSAT score data type:", df["csat_score"].dtype)


# %%
# 28. Find the average, minimum, and maximum hours logged.

print(
    "Average:", df["hours_logged"].mean(),
    "Smallest:", df["hours_logged"].min(),
    "Largest:", df["hours_logged"].max()
)


# %%
# 29. Sort by hours logged from highest to lowest and show top 10.

print(df.sort_values("hours_logged", ascending=False).head(10))


# %%
# 30. Show the 5 tickets with the highest billable amount.

df["billable_amount_usd"] = pd.to_numeric(
    df["billable_amount_usd"].astype("string")
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False),
    errors="coerce"
)

print(df.sort_values("billable_amount_usd", ascending=False).head(5))


# %%
# 31. Remove extra spaces from client names.

df["client_name"] = df["client_name"].str.strip()

print("Extra spaces removed from client names.")


# %%
# 32. Convert client names to title case.

df["client_name"] = df["client_name"].str.title()

print("Client names converted to title case.")


# %%
# 33. Convert ticket_created_at to datetime.

df["ticket_created_at"] = pd.to_datetime(
    df["ticket_created_at"], errors="coerce"
)

print("Ticket created column converted to datetime.")


# %%
# 34. Add a year column.

df["year"] = df["ticket_created_at"].dt.year

print("Year column added.")


# %%
# 35. Add a month column.

df["month"] = df["ticket_created_at"].dt.month

print("Month column added.")


# %%
# 36. Count tickets created in each year.

print("Tickets by year:\n", df["year"].value_counts().sort_index())


# %%
# 37. Add a total column: billing rate multiplied by hours logged.

df["total"] = df["billing_rate_(usd)"] * df["hours_logged"]

print("Total column added.")


# %%
# 38. Fill missing CSAT scores with the average CSAT score.

average_csat = df["csat_score"].mean()

if pd.notna(average_csat):
    df["csat_score"] = df["csat_score"].fillna(average_csat)
    print("Missing CSAT scores filled with the average.")
else:
    print("CSAT scores are all missing; no average is available.")


# %%
# 39. Remove rows where client_name is missing.

df = df.dropna(subset=["client_name"])

print("Rows with missing client names removed. Remaining rows:", df.shape[0])


# %%
# 40. Group by service line and find average hours logged.

print(
    "Average hours by service line:\n",
    df.groupby("serviceline")["hours_logged"].mean()
)



# %%
# 41. Group by priority and count tickets.

print(
    "Ticket count by priority:\n",
    df.groupby("priority")["ticket_id"].count()
)


# %%
# 42. Group by client name and find total billable amount.

print(
    "Total billable amount by client:\n",
    df.groupby("client_name")["billable_amount_usd"].sum()
)


# %%
# 43. Find the top 5 clients by number of tickets.

print(
    "Top 5 clients by ticket count:\n",
    df["client_name"].value_counts().head(5)
)


# %%
# 44. Find average CSAT score for each priority.

print(
    "Average CSAT by priority:\n",
    df.groupby("priority")["csat_score"].mean()
)


# %%
# 45. Draw a bar chart of ticket count by service line.

import matplotlib.pyplot as plt

df["serviceline"].value_counts().plot(kind="bar")

plt.xlabel("Service Line")
plt.ylabel("Number of Tickets")
plt.title("Ticket Count by Service Line")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
# 46. Draw a histogram of hours logged.

df["hours_logged"].dropna().plot(kind="hist", bins=10)

plt.xlabel("Hours Logged")
plt.ylabel("Number of Tickets")
plt.title("Distribution of Hours Logged")
plt.tight_layout()
plt.show()


# %%
# 47. Show tickets where hours logged is greater than 50.

print(df[df["hours_logged"] > 50])


# %%
# 48. Show tickets where status is Closed and priority is P1.

print(
    df[
        (df["status"].astype("string").str.strip().str.lower() == "closed")
        & (df["priority"].astype("string").str.strip().str.upper() == "P1")
    ]
)


# %%
# 49. Convert SLA breached values to True or False.
# Yes, Y, TRUE, and 1 become True; everything else becomes False.

df["sla_breached"] = (
    df["sla_breached"]
    .astype("string")
    .str.strip()
    .str.upper()
    .isin(["YES", "Y", "TRUE", "1"])
)

print("SLA breached values converted to True/False.")
print(df["sla_breached"].value_counts(dropna=False))


# %%
# 50. Save the cleaned DataFrame to a new CSV file.

df.to_csv("it_services_tickets_cleaned.csv", index=False)

print("Cleaned file saved as it_services_tickets_cleaned.csv")