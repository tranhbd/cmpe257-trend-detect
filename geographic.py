"""
Geographic trend forecasting portion of the project.

Current pipeline:
1. Load geographic interest data.
2. Load emerging trend data.
3. Validate the datasets.
4. Merge the datasets together using topic_id.

Next steps:
- Prepare data for future-interest prediction.
- Split data into training/testing sets.
- Train a regression model to predict future interest.

Need to coordinate with partner:
- Confirm how her temporal/emerging trend output will connect to this file.
- Confirm the final column names/schema that her code will produce.
- Confirm which prediction horizons we are using (ex: 1, 2, and 4 weeks).
"""


import pandas as pd


# Load geographic trend data from a CSV file.
# Each row represents the interest level of a topic in a specific state on a specific date.
def load_geographic_data(file_path):
    geographic_data = pd.read_csv(file_path)

    # Convert the date column from text into a datetime value
    # This will allow us to work with the data chronologically later
    geographic_data["date"] = pd.to_datetime(geographic_data["date"])
    return geographic_data

# Load emerging trend results from a CSV file
# This dataset tells us whether each topic emerged and if it did, the date when it emerged.
def load_emerging_data(file_path):
    emerging_data = pd.read_csv(file_path)

    # Convert emergence_date into a datetime value
    # A topic with emerged = 0 may have NAT here because it does not have an emergence date.
    emerging_data["emergence_date"] = pd.to_datetime(emerging_data["emergence_date"])
    return emerging_data

# Preform basic checks on any DataFrame before using it
# This helps us identify missing values or duplicate rows
def validate_data(data):
    print("Missing values:")
    print(data.isnull().sum())

    print("\nDuplicate rows:")
    print(data.duplicated().sum())   


# Combine geographic intrest data with emerging trend information
# topic_id is the common identifier connecting the two datasets.
def merge_data(geographic_data, emerging_data):
    
    # Only take the emerging-data columns that we need
    # We do not include topic_name because geographic_data already contains it
    combined_data = pd.merge(
        geographic_data,
        emerging_data[["topic_id", "emerged", "emergence_date"]],
        on="topic_id",
        how="left"
    )

    # "left" keeps every geographic observation even if there is no matchign emerging-trend record for a topic.
    return combined_data