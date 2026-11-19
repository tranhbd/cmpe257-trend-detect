import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
Geographic trend forecasting portion of the project.

Current pipeline:
1. Load geographic interest data.
2. Load emerging trend data.
3. Validate the datasets.
4. Merge the datasets together using topic_id.

Confirmed project requirements:
- Emerging data has one row per topic.
- Geographic data has multiple rows per topic across states and weeks.
- Emerging output uses the same schema as the dummy emerging dataset.
- Predict geographic interest 1, 2, and 4 weeks ahead.
- Use emergence date to help determine the appropriate prediction cutoff.
- Use Gradient Boosting as the primary regression model.
- Random Forest can be used for comparison if time permits.
"""



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
    # A topic with emerged = 0 may have NaT here because it does not have an emergence date.
    emerging_data["emergence_date"] = pd.to_datetime(emerging_data["emergence_date"])
    return emerging_data

# Perform basic checks on any DataFrame before using it
# This helps us identify missing values or duplicate rows
def validate_data(data):
    print("Missing values:")
    print(data.isnull().sum())

    print("\nDuplicate rows:")
    print(data.duplicated().sum())   


# Combine geographic interest data with emerging trend information
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

    # "left" keeps every geographic observation even if there is no matching emerging-trend record for a topic.
    return combined_data


# Create future interest values for the regression models.
# Each target represents the interest score for the same topic and states 1, 2, or 4 weeks into the future
def create_prediction_targets(combined_data):

    # Sort each topic/state chronologically before looking ahead
    prediction_data = combined_data.sort_values(
        by=["topic_id", "state", "date"]
    ).copy()

    # Group rows so that the future values are taken only from the same topic and the same state
    grouped_data = prediction_data.groupby(
        ["topic_id", "state"]
    )["interest"]

    # Create the future interest values that the models will eventually learn to predict.
    prediction_data["interest_1w"] = grouped_data.shift(-1)
    prediction_data["interest_2w"] = grouped_data.shift(-2)
    prediction_data["interest_4w"] = grouped_data.shift(-4)

    return prediction_data

# Create historical interest features for the regression model. These features give the model the recent interest history for the same topic and state.
def create_lag_features(prediction_data):

    # Make a copy so the original prediction data is not modified
    model_data = prediction_data.copy()

    # Group by topic and state so previous interest values always come from the same topic and the same state.
    grouped_data = model_data.groupby(
        ["topic_id", "state"]
    )["interest"]

    # Interest from 1 week ago
    model_data["interest_lag_1"] = grouped_data.shift(1)

    # Interest from 2 weeks ago
    model_data["interest_lag_2"] = grouped_data.shift(2)

    return model_data

# Prepare complete examples that can be used to train the regression models.
def prepare_model_data(model_data):

    # These columns are required for a complete training example
    required_columns = [
        "interest_lag_2",
        "interest_lag_1",
        "interest",
        "interest_1w",
        "interest_2w",
        "interest_4w"
    ]

    # Remove rows that are missing any required historical features or future prediction targets
    prepared_data = model_data.dropna(
        subset=required_columns
    ).copy()

    return prepared_data

# Split the prepared data into training, validation, and testing sets. The data is split chronologically so the model trains on earlier observations and is evaluated on later observations.
def split_model_data(prepared_data):

    # Sort all unique dates chronologically
    unique_dates = prepared_data["date"].sort_values().unique() 

    # Find the date positions for the 70% and 85% cutoffs
    train_end = int(len(unique_dates) * 0.70)
    validation_end = int(len(unique_dates) * 0.85)

    # Divide the unique dates into three groups
    train_dates = unique_dates[:train_end]
    validation_dates = unique_dates[train_end:validation_end]
    test_dates = unique_dates[validation_end:]

    # Select rows based on which date group they belong to
    train_data = prepared_data[
        prepared_data["date"].isin(train_dates)
    ].copy()

    validation_data = prepared_data[
        prepared_data["date"].isin(validation_dates)
    ].copy()

    test_data = prepared_data[
        prepared_data["date"].isin(test_dates)
    ].copy()

    return train_data, validation_data, test_data

# Train Gradient Boosting models for the 1-week, 2-week, and 4-week prediction horizons.
def train_gradient_boosting(train_data):

    # Historical interest values used as model inputs
    feature_columns = [
        "interest_lag_2",
        "interest_lag_1",
        "interest"
    ]

    # Future interest values that the models will learn to predict
    target_columns = [
        "interest_1w",
        "interest_2w",
        "interest_4w"
    ]

    # Input features used by all three models
    X_train = train_data[feature_columns]

    # Store each trained model in this dictionary
    models = {}

    # Train one Gradient Boosting model for each prediction horizon
    for target in target_columns:

        y_train = train_data[target]

        model = GradientBoostingRegressor(
            random_state=42
        )

        model.fit(X_train, y_train)             # Train the model using the training data

        models[target] = model

    return models

# Evaluate the trained Gradient Boosting models using validation data.
def validate_models(models, validation_data):

    # Same input features that were used during training
    feature_columns = [
        "interest_lag_2",
        "interest_lag_1",
        "interest"
    ]

    X_validation = validation_data[feature_columns]

    # Store the evaluation results for each prediction horizon
    results = {}

    for target, model in models.items():

        # Actual future interest values
        y_actual = validation_data[target]

        # Use the trained model to predict future interests
        y_predicted = model.predict(X_validation)

        # Calculate prediction errors
        mae = mean_absolute_error(y_actual, y_predicted)
        rmse = mean_squared_error(
            y_actual,
            y_predicted
        ) ** 0.5

        results[target] = {
            "MAE": mae,
            "RMSE": rmse
        }

    return results

# Generate 1-week, 2-week, and 4-week forecast using each topic's emergence date as the forecasting cutoff.
def forecast_from_emergence(models, model_data):

    # Only use topics that actually emerged and have an emergence date
    emerged_data = model_data[
        (model_data["emerged"] == 1) &
        (model_data["emergence_date"].notna())
    ].copy()

    # Keep only observations that occur exactly at the emergence date
    cutoff_data = emerged_data[
        emerged_data["date"] == emerged_data["emergence_date"]
    ].copy()

    # Historical features used by the trained models
    feature_columns = [
        "interest_lag_2",
        "interest_lag_1",
        "interest"
    ]

    # Remove cutoff rows that do not have enough historical data
    cutoff_data = cutoff_data.dropna(
        subset=feature_columns
    ).copy()

    X_cutoff = cutoff_data[feature_columns]

    # Generate predictions for each forecasting horizon
    cutoff_data["predicted_interest_1w"] = models[
        "interest_1w"
    ].predict(X_cutoff)

    cutoff_data["predicted_interest_2w"] = models[
        "interest_2w"
    ].predict(X_cutoff)

    cutoff_data["predicted_interest_4w"] = models[
        "interest_4w"
    ].predict(X_cutoff)

    return cutoff_data


# Evaluate the trained Gradient Boosting models using the final test data.
def test_models(models, test_data):

    # Same historical interest features used during training
    feature_columns = [
        "interest_lag_2",
        "interest_lag_1",
        "interest"
    ]

    X_test = test_data[feature_columns]

    # Store the final test results for each prediction horizon
    results = {}

    for target, model in models.items():

        # Actual future interest values
        y_actual = test_data[target]

        # Predict future interest using data the model has not seen
        y_predicted = model.predict(X_test)

        # Calculate final prediction errors
        mae = mean_absolute_error(
            y_actual,
            y_predicted
        )

        rmse = mean_squared_error(
            y_actual,
            y_predicted
        ) ** 0.5

        results[target] = {
            "MAE": mae,
            "RMSE": rmse
        }

    return results

# Call all functions in order from top to bottom.
def main():

    # Load the geographic and emerging trend datasets
    geographic_data = load_geographic_data(
        "data/geographic.csv"
    )

    emerging_data = load_emerging_data(
        "data/emerging.csv"
    )



    # Merge geographic observations with emerging trend information
    combined_data = merge_data(
        geographic_data,
        emerging_data
    )


    # Create the 1-week, 2-week, and 4-week prediction targets
    prediction_data = create_prediction_targets(
        combined_data
    )

    # Create historical interest features
    model_data = create_lag_features(
        prediction_data
    )

    # Remove incomplete training examples
    prepared_data = prepare_model_data(
        model_data
    )


    # Split the data into training, validation, and testing sets
    train_data, validation_data, test_data = split_model_data(
        prepared_data
    )

    print("Training rows:", len(train_data))
    print("Validation rows:", len(validation_data))
    print("Testing rows:", len(test_data))


    # Train the Gradient Boosting models
    print("\nTraining Gradient Boosting models...")

    models = train_gradient_boosting(
        train_data
    )

    print("Gradient Boosting models trained")

    # Evaluate the models using the validation data
    validation_results = validate_models(
        models,
        validation_data
    )

    print("\nVALIDATION RESULTS")

    for target, metrics in validation_results.items():
        print(
            target,
            "MAE:", round(metrics["MAE"], 2),
            "RMSE:", round(metrics["RMSE"], 2)
        )

    # Evaluate the models using the final test data
    test_results = test_models(
        models,
        test_data
    )

    print("\nTEST RESULTS")

    for target, metrics in test_results.items():        
        print(
            target,
            "MAE:", round(metrics["MAE"], 2),
            "RMSE:", round(metrics["RMSE"], 2)
        )
    
    # Generate forecast using each topic's emergence date
    forecasts = forecast_from_emergence(
        models,
        model_data
    )

    forecast_columns = [
        "topic_id",
        "topic_name",
        "state",
        "date",
        "interest",
        "predicted_interest_1w",
        "predicted_interest_2w",
        "predicted_interest_4w"
    ]


    # Save all emergence-date forecasts to a CSV file
    forecasts[forecast_columns].to_csv(
        "data/geographic_forecasts.csv",
        index=False
    )

    print("\nForecasts saved to data/geographic_forecasts.csv")
    return models

if __name__ == "__main__":
    main()