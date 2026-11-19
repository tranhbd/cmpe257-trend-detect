import time

import temporal
import geographic

from importlib import import_module
import pandas as pd
import pymannkendall as mk
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError

STATES = import_module("data-preprocessing.seeds").STATES

RETRIES = 2
RETRY_DELAYS = [60,180]

#HELPER FUNCTION

#get the proper cut off date for live testing
def get_cutoff():
    #get the latest fully completed week, start Sunday and end Saturday
    today = pd.Timestamp.now(tz="America/Los_Angeles").tz_localize(None).normalize()
    days_since_sunday = (today.day_of_week + 1) % 7
    this_sunday = today - pd.Timedelta(days=days_since_sunday)
    last_completed_saturday = this_sunday - pd.Timedelta(days=1)
    
    #move backward 4 weeks to evaluate the forecasts
    return last_completed_saturday-pd.Timedelta(weeks=4)

#get weekly interest score for a topic from google trends
def get_interest(topic, cutoff,geo="US"):
    #request for 3 years of weekly interest, as in training
    start = cutoff - pd.DateOffset(years=3)
    timeframe = f"{start:%Y-%m-%d} {cutoff:%Y-%m-%d}"
    
    #make request to google trends using pytrend
    client = TrendReq(hl="en-US", tz=0)
    
    for attempt in range(RETRIES):
        try:
            client.build_payload(kw_list=[topic],timeframe=timeframe,geo=geo)
            history = client.interest_over_time()
            break
            
        except TooManyRequestsError as err:
            print(
                f"Request faied at {topic}"
                f"({geo}), attempt {attempt+1}: {err}"
            )
            #wait and retries few time before return None
            if attempt < RETRIES -1:
                wait = RETRY_DELAYS[attempt]
                time.sleep(wait)
    if history.empty or topic not in history:
        raise ValueError(f"No data for {topic}.")
    
    #remove weeks that are partly complete
    if "isPartial" in history:
        history = history.loc[~history["isPartial"].astype(bool)]
    #get exact 12 weeks from the cutoff date
        last_sunday = cutoff - pd.Timedelta(days=6)
        weeks = pd.date_range(end=last_sunday, periods=12, freq="W-SUN")
        scores = history[topic].reindex(weeks)
    #catch any weeks that are removed 
    if scores.isna().any():
        raise ValueError(f"'{topic}' is missing weekly scores in the selected window.")
                
    return scores.to_numpy(dtype=float)
    

#get label for a given topic using Man-Kendall method (for live testing)
def get_reference_label(scores):
    first_mean = scores[:4].mean()
    last_mean = scores[-4:].mean()

    if first_mean == 0:
        enough_growth = last_mean > 0
    else:
        enough_growth = last_mean >= first_mean * 1.25

    if not enough_growth:
        return 0
    result = mk.hamed_rao_modification_test(scores)
    
    return int(result.p < 0.05 and result.slope > 0)

#MAIN FUNCTION

def main():
    #training and testing the model
    print("\nModel Preparing")
    #training stage 1
    svm_model = temporal.main()
    #training stage 2
    geo_model = geographic.main()
    print("\nComplete training the model!")
    
    #user enter a new topic
    topic = input("\nEnter a topic: ").strip()
    if not topic:
        print("Please enter a topic")
        return
    
    #stage 1
    cutoff = get_cutoff()
    #get data for entered topic
    scores = get_interest(topic,cutoff)
    #get data for latest 12 weeks
    feature_cols = [f"week_{i}" for i in range(1, 13)]
    features = pd.DataFrame([scores],columns=feature_cols)
    
    print(f"Cutoff date:{cutoff}")
    print("\nStage 1 - Emergence Detection")
    
    #apply the labeling rule
    print(f"12 weekly scores: {scores.tolist()}")
    reference = get_reference_label(scores)
    labels = {0: "Non-emerging", 1:"Emerging"}
    print(f"Labeling rule determines the topic is: {labels[reference]}")
    #run the svm model for the entered topic
    prediction = int(svm_model.predict(features)[0])
    print(f"SVM prediction: {labels[prediction]}")
    
    if prediction == 0:
        print("No geographic forecast for this window.")
        return
    
    #stage 2
    print("\nStage 2 - Geographic Forecasting")
    #get data for each state in the states list using pytrend
    rows = []
    after_cutoff = cutoff + pd.Timedelta(weeks=4)
    for state, geo_code in STATES.items():
        time.sleep(10)
        scores = get_interest(topic,after_cutoff,geo=geo_code)
        #last 4 weeks of score is for validate
        inputs = scores[-7:-4]
        actuals = scores[-4:]
        print(f"{state} actual scores, weeks 1 to 4: {actuals.tolist()}")
        rows.append({
            "state":state,
            "interest_lag_2": inputs[-3], #2 weeks before cutoff
            "interest_lag_1": inputs[-2], #1 week before cutoff
            "interest":inputs[-1], #cutoff week
            "actual_1w": actuals[0],# First week after the cutoff
            "actual_2w": actuals[1],
            "actual_4w": actuals[3],
        })
        
    #format the data
    state_data = pd.DataFrame(rows)
    features = state_data[["interest_lag_2", "interest_lag_1", "interest"]]
    
    #run the geo model for 1,2 and 4 weeks ahead per state
    forecasts = state_data[["state"]].copy()
    for weeks in [1, 2, 4]:
        model = geo_model[f"interest_{weeks}w"]
        forecasts[f"predicted_{weeks}w"] = model.predict(features)
        forecasts[f"actual_{weeks}w"] = state_data[f"actual_{weeks}w"]
    #print result
    print(forecasts.round(1).to_string(index=False))
if __name__ =="__main__":
    main()