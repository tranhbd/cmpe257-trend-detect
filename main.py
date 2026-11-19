import temporal
import geographic

import pandas as pd
import pymannkendall as mk
from pytrends.request import TrendReq

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
def get_interest(topic, cutoff):
    #request for 3 years of weekly interest, as in training
    start = cutoff - pd.DateOffset(years=3)
    timeframe = f"{start:%Y-%m-%d} {cutoff:%Y-%m-%d}"
    
    #make request to google trends using pytrend
    client = TrendReq(hl="en-US", tz=0)
    client.build_payload([topic], geo="US", timeframe=timeframe)
    history = client.interest_over_time()
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


def main():
    #training stage 1
    model = temporal.main()
    
    #user enter a new topic
    topic = input("\nEnter a topic: ").strip()
    if not topic:
        print("Please enter a topic")
        return
    
    cutoff = get_cutoff()
    #get data for entered topic
    scores = get_interest(topic,cutoff)
    #get data for latest 12 weeks
    feature_cols = [f"week_{i}" for i in range(1, 13)]
    features = pd.DataFrame([scores],columns=feature_cols)
    print("\nStage 1 - Emergence Detection")
    
    #apply the labeling rule
    print(f"12 weekly scores: {scores.tolist()}")
    reference = get_reference_label(scores)
    labels = {0: "Non-emerging", 1:"Emerging"}
    print(f"Labeling rule determines the topic is: {labels[reference]}")
    #run the model for the entered topic
    prediction = int(model.predict(features)[0])
    print(f"SVM prediction: {labels[prediction]}")
    
    if prediction == 0:
        print("No geographic forecast for this window.")
        return
    
    #print("\nStage 2 - Geographic Forecasting")
    #geographic.main()
    
if __name__ =="__main__":
    main()