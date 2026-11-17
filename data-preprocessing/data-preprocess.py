"""
This script preprocesses the data collected by collection.py

It identifies emerging topics using the Hamed-Rao modifed Mann-Kendall test
and Sen's slope on non-overlapping 12-week windows. The first window with a
statistically significant positive trend determines the emergence date.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pymannkendall as mk
import warnings

#GLOBAL VARIABLE

#file path
RAW = Path("data/raw")
OUT = Path("data")
TEMP_PATH = RAW/"temporal_raw.csv"
GEO_PATH = RAW/"geographic_raw.csv"

#labeling
WINDOW_SIZE = 12
SIGNIFICANCE_LEVEL = 0.05
MIN_INCREASE = 0.30

#columns
TEMP_COL = ["topic_id","topic_name","date","interest"]
GEO_COL = ["topic_id","topic_name","date","state","interest"]
EMERGING_COL =["topic_id","topic_name","emerged","emergence_date"]
EVENT_COL = ["topic_id", "topic_name", "emergence_date"]
#HELPER FUNCTION

#read csv file
def load_csv(path, col):
    data = pd.read_csv(path, dtype={"topic_id":str})
    
    #check if there are missing columns
    missed_col = set(col) - set(data.columns)
    if missed_col:
        raise ValueError(f"Missing column: {missed_col}")
    
    return data[col].copy()

#format data and remove duplicate
def clean_data(data, keys):
    d = data.copy()
    
    #Convert date to datetime
    d["date"] = pd.to_datetime(data["date"],errors="coerce")
    #convert interest to numeric
    d["interest"] = pd.to_numeric(data["interest"],errors="coerce")
    
    if d.isna().any().any():
        raise ValueError("Missing or invalid values found")
    
    #check duplicate rows
    if d.duplicated(subset=keys).any():
        raise ValueError("Duplicate rows found")
    
    return d

#find emerging date
def find_emergence_date(topic):
    
    #sort to make sure weekly data is in chronological order
    topic = topic.sort_values("date").reset_index(drop=True)
    emergence_dates = []
    #loop through each windows of weeks, 
    # then perfrom Mann-Kendall tests
    for start in range(WINDOW_SIZE,len(topic) - WINDOW_SIZE +1,4):
        prev_window = topic.iloc[start-WINDOW_SIZE:start]
        cur_window = topic.iloc[start:start+WINDOW_SIZE]
        prev_values = prev_window["interest"].to_numpy()
        cur_values = cur_window["interest"].to_numpy()

        #average interest of both windows
        prev_average = np.mean(prev_values)
        cur_average = np.mean(cur_values)
        
        #if previous window is 0
        if prev_average == 0:
            if cur_average > 0:
                increase = float("inf")
            else:
                increase = 0
        else:
            increase =(cur_average-prev_average)/prev_average
        
        #the current window must increase by at least 25%
        if increase < MIN_INCREASE:
            continue  
        #skip windows with indentical interest values
        if len(set(cur_values)) == 1:
            continue
        
        #run the windows with Hamed_Rao
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)

                result = mk.hamed_rao_modification_test(cur_values,alpha=SIGNIFICANCE_LEVEL)

        except RuntimeWarning:
            print(f"Numerical warning for topic: {topic.iloc[0]['topic_name']}")
            print(f"Window values: {cur_values}")
            continue     
           
        #return the windows that the trend emerged
        if(result.p<SIGNIFICANCE_LEVEL and result.slope >0):
            date = cur_window.iloc[0]["date"].strftime("%Y-%m-%d")
            emergence_dates.append(date)
    return sorted(set(emergence_dates))

#create emerging labels
def emerging_label(temp):
    emerging_rows = []
    event_rows = []
    
    #loop through each topic (row group) to find emergence date
    for topic_id, topic_data in temp.groupby("topic_id"):
        #get topic name
        topic_name = topic_data.iloc[0]["topic_name"]
        
        emergence_dates = find_emergence_date(topic_data)
        if emergence_dates:
            emerged = 1
            #some topic have more than one emergence date, for
            #stage 2 return the latest period 
            latest_date = emergence_dates[-1]
        else:
            emerged = 0
            latest_date=""
        #Save all the emergence dates for the topics for stage 1
        for emergence_date in emergence_dates:
            event_rows.append({
                "topic_id": topic_id,
                "topic_name": topic_name,
                "emergence_date": emergence_date
            })
            
        #Save the latest emerged date for the topic for stage 2
        emerging_rows.append({
            "topic_id": topic_id,
            "topic_name": topic_name,
            "emerged": emerged,
            "emergence_date":latest_date
        })
    #format the data
    emerging = pd.DataFrame(
        emerging_rows,
        columns=EMERGING_COL
    )

    events = pd.DataFrame(
        event_rows,
        columns=EVENT_COL
    )

    return emerging, events

#MAIN FUNCTION 

def get_emerging():
    #load data
    temporal = load_csv(TEMP_PATH,TEMP_COL)
    geographic = load_csv(GEO_PATH,GEO_COL)
    
    #clean data
    temporal = clean_data(temporal,["topic_id","date"])
    geographic = clean_data(geographic,["topic_id", "date", "state"])
    
    #labeling
    emerging, events = emerging_label(temporal)
    # Sort topic_id numerically instead of alphabetically
    emerging = emerging.sort_values("topic_id",key=lambda col: pd.to_numeric(col)).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    #save cleaned temporal and geographic files
    temporal.to_csv(OUT/"temporal.csv",index=False,date_format="%Y-%m-%d")
    geographic.to_csv(OUT/"geographic.csv",index=False,date_format="%Y-%m-%d")
    #save emergin labels for stage 2
    emerging.to_csv(OUT/"emerging.csv",index=False)
    #save emerging dates for stage 1
    events.to_csv(OUT/"emergence_events.csv",index=False)
    print(f"Temporal rows: {len(temporal)}")
    print(f"Geographic rows: {len(geographic)}")
    print(f"Total topics: {len(emerging)}")
    print(f"Emerging topics: {emerging['emerged'].sum()}")
    print(f"Total emergence events: {len(events)}")

    
get_emerging()