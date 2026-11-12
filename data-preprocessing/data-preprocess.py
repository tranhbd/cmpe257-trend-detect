"""
This script preprocesses the data collected by collection.py

It identifies emerging topics using the Hamed-Rao modifed Mann-Kendall test
and Sen's slope on non-overlapping 12-week windows. The first window with a
statistically significant positive trend determines the emergence date.
"""
import pandas as pd
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

#columns
TEMP_COL = ["topic_id","topic_name","date","interest"]
GEO_COL = ["topic_id","topic_name","date","state","interest"]
EMERGING_COL =["topic_id","topic_name","emerged","emergence_date"]

#HELPER FUNCTION

#read csv file
def load_csv(path, col):
    data = pd.read_csv(path, dtype={"topic_id":str})
    
    #check if there are missing columns
    missed_col = set(col) - set(data.columns)
    if missed_col:
        raise ValueError(f"Missing column: {missed_col}")
    
    return data[col].copy()

#clean data 
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
    #loop through each row of same topic, divide them
    #by windows of weeks, then perfrom Mann-Kendall tests
    for start in range(0,len(topic) - WINDOW_SIZE +1,WINDOW_SIZE):
        window = topic.iloc[start:start+WINDOW_SIZE]
        values = window["interest"].to_numpy()
        
        #skip windows with indentical interest values
        if len(set(values)) == 1:
            continue
        
        #run the windows with Hamed_Rao
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)

                result = mk.hamed_rao_modification_test(values,alpha=SIGNIFICANCE_LEVEL)

        except RuntimeWarning:
            print(f"Numerical warning for topic: {topic.iloc[0]['topic_name']}")
            print(f"Window values: {values}")
            continue     
           
        #return the last date of the rising/emerging window
        if(result.p<SIGNIFICANCE_LEVEL and result.slope >0):
            return window.iloc[-1]["date"].strftime("%Y-%m-%d")
    return ""

#create emerging labels
def emerging_label(temp):
    rows = []
    
    #loop through each topic (row group) to find emergence date
    for topic_id, topic_data in temp.groupby("topic_id"):
        #get topic name
        topic_name = topic_data.iloc[0]["topic_name"]
        
        emergence_date = find_emergence_date(topic_data)
        if emergence_date:
            emerged = 1
        else:
            emerged = 0
            
        #adding to emerging.csv file
        rows.append({
            "topic_id": topic_id,
            "topic_name": topic_name,
            "emerged": emerged,
            "emergence_date":emergence_date
        })
    
    return pd.DataFrame(rows, columns=EMERGING_COL)

#MAIN FUNCTION 

def get_emerging():
    #load data
    temporal = load_csv(TEMP_PATH,TEMP_COL)
    geographic = load_csv(GEO_PATH,GEO_COL)
    
    #clean data
    temporal = clean_data(temporal,["topic_id","date"])
    geographic = clean_data(geographic,["topic_id", "date", "state"])
    
    #labeling
    emerging = emerging_label(temporal)
    
    #save cleaned temporal and geographic files
    temporal.to_csv(OUT/"temporal.csv",index=False,date_format="%Y-%m-%d")
    geographic.to_csv(OUT/"geographic.csv",index=False,date_format="%Y-%m-%d")
    #save emergin labels
    emerging.to_csv(OUT/"emerging.csv",index=False)
    
    print(f"Temporal rows: {len(temporal)}")
    print(f"Geographic rows: {len(geographic)}")
    print(f"Total topics: {len(emerging)}")
    print(f"Emerging topics: {emerging['emerged'].sum()}")
    
get_emerging()