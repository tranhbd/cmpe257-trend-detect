"""
Collects temporal and geographic search data from Google Trends.

This script should only be run once to create the initial dataset.
Running it again may produce different results because Google Trends data 
and related topics change over time.

"""

from pytrends.request import TrendReq
import pandas as pd

from pathlib import Path
from collections import deque
import re
import time
#GLOBAL VARIABLE

#directory to data
DIR_RAW = Path("data/raw")
TEMPORAL_PATH = DIR_RAW/"raw_temporal.csv"
GEOGRAPHIC_PATH = DIR_RAW/"geographic_raw.csv"

#set timeframe to five years up to today
TIME = "today 3-y"

#Seed topics *****start with 1 seed per group for now, change later
SEED_TOPICS = {
    "technology":["smart glasses"],
    "entertainment":["virtual reality gaming"],
    "travel": ["solo travel"],
    "food":["matcha"],
    "consumer products":["air fryer"]
}

#Geographic area ****start with 5 states for testing, change later
STATES = {
    "CA": "US-CA",
    "TX": "US-TX",
    "NY": "US-NY",
    "FL": "US-FL",
    "IL": "US-IL"
}

#Limit topics discover ****change later
MAX_RELATED_PER_TOPIC = 5
MAX_SEARCHING_DEPTH = 3
MAX_ATTEMPTED_TOPICS = 100
MAX_CANDIDATES= 30

#Delay time to avoid rate limiting
DELAY = 5
RETRIES = 3

#Data structure
TEMPORAL_COL = ["topic_id","topic_name","date","interest"]
GEOGRAPHIC_COL = ["topic_id","topic_name","date","state","interest"]

#google trends
pytrends = TrendReq(hl="en-US",tz=480)

#HELPER FUNCTIONS

#initialize data
def init_data(path, col):
    if not path.exists():
        return pd.DataFrame(columns = col)
    return pd.read_csv(
        path,
        dtype={"topic:id":str}
    )
#save data into csv file
def save_data(data,path,col):
    data[col].to_csv(
        path,
        index=False,
        date_format="%Y-%m-%d"
    )

#standardize topic names
def standardize_topic(topic):
    #check for missing values
    if pd.isna(topic): 
        return ""
    
    #remove spaces
    str(topic).strip()
    re.sub(r"\s+", " ", str(topic).strip()).casefold()

#Send request to Google Trends using PyTrend
def make_request(topic, request_type, geo="US"):
    
    for attempt in range(RETRIES):
        try:
            pytrends.build_payload(
                kw_list=[topic],
                timeframe=TIME,
                geo=geo
            )
            if request_type == "temporal":
                result = pytrends.interest_over_time()
            elif request_type == "related":
                result = pytrends.related_topics()
            else:
                raise ValueError("Invalid request type!")
            
            #delay request to avoid rate limiting
            time.sleep()
            return result
        except Exception as err:
            print(
                f"Request faied at {topic}"
                f"({geo}), attempt {attempt+1}: {err}"
            )
            #wait and retries few time before return None
            if attempt < RETRIES -1:
                time.sleep(10*(attempt + 1))
    return None
            
#Format the raw data
def format_temporal(data, topic, topic_id, search_value):
    
    if data is None or data.empty:
        return pd.DataFrame(columns=TEMPORAL_COL)
    
    data = data.reset_index()
    #only copy the ones that have complete time interval
    if "isPartial" in data.columns:
        data = data[data["isPartial"] == False].copy()
        
    #check and rename the column from google trends result
    if search_value not in data.columns:
        return pd.DataFrame(columns=TEMPORAL_COL)
    data = data.rename(
        columns={search_value: "interest"}
    )
    data["topic_id"] = topic_id
    data["topic_name"] = topic
    
    return data[TEMPORAL_COL].copy()

#MAIN FUNCTION
   
#Searching for topics and get temporal data
def get_temporal_data():
    #create the raw directory
    DIR_RAW.mkdir(parents=True, exist_ok=True)
    
    #create table data
    temp = init_data(TEMPORAL_PATH,TEMPORAL_COL)
    geo = init_data(GEOGRAPHIC_PATH,GEOGRAPHIC_COL)
    
    #keeping queue to avoid duplicates
    queue = deque()
    visited =set()
    
    accepted_topics = []
    ids = {}
    
    next_id = 1
    
    #initialize BFS with seed topics
    for category, seeds in SEED_TOPICS.items():
        for seed in seeds:
            topic_name = standardize_topic(seed)
        #check for empty or duplicates topic name 
        if not topic_name or topic_name in visited:
            continue
        
        queue.append({
            "topic_name": topic_name,
            "topic_mid": None,
            "category": category,
            "depth": 0
        })
        
        visited.add(topic_name)
    
    attempted = 0
    
    #keep searching for topics using seed topics until one of the
    #restriction meets
    while(
        queue
        and len(accepted_topics) < MAX_CANDIDATES
        and attempted < MAX_ATTEMPTED_TOPICS
    ):
        current = queue.popleft()
        
        name = current["topic_name"]
        mid = current["topic_mid"]
        depth = current["depth"]
        
        search_value = (
           mid if mid is not None and pd.notna(mid)
           else name
        )
        attempted+=1
        
        print(
            f"\nattempt {attempted}: {name}"
            f"(depth {depth})"
        )
        
        #collect the topic from google trend
        topic_id = f"T{next_id:03d}"
        
        raw_result = make_request(search_value, request_type="temporal")
        temporal = format_temporal(
            raw_result,
            name,
            topic_id,
            search_value
        )    