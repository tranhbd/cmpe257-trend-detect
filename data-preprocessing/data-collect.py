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
MAX_FIELD_PER_TOPIC = 5
MAX_SEARCHING_DEPTH = 3
MAX_SEARCHING_TOPICS = 100
MAX_ACCEPTED_TOPIC= 30

#Delay time to avoid rate limiting
DELAY = 5
RETRIES = 3

#Data structure
TEMPORAL_COL = ["topic_id","topic_name","date","interest"]
GEOGRAPHIC_COL = ["topic_id","topic_name","date","state","interest"]

#google trends
pytrend = TrendReq(hl="en-US",tz=480)

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
    
    