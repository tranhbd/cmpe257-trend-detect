"""
Collects temporal and geographic search data from Google Trends.

This script should only be run once to create the initial dataset.
Running it again may produce different results because Google Trends data 
and related topics change over time.

"""
import os
from pytrends.request import TrendReq
import pandas as pd

from pathlib import Path
import re
import time
from datetime import date
from dateutil.relativedelta import relativedelta
import random
import requests
from seeds import CATEGORIES, STATES
#GLOBAL VARIABLE

#directory to data
DIR_RAW = Path("./data/raw")
TOPICS_PATH = DIR_RAW / "topics.csv"
TEMPORAL_PATH = DIR_RAW / "temporal_raw.csv"
GEOGRAPHIC_PATH = DIR_RAW / "geographic_raw.csv"
MISSING_GEO_PATH = DIR_RAW/"missing_geo.csv"

#set timeframe to five years up to today
today = date.today()
start_date = today - relativedelta(years=3)
TIME = f"{start_date} {today}"

#Wikipedia
API_URL = "https://en.wikipedia.org/w/api.php"
MAX_TOPICS = 20
MAX_WORDS = 2
random.seed(42)
session = requests.Session()
session.headers.update({
    "User-Agent": "EmergingInterestsResearch/0.1 (research project)",
    "Accept": "application/json"
})

MIN_WEEKS = 104
ACTIVE_WEEKS = 8

#Delay time to avoid rate limiting
DELAY = 15
RETRIES = 3
RETRY_DELAYS = [120, 180, 600, 1200]

#Data structure
TOPIC_COL = ["topic_id", "topic_name", "category"]
TEMPORAL_COL = ["topic_id","topic_name","date","interest"]
GEOGRAPHIC_COL = ["topic_id","topic_name","date","state","interest"]
MISSING_GEO_COL = ["topic_id", "topic_name", "state", "reason"]

#google trends
pytrends = TrendReq(hl="en-US",tz=480)

#HELPER FUNCTIONS

#initialize data
def init_data(path, col):
    if not path.exists():
        return pd.DataFrame(columns = col)
    return pd.read_csv(
        path,
        dtype={"topic_id":str}
    )

#save data into csv file
def save_data(data,path,col):
    
    temp_path = path.with_suffix(".tmp")
    data[col].to_csv(
        temp_path,
        index=False,
        date_format="%Y-%m-%d"
    )
    os.replace(temp_path,path)

#standardize topic names
def standardize_topic(topic):
    #check for missing values
    if pd.isna(topic): 
        return ""
    
    #remove spaces
    str(topic).strip()
    return re.sub(r"\s+", " ", str(topic).strip()).casefold()

#filter out unnecessary keyword from wikipedia call
def filter_topic(name):
    if len(name.split()) > MAX_WORDS:
        return False

    name = name.lower()

    excluded_phrases = [
        " by country",
        " by year",
        " by decade",
        " by century",
        " by region",
        " by continent",
        " by nationality",
        " stubs",
        " templates",
        " articles",
        " pages",
        " lists",
        "competitions",
        "fiction",
        "films",
        "television",
        "novels",
        "books",
        "characters",
        "awards",
        "conferences",
        "journals",
        "magazines",
        "history of",
        "people",
        "researchers",
        "scientists",
        "organizations",
        "companies",
        "criticism",
        "problems in"
    ]

    for phrase in excluded_phrases:
        if phrase in name:
            return False

    return True

#get topics from seed
def get_topics(seed):
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{seed}",
        "cmtype": "subcat",
        "cmlimit": "max",
        "format": "json"
    }

    # Retry failures 
    for attempt in range(3):

        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=15
            )

            response.raise_for_status()
            data = response.json()
            break

        except (requests.exceptions.RequestException, ValueError) as err:

            print(f"Wikipedia request failed: {seed}: {err}")

            if attempt == 2:
                raise RuntimeError(
                    f"Could not collect Wikipedia seed: {seed}"
                ) from err

            time.sleep(60 * (attempt + 1))

    subcategories = data.get(
        "query", {}
    ).get("categorymembers", [])

    valid_topics = []

    for subcategory in subcategories:

        name = subcategory["title"].replace(
            "Category:", "", 1
        )

        if filter_topic(name):
            valid_topics.append(name)

    selected_topics = random.sample(
        valid_topics,
        min(MAX_TOPICS, len(valid_topics))
    )

    print(f"  Valid candidates: {len(valid_topics)}")
    print(f"  Selected: {len(selected_topics)}")

    return selected_topics

#explore all topics from seed topic
def collect_all_topics():
    all_topics = {}
    seen_topics = set()

    for category, seeds in CATEGORIES.items():

        print(f"\n========== {category} ==========")

        category_topics = []

        for seed in seeds:

            topics = get_topics(seed)

            for topic in topics:

                topic_key = standardize_topic(topic)

                if topic_key not in seen_topics:

                    seen_topics.add(topic_key)
                    category_topics.append(topic)

            time.sleep(5)

        all_topics[category] = category_topics

        print(
            f"Total unique topics in {category}: "
            f"{len(category_topics)}"
        )

    return all_topics

#create topics queue file to keep track of the topics visited
def create_topics_csv():
    #Cannot overwrite the file
    if TOPICS_PATH.exists():

        print("Existing topics.csv found.")
        print("Resuming previous collection.")

        return

    all_topics = collect_all_topics()

    rows = []
    topic_number = 1

    for category, topics in all_topics.items():

        for topic in topics:

            rows.append({
                "topic_id": str(topic_number),
                "topic_name": topic,
                "category": category
            })

            topic_number += 1

    topics_data = pd.DataFrame(
        rows,
        columns=TOPIC_COL
    )

    save_data(topics_data,TOPICS_PATH,TOPIC_COL)

    print(
        f"\nCreated topics.csv with "
        f"{len(topics_data)} topics."
    )

#remove topic from topic queue file
def remove_topic(topic_id):
    #open topic file
    topics_data = init_data(TOPICS_PATH,TOPIC_COL)

    #remove the topic with given topic id
    topics_data = topics_data[topics_data["topic_id"] != topic_id].copy()

    #save new topic queue to file
    save_data(topics_data,TOPICS_PATH,TOPIC_COL)

#Send request to Google Trends using PyTrend
def make_request(topic, geo="US"):
    
    for attempt in range(RETRIES):
        try:
            pytrends.build_payload(
                kw_list=[topic],
                timeframe=TIME,
                geo=geo
            )
            result = pytrends.interest_over_time()
            return result
        except Exception as err:
            print(
                f"Request faied at {topic}"
                f"({geo}), attempt {attempt+1}: {err}"
            )
            #wait and retries few time before return None
            if attempt < RETRIES -1:
                wait = RETRY_DELAYS[attempt]
                time.sleep(wait)
        finally:
            #delay to avoid rate limiting
            time.sleep(DELAY)
    return None
            
#Format the raw data
def format_temporal(data, topic, topic_id):
    
    if data is None or data.empty:
        return pd.DataFrame(columns=TEMPORAL_COL)
    
    data = data.reset_index()
    #only copy the ones that have complete time interval
    if "isPartial" in data.columns:
        data = data[data["isPartial"] == False].copy()
        
    #check and rename the column from google trends result
    if topic not in data.columns:
        return pd.DataFrame(columns=TEMPORAL_COL)
    data = data.rename(
        columns={topic: "interest"}
    )
    data["topic_id"] = topic_id
    data["topic_name"] = topic
    
    return data[TEMPORAL_COL].copy()

#validate data
def has_enough_data(data):
    if data is None or data.empty:
        return False
    #check if the topic has enough active weeks
    total_weeks = len(data)
    active_weeks = (data["interest"] > 0).sum()
    
    if total_weeks >= MIN_WEEKS and active_weeks >= ACTIVE_WEEKS:
        return True
    return False
   
#get temporal data
def get_temporal_data(topic, topic_id):
    
    #create table data
    temp = init_data(TEMPORAL_PATH,TEMPORAL_COL)
    
    #Check if the topic was already visited
    existing = temp[
        temp["topic_id"] == topic_id
    ]

    if not existing.empty:
        print("Existing temporal data.")
        return existing.copy()
    result = make_request(topic)
    
    #request fail
    if result is None:
        return None
    
    #format the topic
    formatted = format_temporal(result,topic,topic_id)
    
    #only accept topics with enough temporal data
    if not has_enough_data(formatted):
        print(f"Insufficient data")
        return pd.DataFrame(columns=TEMPORAL_COL)
    
    temp = pd.concat([temp, formatted],
                    ignore_index=True)
    save_data(temp,TEMPORAL_PATH,TEMPORAL_COL)

    print(f"Saved {len(formatted)} temporal rows.")
    return formatted
    
#collect geographic data
def get_geo_data(topic, topic_id):
    geo = init_data(GEOGRAPHIC_PATH,GEOGRAPHIC_COL)
    
    #save missing data because this part is where it likely got ratelimitng
    missing_data = init_data(MISSING_GEO_PATH,MISSING_GEO_COL)
    
    #loop through each states and call pytrends
    for state, geo_code in STATES.items(): 
        #check existing file
        existing = geo[(geo["topic_id"] == topic_id) & (geo["state"] == state)]
        if not existing.empty:
            print(f"{state} is already saved")
            continue
        
        #check if this state was confirmed missing
        missing = missing_data[
            (missing_data["topic_id"] == topic_id)
            & (missing_data["state"] == state)]
        if not missing.empty:
            print(f"no data for {state}")
            continue
        
        print(f"Collecting {state}...")
        result = make_request(topic, geo_code)
        
        #fail request
        if result is None:
            print(f"request failed for {state}"
                  "topic remains in queue")
            return False
        #format the data
        formatted = format_temporal(result,topic,topic_id)
        if formatted.empty:
            new_missing = pd.DataFrame([{
                "topic_id": topic_id,
                "topic_name": topic,
                "state": state,
                "reason": "No geographic data"
            }])
            missing_data = pd.concat([missing_data, 
                        new_missing],
                        ignore_index=True)
            save_data(missing_data,MISSING_GEO_PATH,MISSING_GEO_COL)
            print(f"{state}: no data available")
            continue
        
        #save geo data to file
        formatted["state"] = state
        formatted = formatted[GEOGRAPHIC_COL]
        geo = pd.concat([geo, formatted],
                    ignore_index=True)
        save_data(geo,GEOGRAPHIC_PATH,GEOGRAPHIC_COL)

    return True

# MAIN FUNCTION

def collect_data():
    #create the raw directory
    DIR_RAW.mkdir(parents=True, exist_ok=True)
     
    #Discover topics
    create_topics_csv()
    topics = init_data(TOPICS_PATH,TOPIC_COL)
    total = len(topics)
    print(f"Pending topics: {total}")
    
    #loop through each topics and make call to pytrend
    for index, row in topics.iterrows():
        id = str(row["topic_id"])
        topic = row["topic_name"]
        category = row["category"]
        print(f"Topic {index+1}")
        
        #get temporal data
        temporal = get_temporal_data(topic,id)
        if temporal is None:
            print("Temporal collection stop (request failure)")
            return
        if temporal.empty:
            print(f"topic {topic} rejected and removed from queue")
            remove_topic(id)
            continue
    
        #get geographic data
        geographic = get_geo_data(topic,id)
        if not geographic:
            print("Geographic collection stop (request failure)")
            return
    
        #remove topic from queue after sucessful collection
        remove_topic(id)
        print(f"Complete topic {topic}")
        
    print("\nCollection Finished!")
    
    remaining = init_data(TOPICS_PATH,TOPIC_COL)
    print(f"Remaining topics: {len(remaining)}")
    
collect_data()
        