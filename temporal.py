from pathlib import Path
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA = Path("data")

#GLOBAL VARIABLE
WINDOW_SIZE = 12
STEP_SIZE = 4

#HELPER FUNCTION

#load data files
def load_data():
    temporal = pd.read_csv(
        DATA / "temporal.csv",
        dtype={"topic_id": str},
        parse_dates=["date"]
    )

    events = pd.read_csv(
        DATA / "emergence_events.csv",
        dtype={"topic_id": str},
        parse_dates=["emergence_date"]
    )

    return temporal, events

#generate dataset from given data 
def create_dataset(temporal, events):
    records = []
    
    #get all the emerged events from all topics
    positive_events = set(zip(events["topic_id"],events["emergence_date"]))
    
    #loop through temporal data and get all rows for each topic
    for id, data in temporal.groupby("topic_id"):
        data = data.sort_values("date") #make sure data goes by chronological order
        #get interest data for the topic in 12 weeks and move forward by 4 weeks 
        for start in range(0, len(data) - WINDOW_SIZE+1, STEP_SIZE):
            window = data.iloc[start:start+WINDOW_SIZE]
            #get date and score
            window_start = window["date"].iloc[0]
            interest = window["interest"].to_numpy(dtype=float)
            
            #check if the window falls into the positive events list
            is_emerging = (id, window_start) in positive_events
            
            #create the record with the correct label
            record = {"id": id,
                      "name":window["topic_name"].iloc[0],
                      "start":window_start,
                      "end":window["date"].iloc[-1],
                      "y": int(is_emerging)}
            for week in range(WINDOW_SIZE):
                record[f"week_{week+1}"] = interest[week]
            #add record to dataset
            records.append(record)
    return pd.DataFrame(records)
    
#split dataset by topics (test A), then by dates (test B)
def split_dataset(dataset):
    #Test A
    # Hold out 20% of the topics. Every window from these topics
    # stays out of training.
    topic_ids = sorted(dataset["id"].unique())
    train_topics, held_out_topics = train_test_split(
        topic_ids,
        test_size=0.20,
        random_state=42
    )
    #Test B
    # Reserve the latest 20% of window start dates for testing.
    start_dates = sorted(dataset["start"].unique())
    split_index = int(len(start_dates) * 0.80)
    cutoff = pd.Timestamp(start_dates[split_index])

    familiar_topic = dataset["id"].isin(train_topics)
    unseen_topic = dataset["id"].isin(held_out_topics)

    # A training window must finish before the test period begins.
    early_window = dataset["end"] < cutoff
    later_window = dataset["start"] >= cutoff

    train = dataset[familiar_topic & early_window].copy()
    test_familiar = dataset[familiar_topic & later_window].copy()
    test_unseen = dataset[unseen_topic & later_window].copy()

    return train, test_familiar, test_unseen, cutoff

#SVM training
def train_svm(train, features):
    x = train[features]
    y = train["y"]

    model = make_pipeline(StandardScaler(),SVC(kernel="rbf",
                                               C=1.0,
                                               gamma="scale",
                                               class_weight="balanced"))

    #training the model
    model.fit(x,y)
    return model

#SVM testing
def evaluate_svm(model,test,features, test_name):
    x=test[features]
    y =test["y"]
    
    prediction = model.predict(x)
    
    print(f"\n{test_name}")
    print(f"Topics: {test["id"].nunique()}")
    print(f"Windows: {len(test)}")
    print(classification_report(
        y,
        prediction,
        labels=[0,1],
        target_names=["Non-emerging","emerging"],
        digits=3,
        zero_division=0
    ))
    
#MAIN FUNCTION

def main():
    #create and save dataset for SVM
    temporal, events = load_data()
    dataset = create_dataset(temporal,events)
    output = DATA/"svm_dataset.csv"
    
    train, test_familiar, test_unseen, cutoff = split_dataset(dataset)
    feature_columns = [f"week_{week}" for week in range(1, WINDOW_SIZE + 1)]

    print(f"Training topics: {train["id"].nunique()}")
    print(f"Training windows: {len(train)}")

    model = train_svm(train, feature_columns)

    evaluate_svm(model, test_familiar, feature_columns, "Test A - Familiar topics")
    evaluate_svm(model, test_unseen, feature_columns, "Test B - Unseen topics")

if __name__ == "__main__":
    main()
    


    