from pathlib import Path
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support


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
    # Hold out 20% of the topics. Every window from these topics
    # stays out of training.
    topic_ids = sorted(dataset["id"].unique())
    train_topics, test_topics = train_test_split(
        topic_ids,
        test_size=0.20,
        random_state=42
    )
    #keep all windows of test topics separate from train topics
    train = dataset[dataset["id"].isin(train_topics)].copy()
    test = dataset[dataset["id"].isin(test_topics)].copy()

    return train, test

#SVM training
def train_svm(train, features,label):
    ids = sorted(train["id"].unique())

    #split the dataset to training and validation set
    fit_topics, validate_topics = train_test_split(
        ids,
        test_size=0.20,
        random_state=42,
    )
    #keep all windows of test topics separate from train topics
    fit_set = train[train["id"].isin(fit_topics)].copy()
    validate_set = train[train["id"].isin(validate_topics)].copy()
    print(f"Using {len(fit_topics)} topics for training "
          f"{len(validate_topics)} for validation")
    
    model = make_pipeline(StandardScaler(),SVC(kernel="rbf",
                                               C=1.0,
                                               gamma="scale",
                                               class_weight="balanced"))
    best_c = 1
    best_f1 = -1
    #training and validating the model using diffrent C=1,0.1,and 10
    print("\n Validation resut")
    print(f"{'C':>5} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for c in [1, 0.1,10,100]:
        model.set_params(svc__C=c)
        model.fit(fit_set[features],fit_set[label])
        predictions = model.predict(validate_set[features])

        precision, recall, f1, _ = precision_recall_fscore_support(
            validate_set[label],
            predictions,
            average="binary",
            pos_label=1,
            zero_division=0
        )
       
        print(f"{c:>5g} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}")
        
        #choosing model with best f1
        if f1 > best_f1:
            best_f1 = f1
            best_c = c
    #print the best C value
    print(f"\n Selected C: {best_c}")
    
    #refit the svm on all 192 topics
    model.set_params(svc__C=best_c)
    model.fit(train[features],train[label])
    
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
    
    train, test = split_dataset(dataset)
    features = [f"week_{week}" for week in range(1, WINDOW_SIZE + 1)]

    print(f"Training topics: {train["id"].nunique()}")
    print(f"Training windows: {len(train)}")

    model = train_svm(train, features,label="y")

    evaluate_svm(model, test, features,"Test - Unseen topics")
    return model

if __name__ == "__main__":
    main()
    


    