# Disaster Response Pipeline Project
During any disaster or afterwards, people can use text messages in case of any help needed. This text messages have to be classified and transmitted to the right response organizations. By using predictive modelling, I tried to
classify this text messages.


In this project, I built an ETL pipeline that cleaned messages using NLP pipeline. Random forest model was built to classify the messages. Finally Flask Web App is prepared which shows the related category when a message is written and shows 3 visualizations of the train dataset.

### Files:
- `process_data.py` : ETL script to clean data
- `train_classifier.py` : Script to tokenize messages from clean data also includes ML pipeline. Model is saved as pickle file.
- `run.py` :Script to run Flask Web app that classifies messages  and shows visualizations.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://localhost:3001/
