import streamlit as st
import pandas as pd


#title of the app
st.title("Music Therapy App")

#dropdown menu
categories = ["The Data", "Investigate The Data", "Clean The Data", "Explore The Data"]
selected_category = st.selectbox("Choose one:", categories)

#display the selected category
st.write(f"You selected: {selected_category}")

#markdown section
st.markdown("# Load The Data")
st.markdown("* found this dataset on [Kaggle](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)")

#load the Data
mxmh_survey_results = pd.read_csv("mxmh_survey_results.csv")
    
#display the data
st.write(mxmh_survey_results.head())  

#markdown section
st.markdown("The frequency columns are categorical. We'll have to recode those to be numeric so we can run correlations.")
st.markdown("What are our features?")

#number of features
num_features = len(mxmh_survey_results.columns)
st.write(f"Number of features: {num_features}")

#names of the features
st.write("Features in the dataset:")
st.write(mxmh_survey_results.columns.tolist())  


