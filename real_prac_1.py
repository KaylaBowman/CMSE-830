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
st.subheader("What does the data look like?")
st.markdown("* found this dataset on [Kaggle](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)")

#load the Data
mxmh_survey_results = pd.read_csv("mxmh_survey_results.csv")
    
#display the data
st.write(mxmh_survey_results.head())  

#markdown section
st.subheader("Data Types")
st.write(Data Types:

* Timestamp: Interval 
* Age: Ratio
* Primary streaming service: Nominal
* Hours per day: Ratio
* While working: Binary (Yes, No)
* Instrumentalist: Binary (Yes, No)
* Composer: Binary (Yes, No)
* Fav genre: Nominal
* Exploratory: Binary (Yes, No)
* Foreign languages: Binary (Yes, No)
* BPM: Ratio 
* Frequency [Classical]: Ordinal (Never, Rarely, Sometimes, Very Frequently)
* Frequency [Country]: Ordinal (Never, Rarely, Sometimes, Very Frequently)
* Frequency [EDM]: Ordinal (Never, Rarely, Sometimes, Very Frequently)
* Frequency [Folk]: Ordinal (Never, Rarely, Sometimes, Very Frequently)
* Frequency [Gospel]: Ordinal (Never, Rarely, Sometimes, Very Frequently)
* Frequency [Hip hop]: Ordinal (Never, Rarely, Sometimes, Very Frequently)
* Frequency [Jazz]: Ordinal (Never, Rarely, Sometimes, Very Frequently)
* Frequency [K pop]: Ordinal (Never, Rarely, Sometimes, Very Frequently)
* Frequency [Latin]: Ordinal (Never, Rarely, Sometimes, Very Frequently)
* Frequency [Lofi]: Ordinal (Never, Rarely, Sometimes, Very Frequently)
* Frequency [Metal]: Ordinal (Never, Rarely, Sometimes, Very Frequently)
* Frequency [Pop]: Ordinal (Never, Rarely, Sometimes, Very Frequently)
* Frequency [R&B]: Ordinal (Never, Rarely, Sometimes, Very Frequently)
* Frequency [Rap]: Ordinal (Never, Rarely, Sometimes, Very Frequently)
* Frequency [Rock]: Ordinal (Never, Rarely, Sometimes, Very Frequently)
* Frequency [Video game music]: Ordinal (Never, Rarely, Sometimes, Very Frequently)
* Anxiety: Ordinal (Likert scale, 1-10)
* Depression: Ordinal (Likert scale, 1-10)
* Insomnia: Ordinal (Likert scale, 1-10)
* OCD: Ordinal (Likert scale, 1-10)
* Music effects: Ordinal (No effect, Improve, Worsen)
* Permissions: Binary (Yes, No) )  


