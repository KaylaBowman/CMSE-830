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

#missing vals
st.subheader("Any missing vals?")
#make a heatmap of the missing data
nan_mask = mxmh_survey_results.isna()
nan_array = nan_mask.astype(int).to_numpy()

plt.figure(figsize=(12, 6))
plt.imshow(nan_array.T, interpolation='nearest', aspect='auto', cmap='viridis')
plt.xlabel('mxmh_survey_results Index')
plt.ylabel('Features')
plt.title('Visualizing Missing Values in mxmh_survey_results Dataset')
plt.yticks(range(len(mxmh_survey_results.columns)), mxmh_survey_results.columns)
num_participants = nan_array.shape[0]
plt.xticks(np.linspace(0, num_participants-1, min(10, num_participants)).astype(int))
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
st.pyplot(plt)
