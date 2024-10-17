import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

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

#distribution 
st.subheader("Distribution of Features")

st.markdown("Age")
#age 
fig = px.histogram(mxmh_survey_results, x="Age", title="Age Distribution")
st.plotly_chart(fig)

st.markdown("Streaming Service")
#streaming service
platforms = ['Spotify', 'Pandora', 'YouTube Music', 
             'I do not use a streaming service.', 
             'Apple Music', 'Other streaming service']
popularity = [458, 11, 94, 71, 51, 0]

#create a horizontal bar plot
plt.figure(figsize=(10, 6))  # Set the figure size
plt.barh(platforms, popularity, color='skyblue')
plt.title('Distribution of Primary Streaming Service')
plt.xlabel('Popularity')
plt.ylabel('Streaming Service')

st.pyplot(plt)

st.markdown("Favorite Genre")
#fav genre
plt.figure(figsize=(10, 6))  
plt.hist(mxmh_survey_results["Fav genre"], bins=16, edgecolor='black')

#set the title of the plot
plt.title('Distribution of Fav Genre', fontsize=16)

#set the x-axis title
plt.xlabel('Fav Genre', fontsize=12)
plt.xticks(rotation=45) 

st.pyplot(plt)

st.markdown("Mental Health Stats")
#anxiety 
fig = px.histogram(mxmh_survey_results, x="Anxiety", title="Anxiety Distribution")
st.plotly_chart(fig)

#depression
fig = px.histogram(mxmh_survey_results, x="Depression", title="Depression Distribution")
st.plotly_chart(fig)

#insomnia
fig = px.histogram(mxmh_survey_results, x="Insomnia", title="Insomnia Distribution")
st.plotly_chart(fig)

#OCD
fig = px.histogram(mxmh_survey_results, x="OCD", title="OCD Distribution")
st.plotly_chart(fig)


#frequency
st.markdown("Genre Frequency")

fig = px.histogram(mxmh_survey_results, x=('Frequency [Latin]'), title="Frequency of Latin Listeners")
st.plotly_chart(fig)

fig = px.histogram(mxmh_survey_results, x=('Frequency [Rock]'), title="Frequency of Rock Listeners")
st.plotly_chart(fig)

fig = px.histogram(mxmh_survey_results, x=('Frequency [Classical]'), title="Frequency of Latin Listeners")
st.plotly_chart(fig)

fig = px.histogram(mxmh_survey_results, x=('Frequency [Latin]'), title="Frequency of Pop Listeners")
st.plotly_chart(fig)

