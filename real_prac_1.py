import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

#title of the app
st.title("Welcome To My Music Therapy App")

#dropdown menu
categories = ["The Data", "Investigate The Data", "Clean The Data", "Explore The Data"]
selected_category = st.selectbox("Choose one:", categories)


if selected_category == "The Data":
    
    #display the selected category
    st.write(f"You selected: {selected_category}")
    
    #markdown section
    st.subheader("What does the data look like?")
    st.markdown("* found this dataset on [Kaggle](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)")
    
    #load the Data
    mxmh_survey_results = pd.read_csv("mxmh_survey_results.csv")
        
    #display the data
    st.write(mxmh_survey_results.head())  

if selected_category == "Investigate The Data":

    #load the Data
    mxmh_survey_results = pd.read_csv("mxmh_survey_results.csv")
    
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
    st.header("Distribution of Features")
    
    st.subheader("Age")
    #age 
    fig = px.histogram(mxmh_survey_results, x="Age", title="Age Distribution")
    st.plotly_chart(fig)
    
    st.subheader("Streaming Service")
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
    
    st.subheader("Favorite Genre")
    #fav genre
    plt.figure(figsize=(10, 6))  
    plt.hist(mxmh_survey_results["Fav genre"], bins=16, edgecolor='black')
    
    #set the title of the plot
    plt.title('Distribution of Fav Genre', fontsize=16)
    
    #set the x-axis title
    plt.xlabel('Fav Genre', fontsize=12)
    plt.xticks(rotation=45) 
    
    st.pyplot(plt)
    
    st.subheader("Mental Health Stats")
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
    
    #Experts
    st.subheader("Experts")
    
    fig, ax = plt.subplots()
    sns.histplot(data=mxmh_survey_results, x="Composer", bins=2, label="Composers", multiple="stack", ax=ax)
    sns.histplot(data=mxmh_survey_results, x="Instrumentalist", bins=2, label="Instrumentalists", ax=ax)
    
    ax.legend()
    
    ax.set_title("Distribution of Composers and Instrumentalists")
    
    st.pyplot(fig)
    
    #Music Effects
    st.subheader("Music Effects")
    fig, ax = plt.subplots()
    sns.histplot(data=mxmh_survey_results, x='Music effects', hue='Music effects', palette=['red', 'blue', 'green'])
    ax.legend()
    ax.set_title("Distribution of Perceived Music Effects")
    st.pyplot(fig)
    
    #look at outliers
    st.subheader("Any Outliers?")
    
    fig, ax = plt.subplots()
    sns.histplot(data=mxmh_survey_results, x='Hours per day')
    ax.legend()
    ax.set_title("Hours Per Day")
    st.pyplot(fig)
    
    #hour outliers
    num_24_hours = sum(mxmh_survey_results['Hours per day'] == 24)
    st.write(f"Number of participants reporting 24 hours per day: {num_24_hours}")
    
    #age outliers:
    age_outliers = sum((mxmh_survey_results['Age'] > 70) | (mxmh_survey_results['Age'] < 18))
    st.write(f"Number of participants younger than 18 or older than 70: {age_outliers}")

if selected_category == "Clean The Data":

    #load the Data
    mxmh_survey_results = pd.read_csv("mxmh_survey_results.csv")
    
    #handle missing vals 
    st.subheader("Handle BPM Missing Values")
    st.markdown("Group by genre, then replace with median of genre")

    #get pop median so we can test our replacement worked
    pop_median = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "Pop"]["BPM"].median()
    st.write(f"The median BPM of Pop: {pop_median}")

    #group and replace
    for i, val in enumerate(mxmh_survey_results["BPM"].isna()):
        genre = mxmh_survey_results.loc[i, "Fav genre"]  # Get the genre for the current row
        if genre == "Latin":
            mxmh_survey_results.loc[i, "BPM"] = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "Latin"]["BPM"].median()
        if genre == "Rock":
            mxmh_survey_results.loc[i, "BPM"] = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "Rock"]["BPM"].median()
        if genre == "Video game music":
            mxmh_survey_results.loc[i, "BPM"] = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "Video game music"]["BPM"].median()
        if genre == "Jazz":
            mxmh_survey_results.loc[i, "BPM"] = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "Jazz"]["BPM"].median()
        if genre == "R&B":
            mxmh_survey_results.loc[i, "BPM"] = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "R&B"]["BPM"].median()
        if genre == "K pop":
            mxmh_survey_results.loc[i, "BPM"] = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "K pop"]["BPM"].median()
        if genre == "Country":
            mxmh_survey_results.loc[i, "BPM"] = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "Country"]["BPM"].median()
        if genre == "EDM":
            mxmh_survey_results.loc[i, "BPM"] = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "EDM"]["BPM"].median()
        if genre == "Hip hop":
            mxmh_survey_results.loc[i, "BPM"] = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "Hip hop"]["BPM"].median()
        if genre == "Pop":
            mxmh_survey_results.loc[i, "BPM"] = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "Pop"]["BPM"].median()
        if genre == "Rap":
            mxmh_survey_results.loc[i, "BPM"] = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "Rap"]["BPM"].median()
        if genre == "Classical":
            mxmh_survey_results.loc[i, "BPM"] = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "Classical"]["BPM"].median()
        if genre == "Metal":
            mxmh_survey_results.loc[i, "BPM"] = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "Metal"]["BPM"].median()
        if genre == "Folk":
            mxmh_survey_results.loc[i, "BPM"] = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "Folk"]["BPM"].median()
        if genre == "Lofi":
            mxmh_survey_results.loc[i, "BPM"] = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "Lofi"]["BPM"].median()
        if genre == "Gospel":
            mxmh_survey_results.loc[i, "BPM"] = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "Gospel"]["BPM"].median()

    #see that the values were replaced
    filtered_data = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "Pop"]
    st.write(filtered_data.head())  

    #see that BPM has no missing values
    #missing vals
    st.subheader("No More Missing BPM Vals")
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

    st.subheader("Handle Outliers")
    #I don't trust the participants who say they listen to music 24hrs/day
    cleaned_data = mxmh_survey_results.copy()
    #I will say the max they could realistically listen to is 16 hrs
    cleaned_data = cleaned_data[(cleaned_data["Hours per day"] < 16)]
    #deleted 6 rows
    st.markdown("Deleted all instances of Hours Per Day above 16")
    cleaned_data.shape

    #take away age outliers 
    cleaned_data = cleaned_data[(cleaned_data["Age"] > 18) & (cleaned_data["Age"] < 64)]
    st.markdown("Deleted all instances of Age < 18 and Age > 64 (3 SDs from the 75% percentile)")
    cleaned_data.shape
    
    #recode frequency genre
    st.subheader("Recode Categorical Data")
    st.markdown("Genre Frequencies")

    frequency_mapping = {
    "Never": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Very frequently": 4 }

    # Replace the values in the "Frequency [Country]" column
    cleaned_data["Frequency [Latin]"] = cleaned_data["Frequency [Latin]"].replace(frequency_mapping)
    cleaned_data["Frequency [Rock]"] = cleaned_data["Frequency [Rock]"].replace(frequency_mapping)
    cleaned_data["Frequency [Video game music]"] = cleaned_data["Frequency [Video game music]"].replace(frequency_mapping)
    cleaned_data["Frequency [Jazz]"] = cleaned_data["Frequency [Jazz]"].replace(frequency_mapping)
    cleaned_data["Frequency [R&B]"] = cleaned_data["Frequency [R&B]"].replace(frequency_mapping)
    cleaned_data["Frequency [K pop]"] = cleaned_data["Frequency [K pop]"].replace(frequency_mapping)
    cleaned_data["Frequency [Country]"] = cleaned_data["Frequency [Country]"].replace(frequency_mapping)
    cleaned_data["Frequency [EDM]"] = cleaned_data["Frequency [EDM]"].replace(frequency_mapping)
    cleaned_data["Frequency [Hip hop]"] = cleaned_data["Frequency [Hip hop]"].replace(frequency_mapping)
    cleaned_data["Frequency [Pop]"] = cleaned_data["Frequency [Pop]"].replace(frequency_mapping)
    cleaned_data["Frequency [Rap]"] = cleaned_data["Frequency [Rap]"].replace(frequency_mapping)
    cleaned_data["Frequency [Classical]"] = cleaned_data["Frequency [Classical]"].replace(frequency_mapping)
    cleaned_data["Frequency [Metal]"] = cleaned_data["Frequency [Metal]"].replace(frequency_mapping)
    cleaned_data["Frequency [Folk]"] = cleaned_data["Frequency [Folk]"].replace(frequency_mapping)
    cleaned_data["Frequency [Lofi]"] = cleaned_data["Frequency [Lofi]"].replace(frequency_mapping)
    cleaned_data["Frequency [Gospel]"] = cleaned_data["Frequency [Gospel]"].replace(frequency_mapping)

    #see the changes
    st.write(cleaned_data.head())  

if selected_category == "Explore The Data":
    st.subheader("Any correlations between frequency and mental health?")

    selected_features = ['Frequency [Classical]', "Frequency [Country]", "Frequency [EDM]", "Frequency [Folk]", 
                     "Frequency [Gospel]", "Frequency [Hip hop]", "Frequency [Jazz]", "Frequency [K pop]", "Frequency [Latin]", "Frequency [Lofi]",
                     "Frequency [Metal]", "Frequency [Pop]", "Frequency [R&B]", "Frequency [Rap]", "Frequency [Rock]",  "Anxiety", "Depression", "Insomnia", "OCD"] # Focus on these variables

    # Correlation Heatmap (Interactive)
    correlation_matrix = cleaned_data[selected_features].corr().values
    fig_heatmap = ff.create_annotated_heatmap(
         z=correlation_matrix,
         x=selected_features,
         y=selected_features,
         colorscale='Viridis'
     )
    fig_heatmap.update_layout(
        title="Correlation Heatmap (Interactive)",
        xaxis_title="Features",
        yaxis_title="Features"
    )
    st.plotly_chart(fig_heatmap)
    

    st.subheader("How does mental health vary across age?")


    bins = [18, 25, 31, 36, 41, 46, 51, 58, 64]  
    labels = ['18-24', '25-30', '31-35', '36-40', '41-45', '46-50', '51-57', '58-64']  # Labels for the bins

    # Create the binned column
    cleaned_data['age_binned'] = pd.cut(cleaned_data['Age'], bins=bins, labels=labels, right=False)


    #now plot it 
    fig_violin = px.violin(cleaned_data, x='age_binned', y='Anxiety', box=True, points='all',
                           labels={'Age':'Age', 'Anxiety':'Anxiety'},
                           title="Interactive Violin Plot of Age vs Anxiety")

    st.plotly_chart(fig_violin)


   
