#import packages

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs as go 
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
#import the undersampling package
from imblearn.under_sampling import RandomUnderSampler


##title of the app
##st.title("Welcome To Tunes By Mood: A Music Therapy App Designed For You")
##st.markdown("Please be advised that all recommendations are based on self-reported mental health scores of listeners. Since these recommendations are based on the correlations between listening preferences and mental health, they are not proven to *cause* changes in mood, but rather are *associated* with changes in mood.")

#Have a tab on the left side of the app that leads to the second page (steps behind the data science). This page will have a drop down menu.

option = st.sidebar.selectbox(
    "Navigation:",
    ["Get Recommendations", "App Development"]
)


if option == "App Development":
    
    st.title("Tunes by Mood: App Development")

    #dropdown menu
    categories = ["Data Overview", "Investigate The Data", "Clean The Data", "Explore The Data", "Improving App Performance"]
    selected_category = st.selectbox("Choose a section of the app development process to explore:", categories)
    
    if selected_category == "Data Overview":
        
        st.title("Data Overview:")
        #st.markdown("Please be advised that all recommendations are based on self-reported mental health scores of listeners. Since these recommendations are based on the correlations between listening preferences and mental health, they are not proven to *cause* changes in mood, but rather are *associated* with changes in mood.")
    
        
        #display the selected category
        #st.write(f"You selected: {selected_category}")
        
        #markdown section
        st.subheader("What does the first [dataset](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results) look like?")
        st.markdown("* Purpose: this dataset will build our recommendation system by providing info on the relationships between listening habits and mental health")
        st.markdown("* Mixture of data types (ex: Primary streaming service: Nominal, Hours per day: Ratio, Anxiety: Ordinal, Composer: Binary)")
        st.markdown("Feature description:")
        st.write("* All observations are self-reported. Mental health stats are on a scale of 1-10")
        st.write("* 16 unique genres are considered")
        st.write("* BMP = Beats per minute of favorite genre")
        st.write("* All feature descriptions are included in the link above")
        
        #load the Data
        mxmh_survey_results = pd.read_csv("mxmh_survey_results.csv")
            
        #display the data
        st.write(mxmh_survey_results.head())  
    
        #markdown section
        st.subheader("What does the second [dataset](https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019) look like?")
        st.markdown("* Purpose: this dataset will provide a libary to pull songs from based on user input. Since it also includes a genre column, the two datasets will be joined on genre.")
        st.markdown("* Mixture of data types (ex: Artist: Nominal, Duration_ms: Ratio, Popularity: Ordinal, Explicit: Binary)")
        st.markdown("Feature description:")
        st.write("* Valence: positivity of the track (0 to 1)")
        st.write("* Danceability: considers tempo, beat strength, and rhythm stability")
        st.write("* Energy: intensity of song (0 to 1)")
        st.write("* All feature descriptions are included in the link above")
    
        
        #load the Data
        songs = pd.read_csv("songs_normalize.csv")
            
        #display the data
        st.write(songs.head())  
    
    if selected_category == "Investigate The Data":
        
        st.title("Investigate The Data:")
    
        #load the Data
        mxmh_survey_results = pd.read_csv("mxmh_survey_results.csv")
        
        #missing vals
        st.subheader("Handle missing values:")
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
    
        #plans for missing vals
        st.markdown("Plans To Handle Missing Data")
        st.write("* Age will not be included in main analysis, which focuses on listening habits and mental health. Age outliers will be handled below, but missing age values will not be since they're not incorporated in downstream analysis.")
        st.write("* Missing BPM vals will be replaced by the median of the genre each missing BMP belongs to. Please choose the Clean The Data tab to see how this is done.")
        st. write("* As for Primary Streaming Service, While Working, Music Effects, Instrumentalist, and Foreign Language, these missing vals will not be handled because they don't impact the main analysis. The purpose of this is to retain the important info of those observations (mental health scores and listening frequencies). Dropping these missing values would result in loss of important observed data.")
    
        #distribution 
        st.header("Distribution of Features")
        
        st.subheader("Age")
        #age 
        fig = px.histogram(mxmh_survey_results, x="Age", title="Age Distribution")
        st.plotly_chart(fig)
        st.markdown("Most participants are between late teens and late 20s.")
        
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
        st.markdown("Most participants stream music with Spotify.")
        
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
    
        st.markdown("Most participants are fans of rock. Few are fans of Gospel, Latin, and Lofi.")
        
        st.subheader("Mental Health Stats")
    
        import streamlit as st
        import pandas as pd
        import plotly.express as px

        
        # Subset to include only mental health metrics
        mh_subset = mxmh_survey_results[["Anxiety", "Depression", "OCD", "Insomnia"]]
        
        # Convert wide format to long format
        long_format_df = mh_subset.melt(var_name="Metric", value_name="Score")
        
        # Bin the scores into intervals of 1 (0–1, 1–2, ..., 9–10)
        long_format_df["Score Bin"] = pd.cut(
            long_format_df["Score"], bins=range(0, 12), labels=range(0, 11), right=False
        )
        
        # Count occurrences of each bin for each metric
        binned_counts = long_format_df.groupby(["Metric", "Score Bin"]).size().reset_index(name="Count")
        
        # Create a grouped bar plot
        fig = px.bar(
            binned_counts,
            x="Score Bin",
            y="Count",
            color="Metric",
            barmode="group",
            title="Frequency Distribution of Mental Health Scores (Binned)",
            labels={"Score Bin": "Score (Binned)", "Count": "Frequency"},
        )
        
        # Customize plot layout
        fig.update_layout(
            plot_bgcolor="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True),
        )
        
        # Show the chart in Streamlit
        st.subheader("Mental Health Stats")
        st.plotly_chart(fig)

        st.markdown("Most participants experience anxiety and depression but not OCD or insomnia as much.")
        
        #frequency
        # st.subheader("Genre Frequency")
    
        # #make a subset so we're only focused on the frequency columns
        # frequency_subset = mxmh_survey_results[['Frequency [Classical]',
        #        'Frequency [Country]', 'Frequency [EDM]', 'Frequency [Folk]',
        #        'Frequency [Gospel]', 'Frequency [Hip hop]', 'Frequency [Jazz]',
        #        'Frequency [K pop]', 'Frequency [Latin]', 'Frequency [Lofi]',
        #        'Frequency [Metal]', 'Frequency [Pop]', 'Frequency [R&B]',
        #        'Frequency [Rap]', 'Frequency [Rock]', 'Frequency [Video game music]']]
        
        
        # #rename the columns to keep only genre names using str.replace()
        # frequency_subset.columns = frequency_subset.columns.str.replace(r'Frequency \[(.*)\]', r'\1', regex=True)
        
        # #convert the dataset from wide to long format
        # ##the melt function reshapes the dataframe so that all genre frequencies are in a single column, with an additional column indicating the genre.
        # long_format_df = frequency_subset.melt(var_name='Genre', value_name='Frequency')
    
        # #order 'Frequency' column chronologically 
        # order = ['Never', 'Rarely', 'Sometimes', 'Very Frequently']
        # long_format_df['Genre'] = pd.Categorical(long_format_df['Genre'], categories=order, ordered=True)
    
        
        # #create the histogram
        # fig = px.histogram(long_format_df, 
        #                    x='Genre',  ######### x = frequency and color = genre will give you 4 sets of 16 bars
        #                    color='Frequency',  ###########
        #                   #this will give me 16 sets of 4 bars instead of 4 overlaid sets of bars
        #                    barmode='group', 
        #                    category_orders={
        #                        'Frequency': ['Never', 'Rarely', 'Sometimes', 'Very Frequently']  # Custom order
        #                    },
        #                    title='Frequency Distribution of Music Genres')
        
        # #remove x-axis gridlines
        # fig.update_layout(xaxis=dict(showgrid=False))
        
        # #show the plot 
        # st.plotly_chart(fig)
        
        
        #Experts
        st.subheader("Experts")
        
        fig, ax = plt.subplots()
        sns.histplot(data=mxmh_survey_results, x="Composer", bins=2, label="Composers", multiple="stack", ax=ax)
        sns.histplot(data=mxmh_survey_results, x="Instrumentalist", bins=2, label="Instrumentalists", ax=ax)
        
        ax.legend()
        
        ax.set_title("Distribution of Composers and Instrumentalists")
        
        st.pyplot(fig)
    
        st.markdown("Most of the participants are not instrumentalists nor composers.")
        
        #Music Effects
        st.subheader("Music Effects")
        fig, ax = plt.subplots()
        sns.histplot(data=mxmh_survey_results, x='Music effects', hue='Music effects', palette=['red', 'blue', 'green'])
        ax.legend()
        ax.set_title("Distribution of Perceived Music Effects")
        st.pyplot(fig)
    
        st.markdown("Most of the participants say music does have a positive effect on their mental health.")
        
        st.subheader("Hours Per Day")
        fig, ax = plt.subplots()
        sns.histplot(data=mxmh_survey_results, x='Hours per day')
        ax.legend()
        ax.set_title("Hours Per Day")
        st.pyplot(fig)

        st.markdown("Most of the participants listen to music between 0 and 5 hours per day.")
        #look at outliers
        st.subheader("Consider Outliers")
        
        #hour outliers
        num_24_hours = sum(mxmh_survey_results['Hours per day'] == 24)
        st.write(f"Number of participants reporting 24 hours per day: {num_24_hours}")
        
        #age outliers:
        age_outliers = sum((mxmh_survey_results['Age'] > 70) | (mxmh_survey_results['Age'] < 18))
        st.write(f"Number of participants younger than 18 or older than 70: {age_outliers}")
    
    
        st.subheader("Investigating Second Dataset")
    
        #load the Data
        songs = pd.read_csv("songs_normalize.csv")
        
        st.markdown("No Missing Vals")
    
        #make a heatmap of the missing data
        
        #import numpy and nickname it np
        import numpy as np 
        
        #import matplotlib as plt
        import matplotlib.pyplot as plt
        
        # create a boolean mask: True for NaN, False for finite values
        nan_mask = songs.isna()
        
        # convert boolean mask to integer (False becomes 0, True becomes 1)
        nan_array = nan_mask.astype(int).to_numpy()
        
        # size the plot 12 x 6 
        plt.figure(figsize=(12, 6))
        
        # imshow with interpolation set to 'nearest' and aspect to 'auto'
        # interpoltation is finding the best fit of data 
        im = plt.imshow(nan_array.T, interpolation='nearest', aspect='auto', cmap='viridis')
        
        # label the x axis Planet Index
        plt.xlabel('Songs Index')
        # label the y axis Features
        plt.ylabel('Features')
        # title the whole plot Visualizing Missing Values in a Dataset 
        plt.title('Visualizing Missing Values in Songs Dataset')
        
        # y-axis tick labels to feature names
        # make the y-axis go from 0 to 4 and label them the names of the subset columns
        plt.yticks(range(len(songs.columns)), songs.columns)
        
        # x-axis ticks
        #
        num_songs = nan_array.shape[0]
        plt.xticks(np.linspace(0, num_songs-1, min(10, num_songs)).astype(int))
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        st.pyplot(plt)
    
        #investigating columns 
    
        st.markdown("Distribution of features")
        st.write("The features are imbalanced. Most songs have a mid-high valence, high energy, mid-high danceability, and duration of 200 seconds.")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))  # Set figure size
        sns.histplot(data=songs, x='valence', ax=ax)
        
        # Set title and labels
        ax.set_title('Distribution of Valence')
        ax.set_xlabel('Valence')
        ax.set_ylabel('Count')
        
        # Display the plot in Streamlit
        st.pyplot(fig)
    
        
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))  # Set figure size
        sns.histplot(data=songs, x='energy', ax=ax)
        
        # Set title and labels
        ax.set_title('Distribution of Energy')
        ax.set_xlabel('Energy')
        ax.set_ylabel('Count')
        
        # Display the plot in Streamlit
        st.pyplot(fig)
        
    
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))  # Set figure size
        sns.histplot(data=songs, x='danceability', ax=ax)
        
        # Set title and labels
        ax.set_title('Distribution of Danceability')
        ax.set_xlabel('Danceability')
        ax.set_ylabel('Count')
        
        # Display the plot in Streamlit
        st.pyplot(fig)
    
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))  # Set figure size
        sns.histplot(data=songs, x='duration_ms', ax=ax)
        
        # Set title and labels
        ax.set_title('Distribution of Duration')
        ax.set_xlabel('Duration')
        ax.set_ylabel('Count')
        
        # Display the plot in Streamlit
        st.pyplot(fig)
    
    
    if selected_category == "Clean The Data":
    
        st.title("Clean The Data:")
    
        #load the Data
        mxmh_survey_results = pd.read_csv("mxmh_survey_results.csv")
        
        #handle missing vals 
        st.subheader("Handle BPM Missing Values")
        st.markdown("Group all BPM values by genre, then replace missing vals with median BPM of respective genre.")
    
        #get pop median so we can test our replacement worked
        pop_median = mxmh_survey_results[mxmh_survey_results["Fav genre"] == "Pop"]["BPM"].median()
        st.write(f"Examine songs in the Pop genre to ensure missing vals are replaced with median Pop BPM. The median BPM of Pop: {pop_median}")
    
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
        st.subheader("0 Missing BPM Vals Remaining")
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
        st.markdown("Deleted all instances of self-reported Hours Per Day above 16 to ensure realistic data. This deleted 6 rows of observations. The shape of the dataset is now:")
        cleaned_data.shape
    
        #take away age outliers 
        cleaned_data = cleaned_data[(cleaned_data["Age"] > 18) & (cleaned_data["Age"] < 64)]
        st.markdown("Deleted all instances of Age < 18 and Age > 64 (3 SDs from the 75% percentile). This deleted 50 rows of observations. The shape of the dataset is now:")
        cleaned_data.shape

        #take away MH outliers (below 0 and above 10)
        #We actually don't need to do this. I misinterpreted a plot and thought there were outliers for these features
        #st.write("All mental health surveys were taken on a scale of 0-10, but negative values are included in these columns. All values in those features below 0 and above 10 will be removed now. This deleted 50 rows of observations. The shape of the dataset is now:")
        cleaned_data = cleaned_data[
        (cleaned_data["Anxiety"] >= 0) & (cleaned_data["Anxiety"] <= 10) &
        (cleaned_data["Depression"] >= 0) & (cleaned_data["Depression"] <= 10) &
        (cleaned_data["OCD"] >= 0) & (cleaned_data["OCD"] <= 10) &
        (cleaned_data["Insomnia"] >= 0) & (cleaned_data["Insomnia"] <= 10)
        ]
        #cleaned_data.shape
        
        #recode frequency genre
        st.subheader("Recode Categorical Data")
        st.markdown("Genre Frequencies | 'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Very frequently': 4")
    
        frequency_mapping = {
        "Never": 1,
        "Rarely": 2,
        "Sometimes": 3,
        "Very frequently": 4 }
    
        # Replace the values in the "Frequency [Genre]" columns
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
    
        #make a subset so users can see the changes
        frequency_columns = ["Frequency [Latin]", "Frequency [Rock]", "Frequency [Video game music]", "Frequency [Jazz]",
        "Frequency [R&B]", "Frequency [K pop]", "Frequency [Country]", "Frequency [EDM]", "Frequency [Hip hop]",
        "Frequency [Pop]", "Frequency [Rap]", "Frequency [Classical]", "Frequency [Metal]", "Frequency [Folk]",
        "Frequency [Lofi]", "Frequency [Gospel]"]
    
        frequency_subset = cleaned_data[frequency_columns]
        
        #see the changes
        st.markdown("See these changes in the following subset:")
        st.write(frequency_subset.head())  
    
        st.subheader("Handle Imbalance")
        st.markdown("Fav Genre:")
        st.write("* When deleting Age outliers above, one Fav Genre was removed (Latin). (0 participants list Latin as their favorite genre in the remaining observations.) Now, the remaining participants represent 15 favorite genres. The Fav Genre feature was made more balanced by reducing the three genres with outlier frequencies (rock, metal, pop) to instead have the overall mean frequency of Fav Genre (21). Please see these changes below.")
        st.write("Imbalanced distribution before reducing outlier frequencies to the median frequency:")
        
        #count the occurrences of each genre
        genre_counts = cleaned_data["Fav genre"].value_counts()
        
        #create the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(genre_counts.index, genre_counts.values, color='skyblue', edgecolor='black')
        
        #set the title and labels
        plt.title('Distribution of Fav Genre')
        plt.xlabel('Fav Genre')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        #display the plot 
        st.pyplot(plt)
    
        #get the median frequency
        values = cleaned_data["Fav genre"].value_counts()
        #values.median()
    
        #make the changes to rock
        num = 21
        length = len(cleaned_data[cleaned_data["Fav genre"] == "Rock"])
        drop_these_many = length - num
        random_idx = np.random.choice(cleaned_data[cleaned_data["Fav genre"] == "Rock"].index, drop_these_many, replace=False)
        #drop the selected indices from the DataFrame
        cleaned_data = cleaned_data.drop(random_idx)
        
        #make the changes to metal
        num = 21
        length = len(cleaned_data[cleaned_data["Fav genre"] == "Metal"])
        drop_these_many = length - num
        random_idx = np.random.choice(cleaned_data[cleaned_data["Fav genre"] == "Metal"].index, drop_these_many, replace=False)
        #drop the selected indices from the DataFrame
        cleaned_data = cleaned_data.drop(random_idx)
    
        #make the changes to pop
        num = 21
        length = len(cleaned_data[cleaned_data["Fav genre"] == "Pop"])
        drop_these_many = length - num
        random_idx = np.random.choice(cleaned_data[cleaned_data["Fav genre"] == "Pop"].index, drop_these_many, replace=False)
        #drop the selected indices from the DataFrame
        cleaned_data = cleaned_data.drop(random_idx)
    
        st.write("Less imbalanced distribution after reducing outlier frequencies to the median frequency. Keeping the same y-axis range so the difference can be compared:")
    
        #count the occurrences of each genre
        genre_counts = cleaned_data["Fav genre"].value_counts()
        
        #create the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(genre_counts.index, genre_counts.values, color='skyblue', edgecolor='black')
        
        #set the title and labels
        plt.title('Distribution of Fav Genre')
        plt.xlabel('Fav Genre')
        plt.ylabel('Count')
        plt.ylim(0, 140)
        plt.xticks(rotation=45)
        
        #display the plot 
        st.pyplot(plt)
    
        #######now anxiety balance
        st.subheader("Handle imbalance of Anxiety")
        st.write("Anxiety was balanced by undersampling, considering two classes: values below 5 and above 5")
        st.write("Before:")
    
        #original before plot
        plt.figure(figsize=(10, 6))
        plt.hist(cleaned_data["Anxiety"], bins=11, edgecolor='black')
        
        #set the title of the plot
        plt.title('Distribution of Original Anxiety')
        
        #set the x-axis title
        plt.xlabel('Anxiety Score')
        st.pyplot(plt)
    
        #plot of binary before
        cleaned_data["Anxiety_category"] = np.where(cleaned_data["Anxiety"] >= 5, 1, 0)
        plt.figure(figsize=(10, 6))
        plt.hist(cleaned_data["Anxiety_category"], bins=11, edgecolor='black')
        
        #set the title of the plot
        plt.title('Distribution of Original Anxiety as a Binary')
        
        #set the x-axis title
        plt.xlabel('Anxiety Below 5 (0) and Above 5 (1)')
        plt.xticks([0, 1])
        st.pyplot(plt)
    
        ##############balance anxiety 
        #reset index
        cleaned_data.reset_index(drop=True, inplace=True)
    
        X = cleaned_data.drop(["Anxiety", "Anxiety_category"], axis=1)  
        y = cleaned_data["Anxiety_category"] 
        
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        
        print(f"Before Undersampling: \n{y.value_counts()}")
        print(f"After Undersampling: \n{y_resampled.value_counts()}")
    
        resampled_indices = rus.sample_indices_
    
        anxiety_resampled = cleaned_data.loc[resampled_indices, "Anxiety"]
        
        cleaned_data = X_resampled.copy()  
        cleaned_data["Anxiety"] = anxiety_resampled.values  
    
        #reset index
        cleaned_data.reset_index(drop=True, inplace=True)
    
    
        st.markdown("Binary Anxiety after handling imbalance:")
    
        #remake this column so we can plot it
        cleaned_data["Anxiety_category"] = np.where(cleaned_data["Anxiety"] >= 5, 1, 0)
    
        
        #plot of binary after
    
        plt.figure(figsize=(10, 6))
        plt.hist(cleaned_data["Anxiety_category"], bins=11, edgecolor='black')
        
        #set the title of the plot
        plt.title('Distribution of Balanced Anxiety as a Binary')
        
        #set the x-axis title
        plt.xlabel('Anxiety Below 5 (0) and Above 5 (1)')
        plt.xticks([0, 1])
        st.pyplot(plt)
    
        #drop that column again
        cleaned_data = cleaned_data.drop(["Anxiety_category"], axis=1) 
    
        #reset index just incase
        cleaned_data.reset_index(drop=True, inplace=True)
    
        ######now balance depression
        st.subheader("Handle imbalance of Depression")
        st.write("Depression was also balanced by undersampling, considering two classes: values below 5 and above 5")
        st.write("Before:")
        
    
    
        #original before plot
        plt.figure(figsize=(10, 6))
        plt.hist(cleaned_data["Depression"], bins=11, edgecolor='black')
        
        #set the title of the plot
        plt.title('Distribution of Original Depression')
        
        #set the x-axis title
        plt.xlabel('Depression Score')
        st.pyplot(plt)
    
        #plot of binary before
        cleaned_data["Depression_category"] = np.where(cleaned_data["Depression"] >= 5, 1, 0)
        plt.figure(figsize=(10, 6))
        plt.hist(cleaned_data["Depression_category"], bins=11, edgecolor='black')
        
        #set the title of the plot
        plt.title('Distribution of Original Depression as a Binary')
        
        #set the x-axis title
        plt.xlabel('Depression Below 5 (0) and Above 5 (1)')
        plt.xticks([0, 1])
        st.pyplot(plt)
    
        ##############balance depression
        #reset index
        cleaned_data.reset_index(drop=True, inplace=True)
    
        X = cleaned_data.drop(["Depression", "Depression_category"], axis=1)  
        y = cleaned_data["Depression_category"] 
        
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        
        print(f"Before Undersampling: \n{y.value_counts()}")
        print(f"After Undersampling: \n{y_resampled.value_counts()}")
    
        resampled_indices = rus.sample_indices_
    
        depression_resampled = cleaned_data.loc[resampled_indices, "Depression"]
        
        cleaned_data = X_resampled.copy()  
        cleaned_data["Depression"] = depression_resampled.values  
    
        #reset index
        cleaned_data.reset_index(drop=True, inplace=True)
    
    
        st.markdown("Binary Depression after handling imbalance:")
    
        #remake this column so we can plot it
        cleaned_data["Depression_category"] = np.where(cleaned_data["Depression"] >= 5, 1, 0)
    
        
        #plot of binary after
    
        plt.figure(figsize=(10, 6))
        plt.hist(cleaned_data["Depression_category"], bins=11, edgecolor='black')
        
        #set the title of the plot
        plt.title('Distribution of Balanced Depression as a Binary')
        
        #set the x-axis title
        plt.xlabel('Depression Below 5 (0) and Above 5 (1)')
        plt.xticks([0, 1])
        st.pyplot(plt)
    
        #drop that column again
        cleaned_data = cleaned_data.drop(["Depression_category"], axis=1) 
    
        #reset index just in case
        cleaned_data.reset_index(drop=True, inplace=True)
    
    
    
        ###########see that I should stop balancing now
    
        st.markdown("OCD and insomnia were purposefully left imbalanced. Balancing depression made anxiety slightly imbalanced again. (Please see the plot below.) This happened because undersampling deletes observations, so each time undersampling was done, the class balance is affected. So, balancing by undersampling was only done for the two most prominent mental health metrics among participants, anxiety and depression, to preserve balancing efforts.")
    
        #create the Anxiety_category column 
        cleaned_data["Anxiety_category"] = np.where(cleaned_data["Anxiety"] >= 5, 1, 0)
        
        #create the plot
        fig, ax = plt.subplots(figsize=(10, 6))  
        sns.histplot(data=cleaned_data, x='Anxiety_category', ax=ax)
        
        #set title and labels
        ax.set_title('Distribution of Anxiety After Balancing Depression')
        ax.set_xlabel('Anxiety Category')
        ax.set_ylabel('Count')
        
        #display the plot 
        st.pyplot(fig)
    
    
        st.subheader("Clean the second dataset")
        songs = pd.read_csv("songs_normalize.csv")
    
        st.write("Filter out all explicit songs so the app is appropriate for all users.")
        num_before = (songs["explicit"] == True).sum()
        st.write(f"Number of explicit songs before filtering: {num_before}")
        songs = songs[songs["explicit"] == False]
        num_after = (songs["explicit"] == True).sum()
        st.write(f"Number of explicit songs after filtering: {num_after}")
        #st.write(songs["explicit"].head())  
    
        st.markdown("Some songs are categorized as multiple genres. Let's split that up so each song is listed once per genre that it classifies as. This will create duplicates. For example, I want a pop-rock song to be recommened for pop and rock recommedations.")
        songs["genre"] = songs["genre"].str.split(",")
    
        #explode the dataset so each genre gets its own row
        ######explode() expands the list of genres so each genre has its own row, duplicating other information about the song.
        #####reset_index(drop=True)  resets the index to keep things neat after exploding.
        songs_expanded = songs.explode("genre").reset_index(drop=True)
        
        #what does the dataset look like now
        st.write(songs_expanded["genre"].head())  
    
        #make sure genres are consistent
        #songs_expanded["genre"]==[" Folk/Acoustic"].replace("Folk/Acoustic")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" Folk/Acoustic", "Folk/Acoustic")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" Dance/Electronic", "Dance/Electronic")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" pop", "pop")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" hip hop", "hip hop")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" country", "country")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" metal", "metal")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" R&B", "R&B")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" rock", "rock")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" easy listening", "easy listening")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" latin", "latin")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" classical", "classical")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" blues", "blues")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" jazz", "Jazz")
    
        #changing capitalization and wording
        songs_expanded["genre"] = songs_expanded["genre"].replace("pop", "Pop")
        songs_expanded["genre"] = songs_expanded["genre"].replace("rock", "Rock")
        songs_expanded["genre"] = songs_expanded["genre"].replace("country", "Country")
        songs_expanded["genre"] = songs_expanded["genre"].replace("metal", "Metal")
        songs_expanded["genre"] = songs_expanded["genre"].replace("hip hop", "Hip hop")
        songs_expanded["genre"] = songs_expanded["genre"].replace("Dance/Electronic", "EDM")
        songs_expanded["genre"] = songs_expanded["genre"].replace("Folk/Acoustic", "Folk")
        songs_expanded["genre"] = songs_expanded["genre"].replace("latin", "Latin")
        songs_expanded["genre"] = songs_expanded["genre"].replace("jazz", "Jazz")
        songs_expanded["genre"] = songs_expanded["genre"].replace("classical", "Classical")
    
        st.markdown("I also capitalized the genre names to match the names in the first dataset so we join the two successfully later on.")
        st.write(songs_expanded["genre"].head())  
                
    
        st.subheader("Handle imbalance")
        st.markdown("Even distribution after balancing valence based on two classes: above 0.5 (1) and below 0.5 (0):")
        songs_expanded.reset_index(drop=True, inplace=True)
        songs_expanded["valence_category"] = np.where(songs_expanded["valence"] >= 0.5, 1, 0)
        #separate features (X) and target (y)
        #drop the continuous feature and the categorical version we just made
        X = songs_expanded.drop(["valence", "valence_category"], axis=1)  # Keep only non-target features
        #look at the categorical version as the target 
        y = songs_expanded["valence_category"]  # Target variable
        
        #apply RandomUnderSampler
        #initialize it
        rus = RandomUnderSampler(random_state=42)
        #apply it to X and y and store the changed versions
        X_resampled, y_resampled = rus.fit_resample(X, y)
        
        #print the differences so we can see that the package did its job
        print(f"Before Undersampling: \n{y.value_counts()}")
        print(f"After Undersampling: \n{y_resampled.value_counts()}")
    
        #get the indices of the resampled data
        resampled_indices = rus.sample_indices_
        
        #use the indices to retrieve the original continuous valence values
        valence_resampled = songs_expanded.loc[resampled_indices, "valence"]
        
        #create the final resampled dataset with original continuous valence values
        songs_balanced = X_resampled.copy()  #start with resampled features
        songs_balanced["valence"] = valence_resampled.values  #add back continuous valence
    
        songs_balanced["valence_category"] = np.where(songs_balanced["valence"] >= 0.5, 1, 0)
    
        # Create the plot
        fig, ax = plt.subplots()  # Initialize a Matplotlib figure and axis
        sns.histplot(data=songs_balanced, x='valence_category', ax=ax)
        
        ax.set_title('Distribution of Binary Valence')
        ax.set_xlabel('Valence Category')
        ax.set_ylabel('Count')
        
        # display the plot in Streamlit
        st.pyplot(fig)
    
    
    if selected_category == "Explore The Data":
        
        st.title("Explore The Data:")
    
        ##########repeating the filtering so I can use the filtered_dataset here
        
        #load the Data
        mxmh_survey_results = pd.read_csv("mxmh_survey_results.csv")
    
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
    
      
        cleaned_data = mxmh_survey_results.copy()
        #I will say the max they could realistically listen to is 16 hrs
        cleaned_data = cleaned_data[(cleaned_data["Hours per day"] < 16)]
        #deleted 6 rows
    
        #take away age outliers 
        cleaned_data = cleaned_data[(cleaned_data["Age"] > 18) & (cleaned_data["Age"] < 64)]

        #take away MH outliers             
        cleaned_data = cleaned_data[
        (cleaned_data["Anxiety"] >= 0) & (cleaned_data["Anxiety"] <= 10) &
        (cleaned_data["Depression"] >= 0) & (cleaned_data["Depression"] <= 10) &
        (cleaned_data["OCD"] >= 0) & (cleaned_data["OCD"] <= 10) &
        (cleaned_data["Insomnia"] >= 0) & (cleaned_data["Insomnia"] <= 10)
        ]
        
        #recode frequency genre
    
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
    
    
        cleaned_data = cleaned_data.copy()
        #I will say the max they could realistically listen to is 16 hrs
        cleaned_data = cleaned_data[(cleaned_data["Hours per day"] < 16)]
        #deleted 6 rows
    
        #take away age outliers 
        cleaned_data = cleaned_data[(cleaned_data["Age"] > 18) & (cleaned_data["Age"] < 64)]
        
        #get the median frequency
        values = cleaned_data["Fav genre"].value_counts()
        #values.median()
    
        #make the changes to rock
        num = 21
        length = len(cleaned_data[cleaned_data["Fav genre"] == "Rock"])
        drop_these_many = length - num
        random_idx = np.random.choice(cleaned_data[cleaned_data["Fav genre"] == "Rock"].index, drop_these_many, replace=False)
        #drop the selected indices from the DataFrame
        cleaned_data = cleaned_data.drop(random_idx)
        
        #make the changes to metal
        num = 21
        length = len(cleaned_data[cleaned_data["Fav genre"] == "Metal"])
        drop_these_many = length - num
        random_idx = np.random.choice(cleaned_data[cleaned_data["Fav genre"] == "Metal"].index, drop_these_many, replace=False)
        #drop the selected indices from the DataFrame
        cleaned_data = cleaned_data.drop(random_idx)
    
        #make the changes to pop
        num = 21
        length = len(cleaned_data[cleaned_data["Fav genre"] == "Pop"])
        drop_these_many = length - num
        random_idx = np.random.choice(cleaned_data[cleaned_data["Fav genre"] == "Pop"].index, drop_these_many, replace=False)
        #drop the selected indices from the DataFrame
        cleaned_data = cleaned_data.drop(random_idx)
    
    
        ##############balance anxiety 
        #reset index
        cleaned_data.reset_index(drop=True, inplace=True)
        cleaned_data["Anxiety_category"] = np.where(cleaned_data["Anxiety"] >= 5, 1, 0)
        X = cleaned_data.drop(["Anxiety", "Anxiety_category"], axis=1)  
        y = cleaned_data["Anxiety_category"] 
        
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        
        print(f"Before Undersampling: \n{y.value_counts()}")
        print(f"After Undersampling: \n{y_resampled.value_counts()}")
    
        resampled_indices = rus.sample_indices_
    
        anxiety_resampled = cleaned_data.loc[resampled_indices, "Anxiety"]
        
        cleaned_data = X_resampled.copy()  
        cleaned_data["Anxiety"] = anxiety_resampled.values  
    
        #reset index
        cleaned_data.reset_index(drop=True, inplace=True)
    
    
        ######now balance depression
        cleaned_data["Depression_category"] = np.where(cleaned_data["Depression"] >= 5, 1, 0)
        X = cleaned_data.drop(["Depression", "Depression_category"], axis=1)  
        y = cleaned_data["Depression_category"] 
        
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        
        print(f"Before Undersampling: \n{y.value_counts()}")
        print(f"After Undersampling: \n{y_resampled.value_counts()}")
    
        resampled_indices = rus.sample_indices_
    
        depression_resampled = cleaned_data.loc[resampled_indices, "Depression"]
        
        cleaned_data = X_resampled.copy()  
        cleaned_data["Depression"] = depression_resampled.values  
    
        #reset index
        cleaned_data.reset_index(drop=True, inplace=True)
    
    
    
        songs = pd.read_csv("songs_normalize.csv")
    
        songs = songs[songs["explicit"] == False]
        #st.write(songs.head())  
    
        #st.markdown("Some songs are categorized as multiple genres. Let's split that up so each song is listed once per genre that it classifies as. This will create duplicates. For example, I want a pop-rock song to be recommened for pop and rock recommedations.")
        songs["genre"] = songs["genre"].str.split(",")
    
        #explode the dataset so each genre gets its own row
        ######explode() expands the list of genres so each genre has its own row, duplicating other information about the song.
        #####reset_index(drop=True)  resets the index to keep things neat after exploding.
        songs_expanded = songs.explode("genre").reset_index(drop=True)
        
        
    
        #make sure genres are consistent
        #songs_expanded["genre"]==[" Folk/Acoustic"].replace("Folk/Acoustic")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" Folk/Acoustic", "Folk/Acoustic")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" Dance/Electronic", "Dance/Electronic")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" pop", "pop")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" hip hop", "hip hop")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" country", "country")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" metal", "metal")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" R&B", "R&B")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" rock", "rock")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" easy listening", "easy listening")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" latin", "latin")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" classical", "classical")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" blues", "blues")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" jazz", "Jazz")
    
        #changing capitalization and wording
        songs_expanded["genre"] = songs_expanded["genre"].replace("pop", "Pop")
        songs_expanded["genre"] = songs_expanded["genre"].replace("rock", "Rock")
        songs_expanded["genre"] = songs_expanded["genre"].replace("country", "Country")
        songs_expanded["genre"] = songs_expanded["genre"].replace("metal", "Metal")
        songs_expanded["genre"] = songs_expanded["genre"].replace("hip hop", "Hip hop")
        songs_expanded["genre"] = songs_expanded["genre"].replace("Dance/Electronic", "EDM")
        songs_expanded["genre"] = songs_expanded["genre"].replace("Folk/Acoustic", "Folk")
        songs_expanded["genre"] = songs_expanded["genre"].replace("latin", "Latin")
        songs_expanded["genre"] = songs_expanded["genre"].replace("jazz", "Jazz")
        songs_expanded["genre"] = songs_expanded["genre"].replace("classical", "Classical")
    
      
    
        
        songs_expanded.reset_index(drop=True, inplace=True)
        songs_expanded["valence_category"] = np.where(songs_expanded["valence"] >= 0.5, 1, 0)
        #separate features (X) and target (y)
        #drop the continuous feature and the categorical version we just made
        X = songs_expanded.drop(["valence", "valence_category"], axis=1)  # Keep only non-target features
        #look at the categorical version as the target 
        y = songs_expanded["valence_category"]  # Target variable
        
        #apply RandomUnderSampler
        #initialize it
        rus = RandomUnderSampler(random_state=42)
        #apply it to X and y and store the changed versions
        X_resampled, y_resampled = rus.fit_resample(X, y)
        
        #print the differences so we can see that the package did its job
        print(f"Before Undersampling: \n{y.value_counts()}")
        print(f"After Undersampling: \n{y_resampled.value_counts()}")
    
        #get the indices of the resampled data
        resampled_indices = rus.sample_indices_
        
        #use the indices to retrieve the original continuous valence values
        valence_resampled = songs_expanded.loc[resampled_indices, "valence"]
        
        #create the final resampled dataset with original continuous valence values
        songs_balanced = X_resampled.copy()  #start with resampled features
        songs_balanced["valence"] = valence_resampled.values  #add back continuous valence
    
        songs_balanced["valence_category"] = np.where(songs_balanced["valence"] >= 0.5, 1, 0)
    
    
    
    
    
    
    
    
        ########################### done repeating the filtering
        
        st.subheader("Investigate correlations between listening frequency and mental health stats:")
    
        selected_features = ['Frequency [Classical]', "Frequency [Country]", "Frequency [EDM]", "Frequency [Folk]", 
                         "Frequency [Gospel]", "Frequency [Hip hop]", "Frequency [Jazz]", "Frequency [K pop]", "Frequency [Lofi]",
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

        st.write("All correlations between mental health metrics and listening frequencies are weak.")
        st.write("Anxiety is most highly correlated with Rock (r = .24).")
        st.write("Depression is most highly correlated with Rock (r = .26).")
        st.write("OCD is most highly correlated with Folk (r = .12).")
        st.write("Insomnia is most highly correlated with Metal (r = .16).")
                 
        #hours and mh
        st.subheader("Investigate relationship between hours spent listening per day and reported mental health scores:")
        selected_features = ['Hours per day', "Anxiety", "Depression", "Insomnia", "OCD"] # Focus on these variables
    
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

        st.write("The two are not strongly correlated. The mental health metric most strongly related to Hours Per Day is Insomnia, with a weak pearson coefficient (r = .22).")
        
    
        st.subheader("Investigate how mental health varies across age:")
    
        bins = [18, 25, 31, 36, 41, 46, 51, 58, 64]  
        labels = ['18-24', '25-30', '31-35', '36-40', '41-45', '46-50', '51-57', '58-64']  # Labels for the bins
    
        # Create the binned column
        cleaned_data['age_binned'] = pd.cut(cleaned_data['Age'], bins=bins, labels=labels, right=False)

        # Convert 'age_binned' to an ordered categorical type
        cleaned_data['age_binned'] = pd.Categorical(
        cleaned_data['age_binned'], 
        categories=labels,  # Use the same order as your labels
        ordered=True
        )
    
        #now plot it 
        # fig_violin = px.violin(cleaned_data, x='age_binned', y='Anxiety', box=True, points='all',
        #                        labels={'Age':'Age', 'Anxiety':'Anxiety'},
        #                        title="Interactive Violin Plot of Age vs Anxiety")

        fig_violin = px.violin(
        cleaned_data, 
        x='age_binned', 
        y='Anxiety', 
        box=True, 
        points='all',
        labels={'age_binned': 'Age Group', 'Anxiety': 'Anxiety Level'},
        title="Interactive Violin Plot of Age vs Anxiety",
        category_orders={"age_binned": labels}  # Explicitly set order
        )
    
        st.plotly_chart(fig_violin)

        st.write("Least variability among 41-45 year olds. This group may include less participants.")
        st.write("Most variability among 46-50 year olds.")
        st.write("Lowest three median Anxiety levels among oldest three age groups.")
    
        #fav genre and MH
        st.subheader("Investigate how Fav Genre is associated with mental health scores:")
        fig_violin = px.violin(cleaned_data, x='Fav genre', y='Anxiety', box=True, points='all',
                               labels={'Fav genre':'Favorite Genre', 'Anxiety':'Anxiety'},
                               title="Interactive Violin Plot of Fav Genre vs Anxiety")
        
        st.plotly_chart(fig_violin)

        st.write("Lofi listeners tend to have high Anxiety scores.")
        
    
        fig_violin = px.violin(cleaned_data, x='Fav genre', y='Depression', box=True, points='all',
                               labels={'Fav genre':'Favorite Genre', 'Depression':'Depression'},
                               title="Interactive Violin Plot of Fav Genre vs Depression ")
    
        st.plotly_chart(fig_violin)

        st.write("Lofi listeners also tend to have high Depression scores.")
        st.write("Rap listeners tend to have lower Depression scores.")
    
        fig_violin = px.violin(cleaned_data, x='Fav genre', y='OCD', box=True, points='all',
                               labels={'Fav genre':'Favorite Genre', 'OCD':'OCD'},
                               title="Interactive Violin Plot of Fav Genre vs OCD")
        st.plotly_chart(fig_violin)

        st.write("Very low data on OCD among Gospel listeners.")
        st.write("High variability among Lofi listeners. OCD must have a neutral association with Lofi listening.")
        st.write("Classical and Jazz listeners tend to have lower OCD scores than other listeners.")
    
        fig_violin = px.violin(cleaned_data, x='Fav genre', y='Insomnia', box=True, points='all',
                               labels={'Fav genre':'Favorite Genre', 'Insomnia':'Insomnia'},
                               title="Interactive Violin Plot of Fav Genre vs Insomnia")
        st.plotly_chart(fig_violin)

        #st.write("High variability among Gospel listeners. Insomnia must have a neutral association with Gospel listening.")
        st.write("Country and Pop listeners tend to have lower Insomnia scores.")
        st.write("Lofi listeners tend to high higher Insomnia scores.")
        
    
        
        #look at mental health stat by genre
        st.subheader("Anxiety and Depression Scores by Genre Listening Frequency")
        #sns.boxplot(data=cleaned_data, x="Frequency [Latin]", y = "Anxiety")
        #plt.title('Anxiety Scores of Latin Listeners')
    
        #function to create and display a box plot for a specific genre
        def plot_boxplot(genre, score):
            plt.figure(figsize=(10, 6))  
            sns.boxplot(data=cleaned_data, x=genre, y=score)
            #make it so user can choose MH stat and genre
            plt.title(f'{score} Scores of {genre} Listeners')
            plt.xlabel(genre)  
            plt.ylabel(score)  
            st.pyplot(plt)  
            plt.clf()  
    
    
        #dropdown menu for selecting a mental health score
        score_options = ["Anxiety", "Depression"]
        selected_score = st.selectbox("Choose a mental health category to consider:", score_options)
        
        #dropdown menu for selecting a genre
        genre_options = ["Frequency [Rock]", "Frequency [Classical]", 
                         "Frequency [Pop]", "Frequency [Jazz]", "Frequency [Hip hop]", 
                         "Frequency [EDM]", "Frequency [Video game music]", 
                         "Frequency [Country]", "Frequency [R&B]", 
                         "Frequency [K pop]", "Frequency [Folk]", 
                         "Frequency [Lofi]", "Frequency [Rap]", 
                         "Frequency [Metal]", "Frequency [Gospel]", "Frequency [Latin]"]
        
        selected_genre = st.selectbox("Choose a genre to consider:", genre_options)
        
    
        #call the plot function with the selected genre and score
        plot_boxplot(selected_genre, selected_score)
    
        #feature engineering
        st.subheader("Feature Engineering:")
        st.markdown("Create a dataset with final mental health scores based on genre listening frequency.")
    
        #group average MH scores by highest frequency genre
        st.markdown("Group average mental health scores by all Very Frequent genre responses. That is, for each genre listening frequency, consider only the Very Frequent responses. Then find the average self-reported score for each mental health metric to get an average mental health stat for each metric among those listeners who listen to each genre Very Frequently. Other genre frequencies are not considered to ensure effect power of genre listening on mental health.")
    
        #making latin subsets based on frequency
        cleaned_data_latin1 = cleaned_data[cleaned_data["Frequency [Latin]"] == 1]
        cleaned_data_latin2 = cleaned_data[cleaned_data["Frequency [Latin]"] == 2]
        cleaned_data_latin3 = cleaned_data[cleaned_data["Frequency [Latin]"] == 3]
        cleaned_data_latin4 = cleaned_data[cleaned_data["Frequency [Latin]"] == 4]
        
        
        #now get the average MH scores for each frequency
        ave_anxiety_latin1 = cleaned_data_latin1["Anxiety"].mean()
        ave_dep_latin1 = cleaned_data_latin1["Depression"].mean()
        ave_insom_latin1 = cleaned_data_latin1["Insomnia"].mean()
        ave_ocd_latin1 = cleaned_data_latin1["OCD"].mean()
        
        ave_anxiety_latin2 = cleaned_data_latin2["Anxiety"].mean()
        ave_dep_latin2 = cleaned_data_latin2["Depression"].mean()
        ave_insom_latin2 = cleaned_data_latin2["Insomnia"].mean()
        ave_ocd_latin2 = cleaned_data_latin2["OCD"].mean()
        
        ave_anxiety_latin3 = cleaned_data_latin3["Anxiety"].mean()
        ave_dep_latin3 = cleaned_data_latin3["Depression"].mean()
        ave_insom_latin3 = cleaned_data_latin3["Insomnia"].mean()
        ave_ocd_latin3 = cleaned_data_latin3["OCD"].mean()
        
        ave_anxiety_latin4 = cleaned_data_latin4["Anxiety"].mean()
        ave_dep_latin4 = cleaned_data_latin4["Depression"].mean()
        ave_insom_latin4 = cleaned_data_latin4["Insomnia"].mean()
        ave_ocd_latin4 = cleaned_data_latin4["OCD"].mean()
        
        
        
        #making rock subsets based on frequency
        cleaned_data_rock1 = cleaned_data[cleaned_data["Frequency [Rock]"] == 1]
        cleaned_data_rock2 = cleaned_data[cleaned_data["Frequency [Rock]"] == 2]
        cleaned_data_rock3 = cleaned_data[cleaned_data["Frequency [Rock]"] == 3]
        cleaned_data_rock4 = cleaned_data[cleaned_data["Frequency [Rock]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_rock1 = cleaned_data_rock1["Anxiety"].mean()
        ave_dep_rock1 = cleaned_data_rock1["Depression"].mean()
        ave_insom_rock1 = cleaned_data_rock1["Insomnia"].mean()
        ave_ocd_rock1 = cleaned_data_rock1["OCD"].mean()
        
        ave_anxiety_rock2 = cleaned_data_rock2["Anxiety"].mean()
        ave_dep_rock2 = cleaned_data_rock2["Depression"].mean()
        ave_insom_rock2 = cleaned_data_rock2["Insomnia"].mean()
        ave_ocd_rock2 = cleaned_data_rock2["OCD"].mean()
        
        ave_anxiety_rock3 = cleaned_data_rock3["Anxiety"].mean()
        ave_dep_rock3 = cleaned_data_rock3["Depression"].mean()
        ave_insom_rock3 = cleaned_data_rock3["Insomnia"].mean()
        ave_ocd_rock3 = cleaned_data_rock3["OCD"].mean()
        
        ave_anxiety_rock4 = cleaned_data_rock4["Anxiety"].mean()
        ave_dep_rock4 = cleaned_data_rock4["Depression"].mean()
        ave_insom_rock4 = cleaned_data_rock4["Insomnia"].mean()
        ave_ocd_rock4 = cleaned_data_rock4["OCD"].mean()
        
        
        
        
        #making Video game music subsets based on frequency
        cleaned_data_vgm1 = cleaned_data[cleaned_data["Frequency [Video game music]"] == 1]
        cleaned_data_vgm2 = cleaned_data[cleaned_data["Frequency [Video game music]"] == 2]
        cleaned_data_vgm3 = cleaned_data[cleaned_data["Frequency [Video game music]"] == 3]
        cleaned_data_vgm4 = cleaned_data[cleaned_data["Frequency [Video game music]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_vgm1 = cleaned_data_vgm1["Anxiety"].mean()
        ave_dep_vgm1 = cleaned_data_vgm1["Depression"].mean()
        ave_insom_vgm1 = cleaned_data_vgm1["Insomnia"].mean()
        ave_ocd_vgm1 = cleaned_data_vgm1["OCD"].mean()
        
        ave_anxiety_vgm2 = cleaned_data_vgm2["Anxiety"].mean()
        ave_dep_vgm2 = cleaned_data_vgm2["Depression"].mean()
        ave_insom_vgm2 = cleaned_data_vgm2["Insomnia"].mean()
        ave_ocd_vgm2 = cleaned_data_vgm2["OCD"].mean()
        
        ave_anxiety_vgm3 = cleaned_data_vgm3["Anxiety"].mean()
        ave_dep_vgm3 = cleaned_data_vgm3["Depression"].mean()
        ave_insom_vgm3 = cleaned_data_vgm3["Insomnia"].mean()
        ave_ocd_vgm3 = cleaned_data_vgm3["OCD"].mean()
        
        ave_anxiety_vgm4 = cleaned_data_vgm4["Anxiety"].mean()
        ave_dep_vgm4 = cleaned_data_vgm4["Depression"].mean()
        ave_insom_vgm4 = cleaned_data_vgm4["Insomnia"].mean()
        ave_ocd_vgm4 = cleaned_data_vgm4["OCD"].mean()
        
        
        
        #making Jazz subsets based on frequency
        cleaned_data_jazz1 = cleaned_data[cleaned_data["Frequency [Jazz]"] == 1]
        cleaned_data_jazz2 = cleaned_data[cleaned_data["Frequency [Jazz]"] == 2]
        cleaned_data_jazz3 = cleaned_data[cleaned_data["Frequency [Jazz]"] == 3]
        cleaned_data_jazz4 = cleaned_data[cleaned_data["Frequency [Jazz]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_jazz1 = cleaned_data_jazz1["Anxiety"].mean()
        ave_dep_jazz1 = cleaned_data_jazz1["Depression"].mean()
        ave_insom_jazz1 = cleaned_data_jazz1["Insomnia"].mean()
        ave_ocd_jazz1 = cleaned_data_jazz1["OCD"].mean()
        
        ave_anxiety_jazz2 = cleaned_data_jazz2["Anxiety"].mean()
        ave_dep_jazz2 = cleaned_data_jazz2["Depression"].mean()
        ave_insom_jazz2 = cleaned_data_jazz2["Insomnia"].mean()
        ave_ocd_jazz2 = cleaned_data_jazz2["OCD"].mean()
        
        ave_anxiety_jazz3 = cleaned_data_jazz3["Anxiety"].mean()
        ave_dep_jazz3 = cleaned_data_jazz3["Depression"].mean()
        ave_insom_jazz3 = cleaned_data_jazz3["Insomnia"].mean()
        ave_ocd_jazz3 = cleaned_data_jazz3["OCD"].mean()
        
        ave_anxiety_jazz4 = cleaned_data_jazz4["Anxiety"].mean()
        ave_dep_jazz4 = cleaned_data_jazz4["Depression"].mean()
        ave_insom_jazz4 = cleaned_data_jazz4["Insomnia"].mean()
        ave_ocd_jazz4 = cleaned_data_jazz4["OCD"].mean()
        
        
        
        #making R&B subsets based on frequency
        cleaned_data_rnb1 = cleaned_data[cleaned_data["Frequency [R&B]"] == 1]
        cleaned_data_rnb2 = cleaned_data[cleaned_data["Frequency [R&B]"] == 2]
        cleaned_data_rnb3 = cleaned_data[cleaned_data["Frequency [R&B]"] == 3]
        cleaned_data_rnb4 = cleaned_data[cleaned_data["Frequency [R&B]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_rnb1 = cleaned_data_rnb1["Anxiety"].mean()
        ave_dep_rnb1 = cleaned_data_rnb1["Depression"].mean()
        ave_insom_rnb1 = cleaned_data_rnb1["Insomnia"].mean()
        ave_ocd_rnb1 = cleaned_data_rnb1["OCD"].mean()
        
        ave_anxiety_rnb2 = cleaned_data_rnb2["Anxiety"].mean()
        ave_dep_rnb2 = cleaned_data_rnb2["Depression"].mean()
        ave_insom_rnb2 = cleaned_data_rnb2["Insomnia"].mean()
        ave_ocd_rnb2 = cleaned_data_rnb2["OCD"].mean()
        
        ave_anxiety_rnb3 = cleaned_data_rnb3["Anxiety"].mean()
        ave_dep_rnb3 = cleaned_data_rnb3["Depression"].mean()
        ave_insom_rnb3 = cleaned_data_rnb3["Insomnia"].mean()
        ave_ocd_rnb3 = cleaned_data_rnb3["OCD"].mean()
        
        ave_anxiety_rnb4 = cleaned_data_rnb4["Anxiety"].mean()
        ave_dep_rnb4 = cleaned_data_rnb4["Depression"].mean()
        ave_insom_rnb4 = cleaned_data_rnb4["Insomnia"].mean()
        ave_ocd_rnb4 = cleaned_data_rnb4["OCD"].mean()
        
        
        
        #making K pop subsets based on frequency
        cleaned_data_kpop1 = cleaned_data[cleaned_data["Frequency [K pop]"] == 1]
        cleaned_data_kpop2 = cleaned_data[cleaned_data["Frequency [K pop]"] == 2]
        cleaned_data_kpop3 = cleaned_data[cleaned_data["Frequency [K pop]"] == 3]
        cleaned_data_kpop4 = cleaned_data[cleaned_data["Frequency [K pop]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_kpop1 = cleaned_data_kpop1["Anxiety"].mean()
        ave_dep_kpop1 = cleaned_data_kpop1["Depression"].mean()
        ave_insom_kpop1 = cleaned_data_kpop1["Insomnia"].mean()
        ave_ocd_kpop1 = cleaned_data_kpop1["OCD"].mean()
        
        ave_anxiety_kpop2 = cleaned_data_kpop2["Anxiety"].mean()
        ave_dep_kpop2 = cleaned_data_kpop2["Depression"].mean()
        ave_insom_kpop2 = cleaned_data_kpop2["Insomnia"].mean()
        ave_ocd_kpop2 = cleaned_data_kpop2["OCD"].mean()
        
        ave_anxiety_kpop3 = cleaned_data_kpop3["Anxiety"].mean()
        ave_dep_kpop3 = cleaned_data_kpop3["Depression"].mean()
        ave_insom_kpop3 = cleaned_data_kpop3["Insomnia"].mean()
        ave_ocd_kpop3 = cleaned_data_kpop3["OCD"].mean()
        
        ave_anxiety_kpop4 = cleaned_data_kpop4["Anxiety"].mean()
        ave_dep_kpop4 = cleaned_data_kpop4["Depression"].mean()
        ave_insom_kpop4 = cleaned_data_kpop4["Insomnia"].mean()
        ave_ocd_kpop4 = cleaned_data_kpop4["OCD"].mean()
        
        
        
        #making Country subsets based on frequency
        cleaned_data_country1 = cleaned_data[cleaned_data["Frequency [Country]"] == 1]
        cleaned_data_country2 = cleaned_data[cleaned_data["Frequency [Country]"] == 2]
        cleaned_data_country3 = cleaned_data[cleaned_data["Frequency [Country]"] == 3]
        cleaned_data_country4 = cleaned_data[cleaned_data["Frequency [Country]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_country1 = cleaned_data_country1["Anxiety"].mean()
        ave_dep_country1 = cleaned_data_country1["Depression"].mean()
        ave_insom_country1 = cleaned_data_country1["Insomnia"].mean()
        ave_ocd_country1 = cleaned_data_country1["OCD"].mean()
        
        ave_anxiety_country2 = cleaned_data_country2["Anxiety"].mean()
        ave_dep_country2 = cleaned_data_country2["Depression"].mean()
        ave_insom_country2 = cleaned_data_country2["Insomnia"].mean()
        ave_ocd_country2 = cleaned_data_country2["OCD"].mean()
        
        ave_anxiety_country3 = cleaned_data_country3["Anxiety"].mean()
        ave_dep_country3 = cleaned_data_country3["Depression"].mean()
        ave_insom_country3 = cleaned_data_country3["Insomnia"].mean()
        ave_ocd_country3 = cleaned_data_country3["OCD"].mean()
        
        ave_anxiety_country4 = cleaned_data_country4["Anxiety"].mean()
        ave_dep_country4 = cleaned_data_country4["Depression"].mean()
        ave_insom_country4 = cleaned_data_country4["Insomnia"].mean()
        ave_ocd_country4 = cleaned_data_country4["OCD"].mean()
        
        
        
        #making EDM subsets based on frequency
        cleaned_data_edm1 = cleaned_data[cleaned_data["Frequency [EDM]"] == 1]
        cleaned_data_edm2 = cleaned_data[cleaned_data["Frequency [EDM]"] == 2]
        cleaned_data_edm3 = cleaned_data[cleaned_data["Frequency [EDM]"] == 3]
        cleaned_data_edm4 = cleaned_data[cleaned_data["Frequency [EDM]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_edm1 = cleaned_data_edm1["Anxiety"].mean()
        ave_dep_edm1 = cleaned_data_edm1["Depression"].mean()
        ave_insom_edm1 = cleaned_data_edm1["Insomnia"].mean()
        ave_ocd_edm1 = cleaned_data_edm1["OCD"].mean()
        
        ave_anxiety_edm2 = cleaned_data_edm2["Anxiety"].mean()
        ave_dep_edm2 = cleaned_data_edm2["Depression"].mean()
        ave_insom_edm2 = cleaned_data_edm2["Insomnia"].mean()
        ave_ocd_edm2 = cleaned_data_edm2["OCD"].mean()
        
        ave_anxiety_edm3 = cleaned_data_edm3["Anxiety"].mean()
        ave_dep_edm3 = cleaned_data_edm3["Depression"].mean()
        ave_insom_edm3 = cleaned_data_edm3["Insomnia"].mean()
        ave_ocd_edm3 = cleaned_data_edm3["OCD"].mean()
        
        ave_anxiety_edm4 = cleaned_data_edm4["Anxiety"].mean()
        ave_dep_edm4 = cleaned_data_edm4["Depression"].mean()
        ave_insom_edm4 = cleaned_data_edm4["Insomnia"].mean()
        ave_ocd_edm4 = cleaned_data_edm4["OCD"].mean()
        
        
        
        #making Hip hop subsets based on frequency
        cleaned_data_hiphop1 = cleaned_data[cleaned_data["Frequency [Hip hop]"] == 1]
        cleaned_data_hiphop2 = cleaned_data[cleaned_data["Frequency [Hip hop]"] == 2]
        cleaned_data_hiphop3 = cleaned_data[cleaned_data["Frequency [Hip hop]"] == 3]
        cleaned_data_hiphop4 = cleaned_data[cleaned_data["Frequency [Hip hop]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_hiphop1 = cleaned_data_hiphop1["Anxiety"].mean()
        ave_dep_hiphop1 = cleaned_data_hiphop1["Depression"].mean()
        ave_insom_hiphop1 = cleaned_data_hiphop1["Insomnia"].mean()
        ave_ocd_hiphop1 = cleaned_data_hiphop1["OCD"].mean()
        
        ave_anxiety_hiphop2 = cleaned_data_hiphop2["Anxiety"].mean()
        ave_dep_hiphop2 = cleaned_data_hiphop2["Depression"].mean()
        ave_insom_hiphop2 = cleaned_data_hiphop2["Insomnia"].mean()
        ave_ocd_hiphop2 = cleaned_data_hiphop2["OCD"].mean()
        
        ave_anxiety_hiphop3 = cleaned_data_hiphop3["Anxiety"].mean()
        ave_dep_hiphop3 = cleaned_data_hiphop3["Depression"].mean()
        ave_insom_hiphop3 = cleaned_data_hiphop3["Insomnia"].mean()
        ave_ocd_hiphop3 = cleaned_data_hiphop3["OCD"].mean()
        
        ave_anxiety_hiphop4 = cleaned_data_hiphop4["Anxiety"].mean()
        ave_dep_hiphop4 = cleaned_data_hiphop4["Depression"].mean()
        ave_insom_hiphop4 = cleaned_data_hiphop4["Insomnia"].mean()
        ave_ocd_hiphop4 = cleaned_data_hiphop4["OCD"].mean()
        
        
        
        
        #making Pop subsets based on frequency
        cleaned_data_pop1 = cleaned_data[cleaned_data["Frequency [Pop]"] == 1]
        cleaned_data_pop2 = cleaned_data[cleaned_data["Frequency [Pop]"] == 2]
        cleaned_data_pop3 = cleaned_data[cleaned_data["Frequency [Pop]"] == 3]
        cleaned_data_pop4 = cleaned_data[cleaned_data["Frequency [Pop]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_pop1 = cleaned_data_pop1["Anxiety"].mean()
        ave_dep_pop1 = cleaned_data_pop1["Depression"].mean()
        ave_insom_pop1 = cleaned_data_pop1["Insomnia"].mean()
        ave_ocd_pop1 = cleaned_data_pop1["OCD"].mean()
        
        ave_anxiety_pop2 = cleaned_data_pop2["Anxiety"].mean()
        ave_dep_pop2 = cleaned_data_pop2["Depression"].mean()
        ave_insom_pop2 = cleaned_data_pop2["Insomnia"].mean()
        ave_ocd_pop2 = cleaned_data_pop2["OCD"].mean()
        
        ave_anxiety_pop3 = cleaned_data_pop3["Anxiety"].mean()
        ave_dep_pop3 = cleaned_data_pop3["Depression"].mean()
        ave_insom_pop3 = cleaned_data_pop3["Insomnia"].mean()
        ave_ocd_pop3 = cleaned_data_pop3["OCD"].mean()
        
        ave_anxiety_pop4 = cleaned_data_pop4["Anxiety"].mean()
        ave_dep_pop4 = cleaned_data_pop4["Depression"].mean()
        ave_insom_pop4 = cleaned_data_pop4["Insomnia"].mean()
        ave_ocd_pop4 = cleaned_data_pop4["OCD"].mean()
        
        
        
        
        #making Rap subsets based on frequency
        cleaned_data_rap1 = cleaned_data[cleaned_data["Frequency [Rap]"] == 1]
        cleaned_data_rap2 = cleaned_data[cleaned_data["Frequency [Rap]"] == 2]
        cleaned_data_rap3 = cleaned_data[cleaned_data["Frequency [Rap]"] == 3]
        cleaned_data_rap4 = cleaned_data[cleaned_data["Frequency [Rap]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_rap1 = cleaned_data_rap1["Anxiety"].mean()
        ave_dep_rap1 = cleaned_data_rap1["Depression"].mean()
        ave_insom_rap1 = cleaned_data_rap1["Insomnia"].mean()
        ave_ocd_rap1 = cleaned_data_rap1["OCD"].mean()
        
        ave_anxiety_rap2 = cleaned_data_rap2["Anxiety"].mean()
        ave_dep_rap2 = cleaned_data_rap2["Depression"].mean()
        ave_insom_rap2 = cleaned_data_rap2["Insomnia"].mean()
        ave_ocd_rap2 = cleaned_data_rap2["OCD"].mean()
        
        ave_anxiety_rap3 = cleaned_data_rap3["Anxiety"].mean()
        ave_dep_rap3 = cleaned_data_rap3["Depression"].mean()
        ave_insom_rap3 = cleaned_data_rap3["Insomnia"].mean()
        ave_ocd_rap3 = cleaned_data_rap3["OCD"].mean()
        
        ave_anxiety_rap4 = cleaned_data_rap4["Anxiety"].mean()
        ave_dep_rap4 = cleaned_data_rap4["Depression"].mean()
        ave_insom_rap4 = cleaned_data_rap4["Insomnia"].mean()
        ave_ocd_rap4 = cleaned_data_rap4["OCD"].mean()
        
        
        
        #making Classical subsets based on frequency
        cleaned_data_classical1 = cleaned_data[cleaned_data["Frequency [Classical]"] == 1]
        cleaned_data_classical2 = cleaned_data[cleaned_data["Frequency [Classical]"] == 2]
        cleaned_data_classical3 = cleaned_data[cleaned_data["Frequency [Classical]"] == 3]
        cleaned_data_classical4 = cleaned_data[cleaned_data["Frequency [Classical]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_classical1 = cleaned_data_classical1["Anxiety"].mean()
        ave_dep_classical1 = cleaned_data_classical1["Depression"].mean()
        ave_insom_classical1 = cleaned_data_classical1["Insomnia"].mean()
        ave_ocd_classical1 = cleaned_data_classical1["OCD"].mean()
        
        ave_anxiety_classical2 = cleaned_data_classical2["Anxiety"].mean()
        ave_dep_classical2 = cleaned_data_classical2["Depression"].mean()
        ave_insom_classical2 = cleaned_data_classical2["Insomnia"].mean()
        ave_ocd_classical2 = cleaned_data_classical2["OCD"].mean()
        
        ave_anxiety_classical3 = cleaned_data_classical3["Anxiety"].mean()
        ave_dep_classical3 = cleaned_data_classical3["Depression"].mean()
        ave_insom_classical3 = cleaned_data_classical3["Insomnia"].mean()
        ave_ocd_classical3 = cleaned_data_classical3["OCD"].mean()
        
        ave_anxiety_classical4 = cleaned_data_classical4["Anxiety"].mean()
        ave_dep_classical4 = cleaned_data_classical4["Depression"].mean()
        ave_insom_classical4 = cleaned_data_classical4["Insomnia"].mean()
        ave_ocd_classical4 = cleaned_data_classical4["OCD"].mean()
        
        
        
        #making Metal subsets based on frequency
        cleaned_data_metal1 = cleaned_data[cleaned_data["Frequency [Metal]"] == 1]
        cleaned_data_metal2 = cleaned_data[cleaned_data["Frequency [Metal]"] == 2]
        cleaned_data_metal3 = cleaned_data[cleaned_data["Frequency [Metal]"] == 3]
        cleaned_data_metal4 = cleaned_data[cleaned_data["Frequency [Metal]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_metal1 = cleaned_data_metal1["Anxiety"].mean()
        ave_dep_metal1 = cleaned_data_metal1["Depression"].mean()
        ave_insom_metal1 = cleaned_data_metal1["Insomnia"].mean()
        ave_ocd_metal1 = cleaned_data_metal1["OCD"].mean()
        
        ave_anxiety_metal2 = cleaned_data_metal2["Anxiety"].mean()
        ave_dep_metal2 = cleaned_data_metal2["Depression"].mean()
        ave_insom_metal2 = cleaned_data_metal2["Insomnia"].mean()
        ave_ocd_metal2 = cleaned_data_metal2["OCD"].mean()
        
        ave_anxiety_metal3 = cleaned_data_metal3["Anxiety"].mean()
        ave_dep_metal3 = cleaned_data_metal3["Depression"].mean()
        ave_insom_metal3 = cleaned_data_metal3["Insomnia"].mean()
        ave_ocd_metal3 = cleaned_data_metal3["OCD"].mean()
        
        ave_anxiety_metal4 = cleaned_data_metal4["Anxiety"].mean()
        ave_dep_metal4 = cleaned_data_metal4["Depression"].mean()
        ave_insom_metal4 = cleaned_data_metal4["Insomnia"].mean()
        ave_ocd_metal4 = cleaned_data_metal4["OCD"].mean()
        
        
        
        
        #making Folk subsets based on frequency
        cleaned_data_folk1 = cleaned_data[cleaned_data["Frequency [Folk]"] == 1]
        cleaned_data_folk2 = cleaned_data[cleaned_data["Frequency [Folk]"] == 2]
        cleaned_data_folk3 = cleaned_data[cleaned_data["Frequency [Folk]"] == 3]
        cleaned_data_folk4 = cleaned_data[cleaned_data["Frequency [Folk]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_folk1 = cleaned_data_folk1["Anxiety"].mean()
        ave_dep_folk1 = cleaned_data_folk1["Depression"].mean()
        ave_insom_folk1 = cleaned_data_folk1["Insomnia"].mean()
        ave_ocd_folk1 = cleaned_data_folk1["OCD"].mean()
        
        ave_anxiety_folk2 = cleaned_data_folk2["Anxiety"].mean()
        ave_dep_folk2 = cleaned_data_folk2["Depression"].mean()
        ave_insom_folk2 = cleaned_data_folk2["Insomnia"].mean()
        ave_ocd_folk2 = cleaned_data_folk2["OCD"].mean()
        
        ave_anxiety_folk3 = cleaned_data_folk3["Anxiety"].mean()
        ave_dep_folk3 = cleaned_data_folk3["Depression"].mean()
        ave_insom_folk3 = cleaned_data_folk3["Insomnia"].mean()
        ave_ocd_folk3 = cleaned_data_folk3["OCD"].mean()
        
        ave_anxiety_folk4 = cleaned_data_folk4["Anxiety"].mean()
        ave_dep_folk4 = cleaned_data_folk4["Depression"].mean()
        ave_insom_folk4 = cleaned_data_folk4["Insomnia"].mean()
        ave_ocd_folk4 = cleaned_data_folk4["OCD"].mean()
        
        
        
        
        
        #making Lofi subsets based on frequency
        cleaned_data_lofi1 = cleaned_data[cleaned_data["Frequency [Lofi]"] == 1]
        cleaned_data_lofi2 = cleaned_data[cleaned_data["Frequency [Lofi]"] == 2]
        cleaned_data_lofi3 = cleaned_data[cleaned_data["Frequency [Lofi]"] == 3]
        cleaned_data_lofi4 = cleaned_data[cleaned_data["Frequency [Lofi]"] == 4]
        
        #Now get the average MH scores for each frequency
        ave_anxiety_lofi1 = cleaned_data_lofi1["Anxiety"].mean()
        ave_dep_lofi1 = cleaned_data_lofi1["Depression"].mean()
        ave_insom_lofi1 = cleaned_data_lofi1["Insomnia"].mean()
        ave_ocd_lofi1 = cleaned_data_lofi1["OCD"].mean()
        
        ave_anxiety_lofi2 = cleaned_data_lofi2["Anxiety"].mean()
        ave_dep_lofi2 = cleaned_data_lofi2["Depression"].mean()
        ave_insom_lofi2 = cleaned_data_lofi2["Insomnia"].mean()
        ave_ocd_lofi2 = cleaned_data_lofi2["OCD"].mean()
        
        ave_anxiety_lofi3 = cleaned_data_lofi3["Anxiety"].mean()
        ave_dep_lofi3 = cleaned_data_lofi3["Depression"].mean()
        ave_insom_lofi3 = cleaned_data_lofi3["Insomnia"].mean()
        ave_ocd_lofi3 = cleaned_data_lofi3["OCD"].mean()
        
        ave_anxiety_lofi4 = cleaned_data_lofi4["Anxiety"].mean()
        ave_dep_lofi4 = cleaned_data_lofi4["Depression"].mean()
        ave_insom_lofi4 = cleaned_data_lofi4["Insomnia"].mean()
        ave_ocd_lofi4 = cleaned_data_lofi4["OCD"].mean()
        
        
        
        
        #making Gospel subsets based on frequency
        cleaned_data_gospel1 = cleaned_data[cleaned_data["Frequency [Gospel]"] == 1]
        cleaned_data_gospel2 = cleaned_data[cleaned_data["Frequency [Gospel]"] == 2]
        cleaned_data_gospel3 = cleaned_data[cleaned_data["Frequency [Gospel]"] == 3]
        cleaned_data_gospel4 = cleaned_data[cleaned_data["Frequency [Gospel]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_gospel1 = cleaned_data_gospel1["Anxiety"].mean()
        ave_dep_gospel1 = cleaned_data_gospel1["Depression"].mean()
        ave_insom_gospel1 = cleaned_data_gospel1["Insomnia"].mean()
        ave_ocd_gospel1 = cleaned_data_gospel1["OCD"].mean()
        
        ave_anxiety_gospel2 = cleaned_data_gospel2["Anxiety"].mean()
        ave_dep_gospel2 = cleaned_data_gospel2["Depression"].mean()
        ave_insom_gospel2 = cleaned_data_gospel2["Insomnia"].mean()
        ave_ocd_gospel2 = cleaned_data_gospel2["OCD"].mean()
        
        ave_anxiety_gospel3 = cleaned_data_gospel3["Anxiety"].mean()
        ave_dep_gospel3 = cleaned_data_gospel3["Depression"].mean()
        ave_insom_gospel3 = cleaned_data_gospel3["Insomnia"].mean()
        ave_ocd_gospel3 = cleaned_data_gospel3["OCD"].mean()
        
        ave_anxiety_gospel4 = cleaned_data_gospel4["Anxiety"].mean()
        ave_dep_gospel4 = cleaned_data_gospel4["Depression"].mean()
        ave_insom_gospel4 = cleaned_data_gospel4["Insomnia"].mean()
        ave_ocd_gospel4 = cleaned_data_gospel4["OCD"].mean()
        
        #create a dataframe for these values
        index = ["Classical", "Country", "EDM", "Folk", "Gospel", "Hip hop", "Jazz", "K pop", "Latin", "Lofi", 
                 "Metal", "Pop", "R&B", "Rap", "Rock", "Video game music"]
        columns = ["Anxiety", "Depression", "Insomnia", "OCD"]
        
        mh_by_genre = pd.DataFrame(index=index, columns=columns)
        
    
        #add values
        average_anxiety = [ave_anxiety_classical4, ave_anxiety_country4, ave_anxiety_edm4,  ave_anxiety_folk4, ave_anxiety_gospel4, ave_anxiety_hiphop4, 
            ave_anxiety_jazz4, ave_anxiety_kpop4, ave_anxiety_latin4, ave_anxiety_lofi4, ave_anxiety_metal4, ave_anxiety_pop4, ave_anxiety_rnb4, 
            ave_anxiety_rap4, ave_anxiety_rock4, ave_anxiety_vgm4]
        
        average_depression = [ave_dep_classical4, ave_dep_country4, ave_dep_edm4, ave_dep_folk4, ave_dep_gospel4, ave_dep_hiphop4, ave_dep_jazz4, 
                ave_dep_kpop4, ave_dep_latin4, ave_dep_lofi4, ave_dep_metal4, ave_dep_pop4, ave_dep_rnb4, ave_dep_rap4, ave_dep_rock4, ave_dep_vgm4]
        
        average_ocd = [ave_ocd_classical4, ave_ocd_country4, ave_ocd_edm4, ave_ocd_folk4, ave_ocd_gospel4, ave_ocd_hiphop4, ave_ocd_jazz4, ave_ocd_kpop4, 
            ave_ocd_latin4, ave_ocd_lofi4, ave_ocd_metal4, ave_ocd_pop4, ave_ocd_rnb4, ave_ocd_rap4, ave_ocd_rock4, ave_ocd_vgm4]
        
        average_insomnia = [ave_insom_classical4, ave_insom_country4, ave_insom_edm4, ave_insom_folk4, ave_insom_gospel4, ave_insom_hiphop4, ave_insom_jazz4, 
                ave_insom_kpop4, ave_insom_latin4, ave_insom_lofi4, ave_insom_metal4, ave_insom_pop4, ave_insom_rnb4, ave_insom_rap4, ave_insom_rock4, 
                ave_insom_vgm4]
        
        mh_by_genre["Anxiety"] = average_anxiety
        mh_by_genre["Depression"] = average_depression
        mh_by_genre["Insomnia"] = average_insomnia
        mh_by_genre["OCD"] = average_ocd
        
        st.write(mh_by_genre)  
    
    
        st.subheader("Heatmap of Genre and Average Mental Health Stat (based only on Very Frequent responses)")
        fig = px.imshow(mh_by_genre.T, labels=dict(x="MH Score", y="Genre", color="Score"), color_continuous_scale="Viridis", 
        title="Interactive Heatmap of Mental Health Scores by Genre",)
        # Show the heatmap
        st.plotly_chart(fig)
        st.write("This heatmap plots each average mental health metric among Very Frequent listeners of each genre. For example, the average Anxiety score among participants who report listening to Classical music 'Very Frequently' is 4.29.")
        
        st.subheader("Build A Dataframe For These Engineered Features")
        st.markdown("To identify the mental health metric with the highest variability (SD), observe descriptive statistics of this new dataframe. This metric would provide most variability if only focusing on one. This came out to be Depression (sd = 0.902; Anxiety SD = 0.551, Insomnia SD = 0.499, OCD SD = 0.553)")
        st.write(mh_by_genre.describe())
        
        st.markdown("Next, create a binary feature that expresses whether the average depression score for a given genre is above or below 5. This is how the app will recommend genres to users based on user input (listening goals.)")
    
        mh_by_genre["Dep Effect"] = np.where(mh_by_genre["Depression"] >= 5, 1, 0)

        mh_by_genre

        st.markdown("Add Effect columns for the other mental health categories:")
        
        mh_by_genre["Anx Effect"] = np.where(mh_by_genre["Anxiety"] >= 5, 1, 0)
        mh_by_genre["Ins Effect"] = np.where(mh_by_genre["Insomnia"] >= 5, 1, 0)
        mh_by_genre["OCD Effect"] = np.where(mh_by_genre["OCD"] >= 5, 1, 0)
        
        mh_by_genre

        st.write("There are little to no average OCD and Insomnia scores above 5. This may be a future area of improvement. We can reduce our threshold to the average OCD and Insomnia scores for those Effect columns.")
    
    
        #This dataframe will be used to connect this analysis with the second dataset.
        effect_df = mh_by_genre.reset_index(names='Genre')
        effect_df.drop(["Anxiety", "Depression", "OCD", "Insomnia"], axis=1)
        st.write("Make Genre a column instead of the index so we can merge the datasets on that column.")
        st.write(effect_df)
    
        st.markdown("Join the two datasets by their mutual column (Genre) to result in a merged dataset with song titles, artist, valence, energy, danceability, duration, average anxiety score, average depression score, average, average OCD score, average insomnia score, and effect (whether each mental health metric is above 5 or below 5.")

        st.markdown("See the cleaned and balanced second dataset again:")
        songs_balanced = songs_balanced.drop("valence_category", axis = 1)
        st.write(songs_balanced)

        #First I have to make sure the genre columns are capitalized the same
        st.write("Capitalize the genre column so it matches in both datasets before merging on 'Genre'.")
        songs_balanced.rename(columns={'genre': 'Genre'}, inplace=True)
        st.write(songs_balanced)

        st.markdown("The merged dataset is shown below. Each song will have an average mental health score and a mental health effect (above or below 5), both based on Dataset #1.")
        merged_df = pd.merge(songs_balanced, effect_df, on='Genre', how='left')
        st.write(merged_df)

        st.subheader("Investigate balance between classes:")

        #plot of Dep Effect
    
        plt.figure(figsize=(10, 6))
        plt.hist(merged_df["Dep Effect"], bins=11, edgecolor='black')
        
        #set the title of the plot
        plt.title('Distribution of Dep Effect')
        
        #set the x-axis title
        plt.xlabel('Depression Below 5 (0) and Above 5 (1)')
        plt.xticks([0, 1])
        st.pyplot(plt)

        #plot of Anx Effect
    
        plt.figure(figsize=(10, 6))
        plt.hist(merged_df["Anx Effect"], bins=11, edgecolor='black')
        
        #set the title of the plot
        plt.title('Distribution of Anx Effect')
        
        #set the x-axis title
        plt.xlabel('Anxiety Below 5 (0) and Above 5 (1)')
        plt.xticks([0, 1])
        st.pyplot(plt)

        #plot of OCD Effect
    
        plt.figure(figsize=(10, 6))
        plt.hist(merged_df["OCD Effect"], bins=11, edgecolor='black')
        
        #set the title of the plot
        plt.title('Distribution of OCD Effect')
        
        #set the x-axis title
        plt.xlabel('OCD Below 5 (0) and Above 5 (1)')
        plt.xticks([0, 1])
        st.pyplot(plt)

        #plot of Ins Effect
    
        plt.figure(figsize=(10, 6))
        plt.hist(merged_df["Ins Effect"], bins=11, edgecolor='black')
        
        #set the title of the plot
        plt.title('Distribution of Ins Effect')
        
        #set the x-axis title
        plt.xlabel('Insomnia Below 5 (0) and Above 5 (1)')
        plt.xticks([0, 1])
        st.pyplot(plt)
    
        st.write("These plots demonstrate the balance of mental health effect within the app's library. This library could be augmented with more songs to balance these features.")
        st.write("OCD and Insomnia have all 0s. As discussed above, this could be changed by changing the threshold.")
        st.write("Many 0s in Dep Effect, and not many 0s in Anx Effect.")

        st.subheader("Final feature combinations used to filter song recommendations:")
        st.write("As discussed on the Improving App Performance page, all final recommendations will go through two stages of filtering as described below.")
        st.write("Happy | Genres with a Dep Effect of 0 filtered by above average valence (over the whole dataset).")
        st.write("Sad | Genres with a Dep Effect of 1 filtered by below average valence, danceability, and energy (over the whole dataset).")
        st.write("Calm | Genres with an Anx Effect of 0 filtered by valence, energy, and danceability scores below or equal to 0.6 (over the whole dataset) to identify the most neutral songs. Filtering by the average feature scores resulted in high feature distributions.")
        st.write("Dance | Genres with a Dep Effect of 0 filtered by above average danceability (over the whole dataset).")
        
        #################################### this begins the practice Get Recommendations Page 


        #if selected_category == "Get Recommendations":
    if selected_category == "Improving App Performance":
    
        #title of the app
        st.title("Welcome To Tunes By Mood: A Music Therapy App Designed For You")
        st.markdown("Here, app performance is tested and improved.")
        #st.markdown("Please be advised that all recommendations are based on self-reported mental health scores of listeners. Since these recommendations are based on the correlations between listening preferences and mental health, they are not proven to *cause* changes in mood, but rather are *associated* with changes in mood.")
        st.markdown("Because there is variety within any given genre, it's not enough to meet the user's goals by recommending songs based on genre alone. For example, in the Pop genre, there exists happy, sad, and calm songs. To fill this gap, the app will also consider Energy, Danceability, and Valence to improve its recommendations. The app will recommend genres associated with mental health scores and then select songs from those genres based on features levels that meet the user's goals. (For example, high energy if the goal is to feel happy and average energy if the goal is to feel calm.")
        st.subheader("Find Your Tunes Now")
        
        
        ########################repeating data edits so this dropdown option can use the same updated data
        
        
        #load the Data
        mxmh_survey_results = pd.read_csv("mxmh_survey_results.csv")
        
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
        
        
        cleaned_data = mxmh_survey_results.copy()
        #I will say the max they could realistically listen to is 16 hrs
        cleaned_data = cleaned_data[(cleaned_data["Hours per day"] < 16)]
        #deleted 6 rows
        
        #take away age outliers 
        cleaned_data = cleaned_data[(cleaned_data["Age"] > 18) & (cleaned_data["Age"] < 64)]

        #take away MH outliers             
        cleaned_data = cleaned_data[
        (cleaned_data["Anxiety"] >= 0) & (cleaned_data["Anxiety"] <= 10) &
        (cleaned_data["Depression"] >= 0) & (cleaned_data["Depression"] <= 10) &
        (cleaned_data["OCD"] >= 0) & (cleaned_data["OCD"] <= 10) &
        (cleaned_data["Insomnia"] >= 0) & (cleaned_data["Insomnia"] <= 10)
        ]
        
        #recode frequency genre
        
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
        
        
        #make aged binned
        
        bins = [18, 25, 31, 36, 41, 46, 51, 58, 64]  
        labels = ['18-24', '25-30', '31-35', '36-40', '41-45', '46-50', '51-57', '58-64']  # Labels for the bins
        
        # Create the binned column
        cleaned_data['age_binned'] = pd.cut(cleaned_data['Age'], bins=bins, labels=labels, right=False)
        
        
        
        
        #feature engineering
        
        #making latin subsets based on frequency
        cleaned_data_latin1 = cleaned_data[cleaned_data["Frequency [Latin]"] == 1]
        cleaned_data_latin2 = cleaned_data[cleaned_data["Frequency [Latin]"] == 2]
        cleaned_data_latin3 = cleaned_data[cleaned_data["Frequency [Latin]"] == 3]
        cleaned_data_latin4 = cleaned_data[cleaned_data["Frequency [Latin]"] == 4]
        
        
        #now get the average MH scores for each frequency
        ave_anxiety_latin1 = cleaned_data_latin1["Anxiety"].mean()
        ave_dep_latin1 = cleaned_data_latin1["Depression"].mean()
        ave_insom_latin1 = cleaned_data_latin1["Insomnia"].mean()
        ave_ocd_latin1 = cleaned_data_latin1["OCD"].mean()
        
        ave_anxiety_latin2 = cleaned_data_latin2["Anxiety"].mean()
        ave_dep_latin2 = cleaned_data_latin2["Depression"].mean()
        ave_insom_latin2 = cleaned_data_latin2["Insomnia"].mean()
        ave_ocd_latin2 = cleaned_data_latin2["OCD"].mean()
        
        ave_anxiety_latin3 = cleaned_data_latin3["Anxiety"].mean()
        ave_dep_latin3 = cleaned_data_latin3["Depression"].mean()
        ave_insom_latin3 = cleaned_data_latin3["Insomnia"].mean()
        ave_ocd_latin3 = cleaned_data_latin3["OCD"].mean()
        
        ave_anxiety_latin4 = cleaned_data_latin4["Anxiety"].mean()
        ave_dep_latin4 = cleaned_data_latin4["Depression"].mean()
        ave_insom_latin4 = cleaned_data_latin4["Insomnia"].mean()
        ave_ocd_latin4 = cleaned_data_latin4["OCD"].mean()
        
        
        
        #making rock subsets based on frequency
        cleaned_data_rock1 = cleaned_data[cleaned_data["Frequency [Rock]"] == 1]
        cleaned_data_rock2 = cleaned_data[cleaned_data["Frequency [Rock]"] == 2]
        cleaned_data_rock3 = cleaned_data[cleaned_data["Frequency [Rock]"] == 3]
        cleaned_data_rock4 = cleaned_data[cleaned_data["Frequency [Rock]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_rock1 = cleaned_data_rock1["Anxiety"].mean()
        ave_dep_rock1 = cleaned_data_rock1["Depression"].mean()
        ave_insom_rock1 = cleaned_data_rock1["Insomnia"].mean()
        ave_ocd_rock1 = cleaned_data_rock1["OCD"].mean()
        
        ave_anxiety_rock2 = cleaned_data_rock2["Anxiety"].mean()
        ave_dep_rock2 = cleaned_data_rock2["Depression"].mean()
        ave_insom_rock2 = cleaned_data_rock2["Insomnia"].mean()
        ave_ocd_rock2 = cleaned_data_rock2["OCD"].mean()
        
        ave_anxiety_rock3 = cleaned_data_rock3["Anxiety"].mean()
        ave_dep_rock3 = cleaned_data_rock3["Depression"].mean()
        ave_insom_rock3 = cleaned_data_rock3["Insomnia"].mean()
        ave_ocd_rock3 = cleaned_data_rock3["OCD"].mean()
        
        ave_anxiety_rock4 = cleaned_data_rock4["Anxiety"].mean()
        ave_dep_rock4 = cleaned_data_rock4["Depression"].mean()
        ave_insom_rock4 = cleaned_data_rock4["Insomnia"].mean()
        ave_ocd_rock4 = cleaned_data_rock4["OCD"].mean()
        
        
        
        
        #making Video game music subsets based on frequency
        cleaned_data_vgm1 = cleaned_data[cleaned_data["Frequency [Video game music]"] == 1]
        cleaned_data_vgm2 = cleaned_data[cleaned_data["Frequency [Video game music]"] == 2]
        cleaned_data_vgm3 = cleaned_data[cleaned_data["Frequency [Video game music]"] == 3]
        cleaned_data_vgm4 = cleaned_data[cleaned_data["Frequency [Video game music]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_vgm1 = cleaned_data_vgm1["Anxiety"].mean()
        ave_dep_vgm1 = cleaned_data_vgm1["Depression"].mean()
        ave_insom_vgm1 = cleaned_data_vgm1["Insomnia"].mean()
        ave_ocd_vgm1 = cleaned_data_vgm1["OCD"].mean()
        
        ave_anxiety_vgm2 = cleaned_data_vgm2["Anxiety"].mean()
        ave_dep_vgm2 = cleaned_data_vgm2["Depression"].mean()
        ave_insom_vgm2 = cleaned_data_vgm2["Insomnia"].mean()
        ave_ocd_vgm2 = cleaned_data_vgm2["OCD"].mean()
        
        ave_anxiety_vgm3 = cleaned_data_vgm3["Anxiety"].mean()
        ave_dep_vgm3 = cleaned_data_vgm3["Depression"].mean()
        ave_insom_vgm3 = cleaned_data_vgm3["Insomnia"].mean()
        ave_ocd_vgm3 = cleaned_data_vgm3["OCD"].mean()
        
        ave_anxiety_vgm4 = cleaned_data_vgm4["Anxiety"].mean()
        ave_dep_vgm4 = cleaned_data_vgm4["Depression"].mean()
        ave_insom_vgm4 = cleaned_data_vgm4["Insomnia"].mean()
        ave_ocd_vgm4 = cleaned_data_vgm4["OCD"].mean()
        
        
        
        #making Jazz subsets based on frequency
        cleaned_data_jazz1 = cleaned_data[cleaned_data["Frequency [Jazz]"] == 1]
        cleaned_data_jazz2 = cleaned_data[cleaned_data["Frequency [Jazz]"] == 2]
        cleaned_data_jazz3 = cleaned_data[cleaned_data["Frequency [Jazz]"] == 3]
        cleaned_data_jazz4 = cleaned_data[cleaned_data["Frequency [Jazz]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_jazz1 = cleaned_data_jazz1["Anxiety"].mean()
        ave_dep_jazz1 = cleaned_data_jazz1["Depression"].mean()
        ave_insom_jazz1 = cleaned_data_jazz1["Insomnia"].mean()
        ave_ocd_jazz1 = cleaned_data_jazz1["OCD"].mean()
        
        ave_anxiety_jazz2 = cleaned_data_jazz2["Anxiety"].mean()
        ave_dep_jazz2 = cleaned_data_jazz2["Depression"].mean()
        ave_insom_jazz2 = cleaned_data_jazz2["Insomnia"].mean()
        ave_ocd_jazz2 = cleaned_data_jazz2["OCD"].mean()
        
        ave_anxiety_jazz3 = cleaned_data_jazz3["Anxiety"].mean()
        ave_dep_jazz3 = cleaned_data_jazz3["Depression"].mean()
        ave_insom_jazz3 = cleaned_data_jazz3["Insomnia"].mean()
        ave_ocd_jazz3 = cleaned_data_jazz3["OCD"].mean()
        
        ave_anxiety_jazz4 = cleaned_data_jazz4["Anxiety"].mean()
        ave_dep_jazz4 = cleaned_data_jazz4["Depression"].mean()
        ave_insom_jazz4 = cleaned_data_jazz4["Insomnia"].mean()
        ave_ocd_jazz4 = cleaned_data_jazz4["OCD"].mean()
        
        
        
        #making R&B subsets based on frequency
        cleaned_data_rnb1 = cleaned_data[cleaned_data["Frequency [R&B]"] == 1]
        cleaned_data_rnb2 = cleaned_data[cleaned_data["Frequency [R&B]"] == 2]
        cleaned_data_rnb3 = cleaned_data[cleaned_data["Frequency [R&B]"] == 3]
        cleaned_data_rnb4 = cleaned_data[cleaned_data["Frequency [R&B]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_rnb1 = cleaned_data_rnb1["Anxiety"].mean()
        ave_dep_rnb1 = cleaned_data_rnb1["Depression"].mean()
        ave_insom_rnb1 = cleaned_data_rnb1["Insomnia"].mean()
        ave_ocd_rnb1 = cleaned_data_rnb1["OCD"].mean()
        
        ave_anxiety_rnb2 = cleaned_data_rnb2["Anxiety"].mean()
        ave_dep_rnb2 = cleaned_data_rnb2["Depression"].mean()
        ave_insom_rnb2 = cleaned_data_rnb2["Insomnia"].mean()
        ave_ocd_rnb2 = cleaned_data_rnb2["OCD"].mean()
        
        ave_anxiety_rnb3 = cleaned_data_rnb3["Anxiety"].mean()
        ave_dep_rnb3 = cleaned_data_rnb3["Depression"].mean()
        ave_insom_rnb3 = cleaned_data_rnb3["Insomnia"].mean()
        ave_ocd_rnb3 = cleaned_data_rnb3["OCD"].mean()
        
        ave_anxiety_rnb4 = cleaned_data_rnb4["Anxiety"].mean()
        ave_dep_rnb4 = cleaned_data_rnb4["Depression"].mean()
        ave_insom_rnb4 = cleaned_data_rnb4["Insomnia"].mean()
        ave_ocd_rnb4 = cleaned_data_rnb4["OCD"].mean()
        
        
        
        #making K pop subsets based on frequency
        cleaned_data_kpop1 = cleaned_data[cleaned_data["Frequency [K pop]"] == 1]
        cleaned_data_kpop2 = cleaned_data[cleaned_data["Frequency [K pop]"] == 2]
        cleaned_data_kpop3 = cleaned_data[cleaned_data["Frequency [K pop]"] == 3]
        cleaned_data_kpop4 = cleaned_data[cleaned_data["Frequency [K pop]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_kpop1 = cleaned_data_kpop1["Anxiety"].mean()
        ave_dep_kpop1 = cleaned_data_kpop1["Depression"].mean()
        ave_insom_kpop1 = cleaned_data_kpop1["Insomnia"].mean()
        ave_ocd_kpop1 = cleaned_data_kpop1["OCD"].mean()
        
        ave_anxiety_kpop2 = cleaned_data_kpop2["Anxiety"].mean()
        ave_dep_kpop2 = cleaned_data_kpop2["Depression"].mean()
        ave_insom_kpop2 = cleaned_data_kpop2["Insomnia"].mean()
        ave_ocd_kpop2 = cleaned_data_kpop2["OCD"].mean()
        
        ave_anxiety_kpop3 = cleaned_data_kpop3["Anxiety"].mean()
        ave_dep_kpop3 = cleaned_data_kpop3["Depression"].mean()
        ave_insom_kpop3 = cleaned_data_kpop3["Insomnia"].mean()
        ave_ocd_kpop3 = cleaned_data_kpop3["OCD"].mean()
        
        ave_anxiety_kpop4 = cleaned_data_kpop4["Anxiety"].mean()
        ave_dep_kpop4 = cleaned_data_kpop4["Depression"].mean()
        ave_insom_kpop4 = cleaned_data_kpop4["Insomnia"].mean()
        ave_ocd_kpop4 = cleaned_data_kpop4["OCD"].mean()
        
        
        
        #making Country subsets based on frequency
        cleaned_data_country1 = cleaned_data[cleaned_data["Frequency [Country]"] == 1]
        cleaned_data_country2 = cleaned_data[cleaned_data["Frequency [Country]"] == 2]
        cleaned_data_country3 = cleaned_data[cleaned_data["Frequency [Country]"] == 3]
        cleaned_data_country4 = cleaned_data[cleaned_data["Frequency [Country]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_country1 = cleaned_data_country1["Anxiety"].mean()
        ave_dep_country1 = cleaned_data_country1["Depression"].mean()
        ave_insom_country1 = cleaned_data_country1["Insomnia"].mean()
        ave_ocd_country1 = cleaned_data_country1["OCD"].mean()
        
        ave_anxiety_country2 = cleaned_data_country2["Anxiety"].mean()
        ave_dep_country2 = cleaned_data_country2["Depression"].mean()
        ave_insom_country2 = cleaned_data_country2["Insomnia"].mean()
        ave_ocd_country2 = cleaned_data_country2["OCD"].mean()
        
        ave_anxiety_country3 = cleaned_data_country3["Anxiety"].mean()
        ave_dep_country3 = cleaned_data_country3["Depression"].mean()
        ave_insom_country3 = cleaned_data_country3["Insomnia"].mean()
        ave_ocd_country3 = cleaned_data_country3["OCD"].mean()
        
        ave_anxiety_country4 = cleaned_data_country4["Anxiety"].mean()
        ave_dep_country4 = cleaned_data_country4["Depression"].mean()
        ave_insom_country4 = cleaned_data_country4["Insomnia"].mean()
        ave_ocd_country4 = cleaned_data_country4["OCD"].mean()
        
        
        
        #making EDM subsets based on frequency
        cleaned_data_edm1 = cleaned_data[cleaned_data["Frequency [EDM]"] == 1]
        cleaned_data_edm2 = cleaned_data[cleaned_data["Frequency [EDM]"] == 2]
        cleaned_data_edm3 = cleaned_data[cleaned_data["Frequency [EDM]"] == 3]
        cleaned_data_edm4 = cleaned_data[cleaned_data["Frequency [EDM]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_edm1 = cleaned_data_edm1["Anxiety"].mean()
        ave_dep_edm1 = cleaned_data_edm1["Depression"].mean()
        ave_insom_edm1 = cleaned_data_edm1["Insomnia"].mean()
        ave_ocd_edm1 = cleaned_data_edm1["OCD"].mean()
        
        ave_anxiety_edm2 = cleaned_data_edm2["Anxiety"].mean()
        ave_dep_edm2 = cleaned_data_edm2["Depression"].mean()
        ave_insom_edm2 = cleaned_data_edm2["Insomnia"].mean()
        ave_ocd_edm2 = cleaned_data_edm2["OCD"].mean()
        
        ave_anxiety_edm3 = cleaned_data_edm3["Anxiety"].mean()
        ave_dep_edm3 = cleaned_data_edm3["Depression"].mean()
        ave_insom_edm3 = cleaned_data_edm3["Insomnia"].mean()
        ave_ocd_edm3 = cleaned_data_edm3["OCD"].mean()
        
        ave_anxiety_edm4 = cleaned_data_edm4["Anxiety"].mean()
        ave_dep_edm4 = cleaned_data_edm4["Depression"].mean()
        ave_insom_edm4 = cleaned_data_edm4["Insomnia"].mean()
        ave_ocd_edm4 = cleaned_data_edm4["OCD"].mean()
        
        
        
        #making Hip hop subsets based on frequency
        cleaned_data_hiphop1 = cleaned_data[cleaned_data["Frequency [Hip hop]"] == 1]
        cleaned_data_hiphop2 = cleaned_data[cleaned_data["Frequency [Hip hop]"] == 2]
        cleaned_data_hiphop3 = cleaned_data[cleaned_data["Frequency [Hip hop]"] == 3]
        cleaned_data_hiphop4 = cleaned_data[cleaned_data["Frequency [Hip hop]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_hiphop1 = cleaned_data_hiphop1["Anxiety"].mean()
        ave_dep_hiphop1 = cleaned_data_hiphop1["Depression"].mean()
        ave_insom_hiphop1 = cleaned_data_hiphop1["Insomnia"].mean()
        ave_ocd_hiphop1 = cleaned_data_hiphop1["OCD"].mean()
        
        ave_anxiety_hiphop2 = cleaned_data_hiphop2["Anxiety"].mean()
        ave_dep_hiphop2 = cleaned_data_hiphop2["Depression"].mean()
        ave_insom_hiphop2 = cleaned_data_hiphop2["Insomnia"].mean()
        ave_ocd_hiphop2 = cleaned_data_hiphop2["OCD"].mean()
        
        ave_anxiety_hiphop3 = cleaned_data_hiphop3["Anxiety"].mean()
        ave_dep_hiphop3 = cleaned_data_hiphop3["Depression"].mean()
        ave_insom_hiphop3 = cleaned_data_hiphop3["Insomnia"].mean()
        ave_ocd_hiphop3 = cleaned_data_hiphop3["OCD"].mean()
        
        ave_anxiety_hiphop4 = cleaned_data_hiphop4["Anxiety"].mean()
        ave_dep_hiphop4 = cleaned_data_hiphop4["Depression"].mean()
        ave_insom_hiphop4 = cleaned_data_hiphop4["Insomnia"].mean()
        ave_ocd_hiphop4 = cleaned_data_hiphop4["OCD"].mean()
        
        
        
        
        #making Pop subsets based on frequency
        cleaned_data_pop1 = cleaned_data[cleaned_data["Frequency [Pop]"] == 1]
        cleaned_data_pop2 = cleaned_data[cleaned_data["Frequency [Pop]"] == 2]
        cleaned_data_pop3 = cleaned_data[cleaned_data["Frequency [Pop]"] == 3]
        cleaned_data_pop4 = cleaned_data[cleaned_data["Frequency [Pop]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_pop1 = cleaned_data_pop1["Anxiety"].mean()
        ave_dep_pop1 = cleaned_data_pop1["Depression"].mean()
        ave_insom_pop1 = cleaned_data_pop1["Insomnia"].mean()
        ave_ocd_pop1 = cleaned_data_pop1["OCD"].mean()
        
        ave_anxiety_pop2 = cleaned_data_pop2["Anxiety"].mean()
        ave_dep_pop2 = cleaned_data_pop2["Depression"].mean()
        ave_insom_pop2 = cleaned_data_pop2["Insomnia"].mean()
        ave_ocd_pop2 = cleaned_data_pop2["OCD"].mean()
        
        ave_anxiety_pop3 = cleaned_data_pop3["Anxiety"].mean()
        ave_dep_pop3 = cleaned_data_pop3["Depression"].mean()
        ave_insom_pop3 = cleaned_data_pop3["Insomnia"].mean()
        ave_ocd_pop3 = cleaned_data_pop3["OCD"].mean()
        
        ave_anxiety_pop4 = cleaned_data_pop4["Anxiety"].mean()
        ave_dep_pop4 = cleaned_data_pop4["Depression"].mean()
        ave_insom_pop4 = cleaned_data_pop4["Insomnia"].mean()
        ave_ocd_pop4 = cleaned_data_pop4["OCD"].mean()
        
        
        
        
        #making Rap subsets based on frequency
        cleaned_data_rap1 = cleaned_data[cleaned_data["Frequency [Rap]"] == 1]
        cleaned_data_rap2 = cleaned_data[cleaned_data["Frequency [Rap]"] == 2]
        cleaned_data_rap3 = cleaned_data[cleaned_data["Frequency [Rap]"] == 3]
        cleaned_data_rap4 = cleaned_data[cleaned_data["Frequency [Rap]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_rap1 = cleaned_data_rap1["Anxiety"].mean()
        ave_dep_rap1 = cleaned_data_rap1["Depression"].mean()
        ave_insom_rap1 = cleaned_data_rap1["Insomnia"].mean()
        ave_ocd_rap1 = cleaned_data_rap1["OCD"].mean()
        
        ave_anxiety_rap2 = cleaned_data_rap2["Anxiety"].mean()
        ave_dep_rap2 = cleaned_data_rap2["Depression"].mean()
        ave_insom_rap2 = cleaned_data_rap2["Insomnia"].mean()
        ave_ocd_rap2 = cleaned_data_rap2["OCD"].mean()
        
        ave_anxiety_rap3 = cleaned_data_rap3["Anxiety"].mean()
        ave_dep_rap3 = cleaned_data_rap3["Depression"].mean()
        ave_insom_rap3 = cleaned_data_rap3["Insomnia"].mean()
        ave_ocd_rap3 = cleaned_data_rap3["OCD"].mean()
        
        ave_anxiety_rap4 = cleaned_data_rap4["Anxiety"].mean()
        ave_dep_rap4 = cleaned_data_rap4["Depression"].mean()
        ave_insom_rap4 = cleaned_data_rap4["Insomnia"].mean()
        ave_ocd_rap4 = cleaned_data_rap4["OCD"].mean()
        
        
        
        #making Classical subsets based on frequency
        cleaned_data_classical1 = cleaned_data[cleaned_data["Frequency [Classical]"] == 1]
        cleaned_data_classical2 = cleaned_data[cleaned_data["Frequency [Classical]"] == 2]
        cleaned_data_classical3 = cleaned_data[cleaned_data["Frequency [Classical]"] == 3]
        cleaned_data_classical4 = cleaned_data[cleaned_data["Frequency [Classical]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_classical1 = cleaned_data_classical1["Anxiety"].mean()
        ave_dep_classical1 = cleaned_data_classical1["Depression"].mean()
        ave_insom_classical1 = cleaned_data_classical1["Insomnia"].mean()
        ave_ocd_classical1 = cleaned_data_classical1["OCD"].mean()
        
        ave_anxiety_classical2 = cleaned_data_classical2["Anxiety"].mean()
        ave_dep_classical2 = cleaned_data_classical2["Depression"].mean()
        ave_insom_classical2 = cleaned_data_classical2["Insomnia"].mean()
        ave_ocd_classical2 = cleaned_data_classical2["OCD"].mean()
        
        ave_anxiety_classical3 = cleaned_data_classical3["Anxiety"].mean()
        ave_dep_classical3 = cleaned_data_classical3["Depression"].mean()
        ave_insom_classical3 = cleaned_data_classical3["Insomnia"].mean()
        ave_ocd_classical3 = cleaned_data_classical3["OCD"].mean()
        
        ave_anxiety_classical4 = cleaned_data_classical4["Anxiety"].mean()
        ave_dep_classical4 = cleaned_data_classical4["Depression"].mean()
        ave_insom_classical4 = cleaned_data_classical4["Insomnia"].mean()
        ave_ocd_classical4 = cleaned_data_classical4["OCD"].mean()
        
        
        
        #making Metal subsets based on frequency
        cleaned_data_metal1 = cleaned_data[cleaned_data["Frequency [Metal]"] == 1]
        cleaned_data_metal2 = cleaned_data[cleaned_data["Frequency [Metal]"] == 2]
        cleaned_data_metal3 = cleaned_data[cleaned_data["Frequency [Metal]"] == 3]
        cleaned_data_metal4 = cleaned_data[cleaned_data["Frequency [Metal]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_metal1 = cleaned_data_metal1["Anxiety"].mean()
        ave_dep_metal1 = cleaned_data_metal1["Depression"].mean()
        ave_insom_metal1 = cleaned_data_metal1["Insomnia"].mean()
        ave_ocd_metal1 = cleaned_data_metal1["OCD"].mean()
        
        ave_anxiety_metal2 = cleaned_data_metal2["Anxiety"].mean()
        ave_dep_metal2 = cleaned_data_metal2["Depression"].mean()
        ave_insom_metal2 = cleaned_data_metal2["Insomnia"].mean()
        ave_ocd_metal2 = cleaned_data_metal2["OCD"].mean()
        
        ave_anxiety_metal3 = cleaned_data_metal3["Anxiety"].mean()
        ave_dep_metal3 = cleaned_data_metal3["Depression"].mean()
        ave_insom_metal3 = cleaned_data_metal3["Insomnia"].mean()
        ave_ocd_metal3 = cleaned_data_metal3["OCD"].mean()
        
        ave_anxiety_metal4 = cleaned_data_metal4["Anxiety"].mean()
        ave_dep_metal4 = cleaned_data_metal4["Depression"].mean()
        ave_insom_metal4 = cleaned_data_metal4["Insomnia"].mean()
        ave_ocd_metal4 = cleaned_data_metal4["OCD"].mean()
        
        
        
        
        #making Folk subsets based on frequency
        cleaned_data_folk1 = cleaned_data[cleaned_data["Frequency [Folk]"] == 1]
        cleaned_data_folk2 = cleaned_data[cleaned_data["Frequency [Folk]"] == 2]
        cleaned_data_folk3 = cleaned_data[cleaned_data["Frequency [Folk]"] == 3]
        cleaned_data_folk4 = cleaned_data[cleaned_data["Frequency [Folk]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_folk1 = cleaned_data_folk1["Anxiety"].mean()
        ave_dep_folk1 = cleaned_data_folk1["Depression"].mean()
        ave_insom_folk1 = cleaned_data_folk1["Insomnia"].mean()
        ave_ocd_folk1 = cleaned_data_folk1["OCD"].mean()
        
        ave_anxiety_folk2 = cleaned_data_folk2["Anxiety"].mean()
        ave_dep_folk2 = cleaned_data_folk2["Depression"].mean()
        ave_insom_folk2 = cleaned_data_folk2["Insomnia"].mean()
        ave_ocd_folk2 = cleaned_data_folk2["OCD"].mean()
        
        ave_anxiety_folk3 = cleaned_data_folk3["Anxiety"].mean()
        ave_dep_folk3 = cleaned_data_folk3["Depression"].mean()
        ave_insom_folk3 = cleaned_data_folk3["Insomnia"].mean()
        ave_ocd_folk3 = cleaned_data_folk3["OCD"].mean()
        
        ave_anxiety_folk4 = cleaned_data_folk4["Anxiety"].mean()
        ave_dep_folk4 = cleaned_data_folk4["Depression"].mean()
        ave_insom_folk4 = cleaned_data_folk4["Insomnia"].mean()
        ave_ocd_folk4 = cleaned_data_folk4["OCD"].mean()
        
        
        
        
        
        #making Lofi subsets based on frequency
        cleaned_data_lofi1 = cleaned_data[cleaned_data["Frequency [Lofi]"] == 1]
        cleaned_data_lofi2 = cleaned_data[cleaned_data["Frequency [Lofi]"] == 2]
        cleaned_data_lofi3 = cleaned_data[cleaned_data["Frequency [Lofi]"] == 3]
        cleaned_data_lofi4 = cleaned_data[cleaned_data["Frequency [Lofi]"] == 4]
        
        #Now get the average MH scores for each frequency
        ave_anxiety_lofi1 = cleaned_data_lofi1["Anxiety"].mean()
        ave_dep_lofi1 = cleaned_data_lofi1["Depression"].mean()
        ave_insom_lofi1 = cleaned_data_lofi1["Insomnia"].mean()
        ave_ocd_lofi1 = cleaned_data_lofi1["OCD"].mean()
        
        ave_anxiety_lofi2 = cleaned_data_lofi2["Anxiety"].mean()
        ave_dep_lofi2 = cleaned_data_lofi2["Depression"].mean()
        ave_insom_lofi2 = cleaned_data_lofi2["Insomnia"].mean()
        ave_ocd_lofi2 = cleaned_data_lofi2["OCD"].mean()
        
        ave_anxiety_lofi3 = cleaned_data_lofi3["Anxiety"].mean()
        ave_dep_lofi3 = cleaned_data_lofi3["Depression"].mean()
        ave_insom_lofi3 = cleaned_data_lofi3["Insomnia"].mean()
        ave_ocd_lofi3 = cleaned_data_lofi3["OCD"].mean()
        
        ave_anxiety_lofi4 = cleaned_data_lofi4["Anxiety"].mean()
        ave_dep_lofi4 = cleaned_data_lofi4["Depression"].mean()
        ave_insom_lofi4 = cleaned_data_lofi4["Insomnia"].mean()
        ave_ocd_lofi4 = cleaned_data_lofi4["OCD"].mean()
        
        
        
        
        #making Gospel subsets based on frequency
        cleaned_data_gospel1 = cleaned_data[cleaned_data["Frequency [Gospel]"] == 1]
        cleaned_data_gospel2 = cleaned_data[cleaned_data["Frequency [Gospel]"] == 2]
        cleaned_data_gospel3 = cleaned_data[cleaned_data["Frequency [Gospel]"] == 3]
        cleaned_data_gospel4 = cleaned_data[cleaned_data["Frequency [Gospel]"] == 4]
        
        #now get the average MH scores for each frequency
        ave_anxiety_gospel1 = cleaned_data_gospel1["Anxiety"].mean()
        ave_dep_gospel1 = cleaned_data_gospel1["Depression"].mean()
        ave_insom_gospel1 = cleaned_data_gospel1["Insomnia"].mean()
        ave_ocd_gospel1 = cleaned_data_gospel1["OCD"].mean()
        
        ave_anxiety_gospel2 = cleaned_data_gospel2["Anxiety"].mean()
        ave_dep_gospel2 = cleaned_data_gospel2["Depression"].mean()
        ave_insom_gospel2 = cleaned_data_gospel2["Insomnia"].mean()
        ave_ocd_gospel2 = cleaned_data_gospel2["OCD"].mean()
        
        ave_anxiety_gospel3 = cleaned_data_gospel3["Anxiety"].mean()
        ave_dep_gospel3 = cleaned_data_gospel3["Depression"].mean()
        ave_insom_gospel3 = cleaned_data_gospel3["Insomnia"].mean()
        ave_ocd_gospel3 = cleaned_data_gospel3["OCD"].mean()
        
        ave_anxiety_gospel4 = cleaned_data_gospel4["Anxiety"].mean()
        ave_dep_gospel4 = cleaned_data_gospel4["Depression"].mean()
        ave_insom_gospel4 = cleaned_data_gospel4["Insomnia"].mean()
        ave_ocd_gospel4 = cleaned_data_gospel4["OCD"].mean()
        
        #create a dataframe for these values
        index = ["Classical", "Country", "EDM", "Folk", "Gospel", "Hip hop", "Jazz", "K pop", "Latin", "Lofi", 
                 "Metal", "Pop", "R&B", "Rap", "Rock", "Video game music"]
        columns = ["Anxiety", "Depression", "Insomnia", "OCD"]
        
        mh_by_genre = pd.DataFrame(index=index, columns=columns)
        
        
        #add values
        average_anxiety = [ave_anxiety_classical4, ave_anxiety_country4, ave_anxiety_edm4,  ave_anxiety_folk4, ave_anxiety_gospel4, ave_anxiety_hiphop4, 
            ave_anxiety_jazz4, ave_anxiety_kpop4, ave_anxiety_latin4, ave_anxiety_lofi4, ave_anxiety_metal4, ave_anxiety_pop4, ave_anxiety_rnb4, 
            ave_anxiety_rap4, ave_anxiety_rock4, ave_anxiety_vgm4]
        
        average_depression = [ave_dep_classical4, ave_dep_country4, ave_dep_edm4, ave_dep_folk4, ave_dep_gospel4, ave_dep_hiphop4, ave_dep_jazz4, 
                ave_dep_kpop4, ave_dep_latin4, ave_dep_lofi4, ave_dep_metal4, ave_dep_pop4, ave_dep_rnb4, ave_dep_rap4, ave_dep_rock4, ave_dep_vgm4]
        
        average_ocd = [ave_ocd_classical4, ave_ocd_country4, ave_ocd_edm4, ave_ocd_folk4, ave_ocd_gospel4, ave_ocd_hiphop4, ave_ocd_jazz4, ave_ocd_kpop4, 
            ave_ocd_latin4, ave_ocd_lofi4, ave_ocd_metal4, ave_ocd_pop4, ave_ocd_rnb4, ave_ocd_rap4, ave_ocd_rock4, ave_ocd_vgm4]
        
        average_insomnia = [ave_insom_classical4, ave_insom_country4, ave_insom_edm4, ave_insom_folk4, ave_insom_gospel4, ave_insom_hiphop4, ave_insom_jazz4, 
                ave_insom_kpop4, ave_insom_latin4, ave_insom_lofi4, ave_insom_metal4, ave_insom_pop4, ave_insom_rnb4, ave_insom_rap4, ave_insom_rock4, 
                ave_insom_vgm4]
        
        mh_by_genre["Anxiety"] = average_anxiety
        mh_by_genre["Depression"] = average_depression
        mh_by_genre["Insomnia"] = average_insomnia
        mh_by_genre["OCD"] = average_ocd
        
        mh_by_genre["Dep Effect"] = np.where(mh_by_genre["Depression"] >= 5, 1, 0)
        mh_by_genre["Anx Effect"] = np.where(mh_by_genre["Anxiety"] >= 5, 1, 0)
        mh_by_genre["Ins Effect"] = np.where(mh_by_genre["Insomnia"] >= 5, 1, 0)
        mh_by_genre["OCD Effect"] = np.where(mh_by_genre["OCD"] >= 5, 1, 0)
        
        #This dataframe will be used to connect this analysis with the second dataset.
        effect_df = mh_by_genre.reset_index(names='Genre')
        effect_df.drop(["Anxiety", "Depression", "OCD", "Insomnia"], axis=1)
        
        
        cleaned_data = cleaned_data.copy()
        #I will say the max they could realistically listen to is 16 hrs
        cleaned_data = cleaned_data[(cleaned_data["Hours per day"] < 16)]
        #deleted 6 rows
        
        #take away age outliers 
        cleaned_data = cleaned_data[(cleaned_data["Age"] > 18) & (cleaned_data["Age"] < 64)]
        
        #get the median frequency
        values = cleaned_data["Fav genre"].value_counts()
        #values.median()
        
        #make the changes to rock
        num = 21
        length = len(cleaned_data[cleaned_data["Fav genre"] == "Rock"])
        drop_these_many = length - num
        random_idx = np.random.choice(cleaned_data[cleaned_data["Fav genre"] == "Rock"].index, drop_these_many, replace=False)
        #drop the selected indices from the DataFrame
        cleaned_data = cleaned_data.drop(random_idx)
        
        #make the changes to metal
        num = 21
        length = len(cleaned_data[cleaned_data["Fav genre"] == "Metal"])
        drop_these_many = length - num
        random_idx = np.random.choice(cleaned_data[cleaned_data["Fav genre"] == "Metal"].index, drop_these_many, replace=False)
        #drop the selected indices from the DataFrame
        cleaned_data = cleaned_data.drop(random_idx)
        
        #make the changes to pop
        num = 21
        length = len(cleaned_data[cleaned_data["Fav genre"] == "Pop"])
        drop_these_many = length - num
        random_idx = np.random.choice(cleaned_data[cleaned_data["Fav genre"] == "Pop"].index, drop_these_many, replace=False)
        #drop the selected indices from the DataFrame
        cleaned_data = cleaned_data.drop(random_idx)
        
        
        ##############balance anxiety 
        #reset index
        cleaned_data.reset_index(drop=True, inplace=True)
        cleaned_data["Anxiety_category"] = np.where(cleaned_data["Anxiety"] >= 5, 1, 0)
        X = cleaned_data.drop(["Anxiety", "Anxiety_category"], axis=1)  
        y = cleaned_data["Anxiety_category"] 
        
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        
        print(f"Before Undersampling: \n{y.value_counts()}")
        print(f"After Undersampling: \n{y_resampled.value_counts()}")
        
        resampled_indices = rus.sample_indices_
        
        anxiety_resampled = cleaned_data.loc[resampled_indices, "Anxiety"]
        
        cleaned_data = X_resampled.copy()  
        cleaned_data["Anxiety"] = anxiety_resampled.values  
        
        #reset index
        cleaned_data.reset_index(drop=True, inplace=True)
        
        
        ######now balance depression
        cleaned_data["Depression_category"] = np.where(cleaned_data["Depression"] >= 5, 1, 0)
        X = cleaned_data.drop(["Depression", "Depression_category"], axis=1)  
        y = cleaned_data["Depression_category"] 
        
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        
        print(f"Before Undersampling: \n{y.value_counts()}")
        print(f"After Undersampling: \n{y_resampled.value_counts()}")
        
        resampled_indices = rus.sample_indices_
        
        depression_resampled = cleaned_data.loc[resampled_indices, "Depression"]
        
        cleaned_data = X_resampled.copy()  
        cleaned_data["Depression"] = depression_resampled.values  
        
        #reset index
        cleaned_data.reset_index(drop=True, inplace=True)
        
        
        
        
        ###### adding dataset #2 edits from the Explore Data section so we can use dataset #2 in Get Recommendations
        
        songs = pd.read_csv("songs_normalize.csv")
        
        songs = songs[songs["explicit"] == False]
        
        
        #Some songs are categorized as multiple genres. Let's split that up so each song is listed once per genre that it classifies as. This will create duplicates. For example, I want a pop-rock song to be recommened for pop and rock recommedations.")
        songs["genre"] = songs["genre"].str.split(",")
        
        #explode the dataset so each genre gets its own row
        ######explode() expands the list of genres so each genre has its own row, duplicating other information about the song.
        #####reset_index(drop=True)  resets the index to keep things neat after exploding.
        songs_expanded = songs.explode("genre").reset_index(drop=True)
        
        
        
        #make sure genres are consistent
        #songs_expanded["genre"]==[" Folk/Acoustic"].replace("Folk/Acoustic")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" Folk/Acoustic", "Folk/Acoustic")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" Dance/Electronic", "Dance/Electronic")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" pop", "pop")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" hip hop", "hip hop")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" country", "country")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" metal", "metal")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" R&B", "R&B")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" rock", "rock")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" easy listening", "easy listening")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" latin", "latin")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" classical", "classical")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" blues", "blues")
        songs_expanded["genre"] = songs_expanded["genre"].replace(" jazz", "Jazz")
        
        #changing capitalization and wording
        songs_expanded["genre"] = songs_expanded["genre"].replace("pop", "Pop")
        songs_expanded["genre"] = songs_expanded["genre"].replace("rock", "Rock")
        songs_expanded["genre"] = songs_expanded["genre"].replace("country", "Country")
        songs_expanded["genre"] = songs_expanded["genre"].replace("metal", "Metal")
        songs_expanded["genre"] = songs_expanded["genre"].replace("hip hop", "Hip hop")
        songs_expanded["genre"] = songs_expanded["genre"].replace("Dance/Electronic", "EDM")
        songs_expanded["genre"] = songs_expanded["genre"].replace("Folk/Acoustic", "Folk")
        songs_expanded["genre"] = songs_expanded["genre"].replace("latin", "Latin")
        songs_expanded["genre"] = songs_expanded["genre"].replace("jazz", "Jazz")
        songs_expanded["genre"] = songs_expanded["genre"].replace("classical", "Classical")
        
        
        
        
        songs_expanded.reset_index(drop=True, inplace=True)
        songs_expanded["valence_category"] = np.where(songs_expanded["valence"] >= 0.5, 1, 0)
        #separate features (X) and target (y)
        #drop the continuous feature and the categorical version we just made
        X = songs_expanded.drop(["valence", "valence_category"], axis=1)  # Keep only non-target features
        #look at the categorical version as the target 
        y = songs_expanded["valence_category"]  # Target variable
        
        #apply RandomUnderSampler
        #initialize it
        rus = RandomUnderSampler(random_state=42)
        #apply it to X and y and store the changed versions
        X_resampled, y_resampled = rus.fit_resample(X, y)
        
        #print the differences so we can see that the package did its job
        print(f"Before Undersampling: \n{y.value_counts()}")
        print(f"After Undersampling: \n{y_resampled.value_counts()}")
        
        #get the indices of the resampled data
        resampled_indices = rus.sample_indices_
        
        #use the indices to retrieve the original continuous valence values
        valence_resampled = songs_expanded.loc[resampled_indices, "valence"]
        
        #create the final resampled dataset with original continuous valence values
        songs_balanced = X_resampled.copy()  #start with resampled features
        songs_balanced["valence"] = valence_resampled.values  #add back continuous valence
        
        songs_balanced["valence_category"] = np.where(songs_balanced["valence"] >= 0.5, 1, 0)
        
        
        ####### adding code where I merge the datasets
        
        #delete valence_category 
        #songs_balanced = songs_balanced.drop("valence_category", axis = 1)
        
        
        #First I have to make sure the genre columns are capitalized the same
        songs_balanced.rename(columns={'genre': 'Genre'}, inplace=True)
        
        #merge them
        merged_df = pd.merge(songs_balanced, effect_df, on='Genre', how='left')
        
        
        
        
        ####################### done replicating the filtering done above
        
        
        
        #I'm editing the code below so that it uses merged_df and not mh_by_genre
        ###mood_increase_genres = mh_by_genre[mh_by_genre["Dep Effect"] == 0]
        ###mood_decrease_genres = mh_by_genre[mh_by_genre["Dep Effect"] == 1]
        ###increase_recommendations = mood_increase_genres.index
        ###decrease_recommendations = mood_decrease_genres.index
    
        
        
        
        feel_happy = merged_df[merged_df["Dep Effect"] == 0]
        feel_sad = merged_df[merged_df["Dep Effect"] == 1]
        #select only rows where Anxiety score is less 5/10
        feel_calm = merged_df[merged_df["Anx Effect"] == 0]
        #get average dance score 
        avg_danceability = merged_df["danceability"].mean()
        high_danceability_df = merged_df[merged_df["danceability"] >= avg_danceability]
        low_danceability_df =  merged_df[merged_df["danceability"] < avg_danceability]
        
        #feel_happy_recs = feel_happy[["artist", "song", "year"]]
        feel_happy_recs = feel_happy
        #feel_sad_recs = feel_sad[["artist", "song", "year"]]
        feel_sad_recs = feel_sad
        #feel_calm_recs = feel_calm[["artist", "song", "year"]]
        feel_calm_recs = feel_calm
        #feel_dancey_recs = high_danceability_df[["artist", "song", "year"]]
        feel_dancey_recs = high_danceability_df

        st.markdown("Please be advised that all recommendations are based on self-reported mental health scores of listeners. Since these recommendations are based on the correlations between listening preferences and mental health, they are not proven to *cause* changes in mood, but rather are *associated* with changes in mood.")
        st.markdown("Please choose a listening goal to recieve aligned genre recommendations.")
        #dropdown menu
        categories = ["Happy"]
        selected_category = st.selectbox("Choose a listening goal:", categories)
        
        if selected_category == "Happy":
        
            #display the selected category
            st.write(f"You selected: {selected_category}")
        
            st.markdown("Here are your recommended genres based averge mental health metrics by genre, demonstrated in the plots below:")
            #st.write(feel_happy_recs)
            st.write(feel_happy["Genre"].unique())
            st.markdown("Here are song suggestions from those genres.")
            st.markdown("App performance tip: Sad songs are included in this list based on lyrics - for example, What Hurts The Most by Rascal Flatts. This is where feature selection is important. Valence may be more reliable than danceability and energy. For example, 'What's Hurts The Most' by Rascal Flats has high danceability and energy, but low valence. With this, a combination of these features will be used to improve these recommendations, with a focus placed on valence.")
            st.write(feel_happy)
        
            #include a visualization
            # Set the plot style
            sns.set(style="whitegrid")
        
            # Create a figure and axis
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
            # Create a bar plot for each mental health measure
            sns.barplot(x=mh_by_genre.index, y='Anxiety', data=mh_by_genre, ax=axes[0, 0], palette='viridis')
            axes[0, 0].set_title('Anxiety Levels by Genre')
            axes[0, 0].set_ylabel('Anxiety Level')
            axes[0, 0].tick_params(axis='x', rotation=45)
                
            sns.barplot(x=mh_by_genre.index, y='Depression', data=mh_by_genre, ax=axes[0, 1], palette='viridis')
            axes[0, 1].set_title('Depression Levels by Genre')
            axes[0, 1].set_ylabel('Depression Level')
            axes[0, 1].tick_params(axis='x', rotation=45)
                
            sns.barplot(x=mh_by_genre.index, y='Insomnia', data=mh_by_genre, ax=axes[1, 0], palette='viridis')
            axes[1, 0].set_title('Insomnia Levels by Genre')
            axes[1, 0].set_ylabel('Insomnia Level')
            axes[1, 0].tick_params(axis='x', rotation=45)
                
            sns.barplot(x=mh_by_genre.index, y='OCD', data=mh_by_genre, ax=axes[1, 1], palette='viridis')
            axes[1, 1].set_title('OCD Levels by Genre')
            axes[1, 1].set_ylabel('OCD Level')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
            plt.tight_layout()
                
            #show the plot
            st.pyplot(plt)

            st.markdown("This plot demonstrates the need to enhance recommendations further. There's too much variety in each genre to base recs off genre alone.")

            plt.close(fig)
            
            # Melt the DataFrame to a long format
            happy_features = feel_happy[["valence", "danceability", "energy"]]
            df_long = happy_features.melt(var_name="Feature", value_name="Value")
            
            # Create the scatterplot
            fig = px.scatter(
                df_long, 
                x=df_long.index,  # The index as x-axis
                y="Value", 
                color="Feature",
                title="Scatterplot of Features",
                labels={"index": "Row Index", "Value": "Feature Value"}
            )
            
            # Display the plot in Streamlit
            import streamlit as st
            st.plotly_chart(fig)



            st.markdown("Here are your improved recommendations (songs with above average valence); songs within the recommended genres that best fit your listening goals based on the stats demonstrated in the plots below:")
            #st.write(feel_happy_recs)
            avg_valence = merged_df["valence"].mean()
            above_avg_valence = feel_happy[feel_happy["valence"] >= avg_valence]
            st.write(above_avg_valence)

            st.markdown("Variability has decreased. These recommendations will include less sad songs since they are limited to those songs in the recommended genres with above average valence.")


            #plt.close(fig)
            
            # Melt the DataFrame to a long format
            df_long = above_avg_valence[["valence", "danceability", "energy"]]
            df_long = df_long.melt(var_name="Feature", value_name="Value")
            
            # Create the scatterplot
            fig = px.scatter(
                df_long, 
                x=df_long.index,  # The index as x-axis
                y="Value", 
                color="Feature",
                title="Scatterplot of Features",
                labels={"index": "Row Index", "Value": "Feature Value"}
            )

            # Update the y-axis range and ticks
            fig.update_yaxes(range=[0, 1], tick0=0, dtick=0.2)
            
            # Display the plot in Streamlit
            import streamlit as st
            st.plotly_chart(fig)


            st.subheader("Final feature combinations used to filter song recommendations:")
            st. write("Happy | Genres with a Dep Effect of 0 filtered by above average valence (over the whole dataset).")
            st. write("Sad | Genres with a Dep Effect of 1 filtered by below average valence, danceability, and energy (over the whole dataset).")
            st. write("Calm | Genres with an Anx Effect of 0 filtered by valence, energy, and danceability scores below or equal to 0.6 (over the whole dataset) to identify the most neutral songs. Filtering by the average feature scores resulted in high feature distributions.")
            st. write("Dance| Genres with a Dep Effect of 0 filtered by above average danceability (over the whole dataset).")
        
        
        

################################################################################# this begins the Final Get Recommendations Page     
#if selected_category == "Get Recommendations":
if option == "Get Recommendations":

    #add a banner
    st.markdown(
        """
        <style>
            .banner {
                background: linear-gradient(to right, #6a11cb, #2575fc);
                color: white;
                padding: 20px;
                text-align: center;
                font-size: 30px;
                border-radius: 10px;
            }
        </style>
        <div class="banner">
            🎵 <b>Tunes By Mood</b> 🎶 <br> <i>Feel Better, One Song at a Time</i>
        </div>
        """,
        unsafe_allow_html=True
    )

    #title of the app
    st.title("Welcome To Tunes By Mood: A Music Therapy App Designed For You")
    st.markdown("These recommendations are based on insights into how music listening correlates with mood changes, offering supportive suggestions tailored to you.")
    st.subheader("Find Your Tunes Now")
    
    
    ########################repeating data edits so this dropdown option can use the same updated data
    
    
    #load the Data
    mxmh_survey_results = pd.read_csv("mxmh_survey_results.csv")
    
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
    
    
    cleaned_data = mxmh_survey_results.copy()
    #I will say the max they could realistically listen to is 16 hrs
    cleaned_data = cleaned_data[(cleaned_data["Hours per day"] < 16)]
    #deleted 6 rows
    
    #take away age outliers 
    cleaned_data = cleaned_data[(cleaned_data["Age"] > 18) & (cleaned_data["Age"] < 64)]

    #take away MH outliers             
    cleaned_data = cleaned_data[
    (cleaned_data["Anxiety"] >= 0) & (cleaned_data["Anxiety"] <= 10) &
    (cleaned_data["Depression"] >= 0) & (cleaned_data["Depression"] <= 10) &
    (cleaned_data["OCD"] >= 0) & (cleaned_data["OCD"] <= 10) &
    (cleaned_data["Insomnia"] >= 0) & (cleaned_data["Insomnia"] <= 10)
    ]
    
    #recode frequency genre
    
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



    #################adding in the edits that were out of place below
    # cleaned_data = cleaned_data.copy()
    # #I will say the max they could realistically listen to is 16 hrs
    # cleaned_data = cleaned_data[(cleaned_data["Hours per day"] < 16)]
    # #deleted 6 rows
    
    # #take away age outliers 
    # cleaned_data = cleaned_data[(cleaned_data["Age"] > 18) & (cleaned_data["Age"] < 64)]
    
    # #get the median frequency
    # values = cleaned_data["Fav genre"].value_counts()
    # #values.median()
    
    #make the changes to rock
    num = 21
    length = len(cleaned_data[cleaned_data["Fav genre"] == "Rock"])
    drop_these_many = length - num
    random_idx = np.random.choice(cleaned_data[cleaned_data["Fav genre"] == "Rock"].index, drop_these_many, replace=False)
    #drop the selected indices from the DataFrame
    cleaned_data = cleaned_data.drop(random_idx)
    
    #make the changes to metal
    num = 21
    length = len(cleaned_data[cleaned_data["Fav genre"] == "Metal"])
    drop_these_many = length - num
    random_idx = np.random.choice(cleaned_data[cleaned_data["Fav genre"] == "Metal"].index, drop_these_many, replace=False)
    #drop the selected indices from the DataFrame
    cleaned_data = cleaned_data.drop(random_idx)
    
    #make the changes to pop
    num = 21
    length = len(cleaned_data[cleaned_data["Fav genre"] == "Pop"])
    drop_these_many = length - num
    random_idx = np.random.choice(cleaned_data[cleaned_data["Fav genre"] == "Pop"].index, drop_these_many, replace=False)
    #drop the selected indices from the DataFrame
    cleaned_data = cleaned_data.drop(random_idx)
    
    
    ##############balance anxiety 
    #reset index
    cleaned_data.reset_index(drop=True, inplace=True)
    cleaned_data["Anxiety_category"] = np.where(cleaned_data["Anxiety"] >= 5, 1, 0)
    X = cleaned_data.drop(["Anxiety", "Anxiety_category"], axis=1)  
    y = cleaned_data["Anxiety_category"] 
    
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    print(f"Before Undersampling: \n{y.value_counts()}")
    print(f"After Undersampling: \n{y_resampled.value_counts()}")
    
    resampled_indices = rus.sample_indices_
    
    anxiety_resampled = cleaned_data.loc[resampled_indices, "Anxiety"]
    
    cleaned_data = X_resampled.copy()  
    cleaned_data["Anxiety"] = anxiety_resampled.values  
    
    #reset index
    cleaned_data.reset_index(drop=True, inplace=True)
    
    
    ######now balance depression
    cleaned_data["Depression_category"] = np.where(cleaned_data["Depression"] >= 5, 1, 0)
    X = cleaned_data.drop(["Depression", "Depression_category"], axis=1)  
    y = cleaned_data["Depression_category"] 
    
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    print(f"Before Undersampling: \n{y.value_counts()}")
    print(f"After Undersampling: \n{y_resampled.value_counts()}")
    
    resampled_indices = rus.sample_indices_
    
    depression_resampled = cleaned_data.loc[resampled_indices, "Depression"]
    
    cleaned_data = X_resampled.copy()  
    cleaned_data["Depression"] = depression_resampled.values  
    
    #reset index
    cleaned_data.reset_index(drop=True, inplace=True)
    
    ############# done moving the edits that were down below
    
    
    #make aged binned
    
    bins = [18, 25, 31, 36, 41, 46, 51, 58, 64]  
    labels = ['18-24', '25-30', '31-35', '36-40', '41-45', '46-50', '51-57', '58-64']  # Labels for the bins
    
    # Create the binned column
    cleaned_data['age_binned'] = pd.cut(cleaned_data['Age'], bins=bins, labels=labels, right=False)
    
    
    
    
    #feature engineering
    
    #making latin subsets based on frequency
    cleaned_data_latin1 = cleaned_data[cleaned_data["Frequency [Latin]"] == 1]
    cleaned_data_latin2 = cleaned_data[cleaned_data["Frequency [Latin]"] == 2]
    cleaned_data_latin3 = cleaned_data[cleaned_data["Frequency [Latin]"] == 3]
    cleaned_data_latin4 = cleaned_data[cleaned_data["Frequency [Latin]"] == 4]
    
    
    #now get the average MH scores for each frequency
    ave_anxiety_latin1 = cleaned_data_latin1["Anxiety"].mean()
    ave_dep_latin1 = cleaned_data_latin1["Depression"].mean()
    ave_insom_latin1 = cleaned_data_latin1["Insomnia"].mean()
    ave_ocd_latin1 = cleaned_data_latin1["OCD"].mean()
    
    ave_anxiety_latin2 = cleaned_data_latin2["Anxiety"].mean()
    ave_dep_latin2 = cleaned_data_latin2["Depression"].mean()
    ave_insom_latin2 = cleaned_data_latin2["Insomnia"].mean()
    ave_ocd_latin2 = cleaned_data_latin2["OCD"].mean()
    
    ave_anxiety_latin3 = cleaned_data_latin3["Anxiety"].mean()
    ave_dep_latin3 = cleaned_data_latin3["Depression"].mean()
    ave_insom_latin3 = cleaned_data_latin3["Insomnia"].mean()
    ave_ocd_latin3 = cleaned_data_latin3["OCD"].mean()
    
    ave_anxiety_latin4 = cleaned_data_latin4["Anxiety"].mean()
    ave_dep_latin4 = cleaned_data_latin4["Depression"].mean()
    ave_insom_latin4 = cleaned_data_latin4["Insomnia"].mean()
    ave_ocd_latin4 = cleaned_data_latin4["OCD"].mean()
    
    
    
    #making rock subsets based on frequency
    cleaned_data_rock1 = cleaned_data[cleaned_data["Frequency [Rock]"] == 1]
    cleaned_data_rock2 = cleaned_data[cleaned_data["Frequency [Rock]"] == 2]
    cleaned_data_rock3 = cleaned_data[cleaned_data["Frequency [Rock]"] == 3]
    cleaned_data_rock4 = cleaned_data[cleaned_data["Frequency [Rock]"] == 4]
    
    #now get the average MH scores for each frequency
    ave_anxiety_rock1 = cleaned_data_rock1["Anxiety"].mean()
    ave_dep_rock1 = cleaned_data_rock1["Depression"].mean()
    ave_insom_rock1 = cleaned_data_rock1["Insomnia"].mean()
    ave_ocd_rock1 = cleaned_data_rock1["OCD"].mean()
    
    ave_anxiety_rock2 = cleaned_data_rock2["Anxiety"].mean()
    ave_dep_rock2 = cleaned_data_rock2["Depression"].mean()
    ave_insom_rock2 = cleaned_data_rock2["Insomnia"].mean()
    ave_ocd_rock2 = cleaned_data_rock2["OCD"].mean()
    
    ave_anxiety_rock3 = cleaned_data_rock3["Anxiety"].mean()
    ave_dep_rock3 = cleaned_data_rock3["Depression"].mean()
    ave_insom_rock3 = cleaned_data_rock3["Insomnia"].mean()
    ave_ocd_rock3 = cleaned_data_rock3["OCD"].mean()
    
    ave_anxiety_rock4 = cleaned_data_rock4["Anxiety"].mean()
    ave_dep_rock4 = cleaned_data_rock4["Depression"].mean()
    ave_insom_rock4 = cleaned_data_rock4["Insomnia"].mean()
    ave_ocd_rock4 = cleaned_data_rock4["OCD"].mean()
    
    
    
    
    #making Video game music subsets based on frequency
    cleaned_data_vgm1 = cleaned_data[cleaned_data["Frequency [Video game music]"] == 1]
    cleaned_data_vgm2 = cleaned_data[cleaned_data["Frequency [Video game music]"] == 2]
    cleaned_data_vgm3 = cleaned_data[cleaned_data["Frequency [Video game music]"] == 3]
    cleaned_data_vgm4 = cleaned_data[cleaned_data["Frequency [Video game music]"] == 4]
    
    #now get the average MH scores for each frequency
    ave_anxiety_vgm1 = cleaned_data_vgm1["Anxiety"].mean()
    ave_dep_vgm1 = cleaned_data_vgm1["Depression"].mean()
    ave_insom_vgm1 = cleaned_data_vgm1["Insomnia"].mean()
    ave_ocd_vgm1 = cleaned_data_vgm1["OCD"].mean()
    
    ave_anxiety_vgm2 = cleaned_data_vgm2["Anxiety"].mean()
    ave_dep_vgm2 = cleaned_data_vgm2["Depression"].mean()
    ave_insom_vgm2 = cleaned_data_vgm2["Insomnia"].mean()
    ave_ocd_vgm2 = cleaned_data_vgm2["OCD"].mean()
    
    ave_anxiety_vgm3 = cleaned_data_vgm3["Anxiety"].mean()
    ave_dep_vgm3 = cleaned_data_vgm3["Depression"].mean()
    ave_insom_vgm3 = cleaned_data_vgm3["Insomnia"].mean()
    ave_ocd_vgm3 = cleaned_data_vgm3["OCD"].mean()
    
    ave_anxiety_vgm4 = cleaned_data_vgm4["Anxiety"].mean()
    ave_dep_vgm4 = cleaned_data_vgm4["Depression"].mean()
    ave_insom_vgm4 = cleaned_data_vgm4["Insomnia"].mean()
    ave_ocd_vgm4 = cleaned_data_vgm4["OCD"].mean()
    
    
    
    #making Jazz subsets based on frequency
    cleaned_data_jazz1 = cleaned_data[cleaned_data["Frequency [Jazz]"] == 1]
    cleaned_data_jazz2 = cleaned_data[cleaned_data["Frequency [Jazz]"] == 2]
    cleaned_data_jazz3 = cleaned_data[cleaned_data["Frequency [Jazz]"] == 3]
    cleaned_data_jazz4 = cleaned_data[cleaned_data["Frequency [Jazz]"] == 4]
    
    #now get the average MH scores for each frequency
    ave_anxiety_jazz1 = cleaned_data_jazz1["Anxiety"].mean()
    ave_dep_jazz1 = cleaned_data_jazz1["Depression"].mean()
    ave_insom_jazz1 = cleaned_data_jazz1["Insomnia"].mean()
    ave_ocd_jazz1 = cleaned_data_jazz1["OCD"].mean()
    
    ave_anxiety_jazz2 = cleaned_data_jazz2["Anxiety"].mean()
    ave_dep_jazz2 = cleaned_data_jazz2["Depression"].mean()
    ave_insom_jazz2 = cleaned_data_jazz2["Insomnia"].mean()
    ave_ocd_jazz2 = cleaned_data_jazz2["OCD"].mean()
    
    ave_anxiety_jazz3 = cleaned_data_jazz3["Anxiety"].mean()
    ave_dep_jazz3 = cleaned_data_jazz3["Depression"].mean()
    ave_insom_jazz3 = cleaned_data_jazz3["Insomnia"].mean()
    ave_ocd_jazz3 = cleaned_data_jazz3["OCD"].mean()
    
    ave_anxiety_jazz4 = cleaned_data_jazz4["Anxiety"].mean()
    ave_dep_jazz4 = cleaned_data_jazz4["Depression"].mean()
    ave_insom_jazz4 = cleaned_data_jazz4["Insomnia"].mean()
    ave_ocd_jazz4 = cleaned_data_jazz4["OCD"].mean()
    
    
    
    #making R&B subsets based on frequency
    cleaned_data_rnb1 = cleaned_data[cleaned_data["Frequency [R&B]"] == 1]
    cleaned_data_rnb2 = cleaned_data[cleaned_data["Frequency [R&B]"] == 2]
    cleaned_data_rnb3 = cleaned_data[cleaned_data["Frequency [R&B]"] == 3]
    cleaned_data_rnb4 = cleaned_data[cleaned_data["Frequency [R&B]"] == 4]
    
    #now get the average MH scores for each frequency
    ave_anxiety_rnb1 = cleaned_data_rnb1["Anxiety"].mean()
    ave_dep_rnb1 = cleaned_data_rnb1["Depression"].mean()
    ave_insom_rnb1 = cleaned_data_rnb1["Insomnia"].mean()
    ave_ocd_rnb1 = cleaned_data_rnb1["OCD"].mean()
    
    ave_anxiety_rnb2 = cleaned_data_rnb2["Anxiety"].mean()
    ave_dep_rnb2 = cleaned_data_rnb2["Depression"].mean()
    ave_insom_rnb2 = cleaned_data_rnb2["Insomnia"].mean()
    ave_ocd_rnb2 = cleaned_data_rnb2["OCD"].mean()
    
    ave_anxiety_rnb3 = cleaned_data_rnb3["Anxiety"].mean()
    ave_dep_rnb3 = cleaned_data_rnb3["Depression"].mean()
    ave_insom_rnb3 = cleaned_data_rnb3["Insomnia"].mean()
    ave_ocd_rnb3 = cleaned_data_rnb3["OCD"].mean()
    
    ave_anxiety_rnb4 = cleaned_data_rnb4["Anxiety"].mean()
    ave_dep_rnb4 = cleaned_data_rnb4["Depression"].mean()
    ave_insom_rnb4 = cleaned_data_rnb4["Insomnia"].mean()
    ave_ocd_rnb4 = cleaned_data_rnb4["OCD"].mean()
    
    
    
    #making K pop subsets based on frequency
    cleaned_data_kpop1 = cleaned_data[cleaned_data["Frequency [K pop]"] == 1]
    cleaned_data_kpop2 = cleaned_data[cleaned_data["Frequency [K pop]"] == 2]
    cleaned_data_kpop3 = cleaned_data[cleaned_data["Frequency [K pop]"] == 3]
    cleaned_data_kpop4 = cleaned_data[cleaned_data["Frequency [K pop]"] == 4]
    
    #now get the average MH scores for each frequency
    ave_anxiety_kpop1 = cleaned_data_kpop1["Anxiety"].mean()
    ave_dep_kpop1 = cleaned_data_kpop1["Depression"].mean()
    ave_insom_kpop1 = cleaned_data_kpop1["Insomnia"].mean()
    ave_ocd_kpop1 = cleaned_data_kpop1["OCD"].mean()
    
    ave_anxiety_kpop2 = cleaned_data_kpop2["Anxiety"].mean()
    ave_dep_kpop2 = cleaned_data_kpop2["Depression"].mean()
    ave_insom_kpop2 = cleaned_data_kpop2["Insomnia"].mean()
    ave_ocd_kpop2 = cleaned_data_kpop2["OCD"].mean()
    
    ave_anxiety_kpop3 = cleaned_data_kpop3["Anxiety"].mean()
    ave_dep_kpop3 = cleaned_data_kpop3["Depression"].mean()
    ave_insom_kpop3 = cleaned_data_kpop3["Insomnia"].mean()
    ave_ocd_kpop3 = cleaned_data_kpop3["OCD"].mean()
    
    ave_anxiety_kpop4 = cleaned_data_kpop4["Anxiety"].mean()
    ave_dep_kpop4 = cleaned_data_kpop4["Depression"].mean()
    ave_insom_kpop4 = cleaned_data_kpop4["Insomnia"].mean()
    ave_ocd_kpop4 = cleaned_data_kpop4["OCD"].mean()
    
    
    
    #making Country subsets based on frequency
    cleaned_data_country1 = cleaned_data[cleaned_data["Frequency [Country]"] == 1]
    cleaned_data_country2 = cleaned_data[cleaned_data["Frequency [Country]"] == 2]
    cleaned_data_country3 = cleaned_data[cleaned_data["Frequency [Country]"] == 3]
    cleaned_data_country4 = cleaned_data[cleaned_data["Frequency [Country]"] == 4]
    
    #now get the average MH scores for each frequency
    ave_anxiety_country1 = cleaned_data_country1["Anxiety"].mean()
    ave_dep_country1 = cleaned_data_country1["Depression"].mean()
    ave_insom_country1 = cleaned_data_country1["Insomnia"].mean()
    ave_ocd_country1 = cleaned_data_country1["OCD"].mean()
    
    ave_anxiety_country2 = cleaned_data_country2["Anxiety"].mean()
    ave_dep_country2 = cleaned_data_country2["Depression"].mean()
    ave_insom_country2 = cleaned_data_country2["Insomnia"].mean()
    ave_ocd_country2 = cleaned_data_country2["OCD"].mean()
    
    ave_anxiety_country3 = cleaned_data_country3["Anxiety"].mean()
    ave_dep_country3 = cleaned_data_country3["Depression"].mean()
    ave_insom_country3 = cleaned_data_country3["Insomnia"].mean()
    ave_ocd_country3 = cleaned_data_country3["OCD"].mean()
    
    ave_anxiety_country4 = cleaned_data_country4["Anxiety"].mean()
    ave_dep_country4 = cleaned_data_country4["Depression"].mean()
    ave_insom_country4 = cleaned_data_country4["Insomnia"].mean()
    ave_ocd_country4 = cleaned_data_country4["OCD"].mean()
    
    
    
    #making EDM subsets based on frequency
    cleaned_data_edm1 = cleaned_data[cleaned_data["Frequency [EDM]"] == 1]
    cleaned_data_edm2 = cleaned_data[cleaned_data["Frequency [EDM]"] == 2]
    cleaned_data_edm3 = cleaned_data[cleaned_data["Frequency [EDM]"] == 3]
    cleaned_data_edm4 = cleaned_data[cleaned_data["Frequency [EDM]"] == 4]
    
    #now get the average MH scores for each frequency
    ave_anxiety_edm1 = cleaned_data_edm1["Anxiety"].mean()
    ave_dep_edm1 = cleaned_data_edm1["Depression"].mean()
    ave_insom_edm1 = cleaned_data_edm1["Insomnia"].mean()
    ave_ocd_edm1 = cleaned_data_edm1["OCD"].mean()
    
    ave_anxiety_edm2 = cleaned_data_edm2["Anxiety"].mean()
    ave_dep_edm2 = cleaned_data_edm2["Depression"].mean()
    ave_insom_edm2 = cleaned_data_edm2["Insomnia"].mean()
    ave_ocd_edm2 = cleaned_data_edm2["OCD"].mean()
    
    ave_anxiety_edm3 = cleaned_data_edm3["Anxiety"].mean()
    ave_dep_edm3 = cleaned_data_edm3["Depression"].mean()
    ave_insom_edm3 = cleaned_data_edm3["Insomnia"].mean()
    ave_ocd_edm3 = cleaned_data_edm3["OCD"].mean()
    
    ave_anxiety_edm4 = cleaned_data_edm4["Anxiety"].mean()
    ave_dep_edm4 = cleaned_data_edm4["Depression"].mean()
    ave_insom_edm4 = cleaned_data_edm4["Insomnia"].mean()
    ave_ocd_edm4 = cleaned_data_edm4["OCD"].mean()
    
    
    
    #making Hip hop subsets based on frequency
    cleaned_data_hiphop1 = cleaned_data[cleaned_data["Frequency [Hip hop]"] == 1]
    cleaned_data_hiphop2 = cleaned_data[cleaned_data["Frequency [Hip hop]"] == 2]
    cleaned_data_hiphop3 = cleaned_data[cleaned_data["Frequency [Hip hop]"] == 3]
    cleaned_data_hiphop4 = cleaned_data[cleaned_data["Frequency [Hip hop]"] == 4]
    
    #now get the average MH scores for each frequency
    ave_anxiety_hiphop1 = cleaned_data_hiphop1["Anxiety"].mean()
    ave_dep_hiphop1 = cleaned_data_hiphop1["Depression"].mean()
    ave_insom_hiphop1 = cleaned_data_hiphop1["Insomnia"].mean()
    ave_ocd_hiphop1 = cleaned_data_hiphop1["OCD"].mean()
    
    ave_anxiety_hiphop2 = cleaned_data_hiphop2["Anxiety"].mean()
    ave_dep_hiphop2 = cleaned_data_hiphop2["Depression"].mean()
    ave_insom_hiphop2 = cleaned_data_hiphop2["Insomnia"].mean()
    ave_ocd_hiphop2 = cleaned_data_hiphop2["OCD"].mean()
    
    ave_anxiety_hiphop3 = cleaned_data_hiphop3["Anxiety"].mean()
    ave_dep_hiphop3 = cleaned_data_hiphop3["Depression"].mean()
    ave_insom_hiphop3 = cleaned_data_hiphop3["Insomnia"].mean()
    ave_ocd_hiphop3 = cleaned_data_hiphop3["OCD"].mean()
    
    ave_anxiety_hiphop4 = cleaned_data_hiphop4["Anxiety"].mean()
    ave_dep_hiphop4 = cleaned_data_hiphop4["Depression"].mean()
    ave_insom_hiphop4 = cleaned_data_hiphop4["Insomnia"].mean()
    ave_ocd_hiphop4 = cleaned_data_hiphop4["OCD"].mean()
    
    
    
    
    #making Pop subsets based on frequency
    cleaned_data_pop1 = cleaned_data[cleaned_data["Frequency [Pop]"] == 1]
    cleaned_data_pop2 = cleaned_data[cleaned_data["Frequency [Pop]"] == 2]
    cleaned_data_pop3 = cleaned_data[cleaned_data["Frequency [Pop]"] == 3]
    cleaned_data_pop4 = cleaned_data[cleaned_data["Frequency [Pop]"] == 4]
    
    #now get the average MH scores for each frequency
    ave_anxiety_pop1 = cleaned_data_pop1["Anxiety"].mean()
    ave_dep_pop1 = cleaned_data_pop1["Depression"].mean()
    ave_insom_pop1 = cleaned_data_pop1["Insomnia"].mean()
    ave_ocd_pop1 = cleaned_data_pop1["OCD"].mean()
    
    ave_anxiety_pop2 = cleaned_data_pop2["Anxiety"].mean()
    ave_dep_pop2 = cleaned_data_pop2["Depression"].mean()
    ave_insom_pop2 = cleaned_data_pop2["Insomnia"].mean()
    ave_ocd_pop2 = cleaned_data_pop2["OCD"].mean()
    
    ave_anxiety_pop3 = cleaned_data_pop3["Anxiety"].mean()
    ave_dep_pop3 = cleaned_data_pop3["Depression"].mean()
    ave_insom_pop3 = cleaned_data_pop3["Insomnia"].mean()
    ave_ocd_pop3 = cleaned_data_pop3["OCD"].mean()
    
    ave_anxiety_pop4 = cleaned_data_pop4["Anxiety"].mean()
    ave_dep_pop4 = cleaned_data_pop4["Depression"].mean()
    ave_insom_pop4 = cleaned_data_pop4["Insomnia"].mean()
    ave_ocd_pop4 = cleaned_data_pop4["OCD"].mean()
    
    
    
    
    #making Rap subsets based on frequency
    cleaned_data_rap1 = cleaned_data[cleaned_data["Frequency [Rap]"] == 1]
    cleaned_data_rap2 = cleaned_data[cleaned_data["Frequency [Rap]"] == 2]
    cleaned_data_rap3 = cleaned_data[cleaned_data["Frequency [Rap]"] == 3]
    cleaned_data_rap4 = cleaned_data[cleaned_data["Frequency [Rap]"] == 4]
    
    #now get the average MH scores for each frequency
    ave_anxiety_rap1 = cleaned_data_rap1["Anxiety"].mean()
    ave_dep_rap1 = cleaned_data_rap1["Depression"].mean()
    ave_insom_rap1 = cleaned_data_rap1["Insomnia"].mean()
    ave_ocd_rap1 = cleaned_data_rap1["OCD"].mean()
    
    ave_anxiety_rap2 = cleaned_data_rap2["Anxiety"].mean()
    ave_dep_rap2 = cleaned_data_rap2["Depression"].mean()
    ave_insom_rap2 = cleaned_data_rap2["Insomnia"].mean()
    ave_ocd_rap2 = cleaned_data_rap2["OCD"].mean()
    
    ave_anxiety_rap3 = cleaned_data_rap3["Anxiety"].mean()
    ave_dep_rap3 = cleaned_data_rap3["Depression"].mean()
    ave_insom_rap3 = cleaned_data_rap3["Insomnia"].mean()
    ave_ocd_rap3 = cleaned_data_rap3["OCD"].mean()
    
    ave_anxiety_rap4 = cleaned_data_rap4["Anxiety"].mean()
    ave_dep_rap4 = cleaned_data_rap4["Depression"].mean()
    ave_insom_rap4 = cleaned_data_rap4["Insomnia"].mean()
    ave_ocd_rap4 = cleaned_data_rap4["OCD"].mean()
    
    
    
    #making Classical subsets based on frequency
    cleaned_data_classical1 = cleaned_data[cleaned_data["Frequency [Classical]"] == 1]
    cleaned_data_classical2 = cleaned_data[cleaned_data["Frequency [Classical]"] == 2]
    cleaned_data_classical3 = cleaned_data[cleaned_data["Frequency [Classical]"] == 3]
    cleaned_data_classical4 = cleaned_data[cleaned_data["Frequency [Classical]"] == 4]
    
    #now get the average MH scores for each frequency
    ave_anxiety_classical1 = cleaned_data_classical1["Anxiety"].mean()
    ave_dep_classical1 = cleaned_data_classical1["Depression"].mean()
    ave_insom_classical1 = cleaned_data_classical1["Insomnia"].mean()
    ave_ocd_classical1 = cleaned_data_classical1["OCD"].mean()
    
    ave_anxiety_classical2 = cleaned_data_classical2["Anxiety"].mean()
    ave_dep_classical2 = cleaned_data_classical2["Depression"].mean()
    ave_insom_classical2 = cleaned_data_classical2["Insomnia"].mean()
    ave_ocd_classical2 = cleaned_data_classical2["OCD"].mean()
    
    ave_anxiety_classical3 = cleaned_data_classical3["Anxiety"].mean()
    ave_dep_classical3 = cleaned_data_classical3["Depression"].mean()
    ave_insom_classical3 = cleaned_data_classical3["Insomnia"].mean()
    ave_ocd_classical3 = cleaned_data_classical3["OCD"].mean()
    
    ave_anxiety_classical4 = cleaned_data_classical4["Anxiety"].mean()
    ave_dep_classical4 = cleaned_data_classical4["Depression"].mean()
    ave_insom_classical4 = cleaned_data_classical4["Insomnia"].mean()
    ave_ocd_classical4 = cleaned_data_classical4["OCD"].mean()
    
    
    
    #making Metal subsets based on frequency
    cleaned_data_metal1 = cleaned_data[cleaned_data["Frequency [Metal]"] == 1]
    cleaned_data_metal2 = cleaned_data[cleaned_data["Frequency [Metal]"] == 2]
    cleaned_data_metal3 = cleaned_data[cleaned_data["Frequency [Metal]"] == 3]
    cleaned_data_metal4 = cleaned_data[cleaned_data["Frequency [Metal]"] == 4]
    
    #now get the average MH scores for each frequency
    ave_anxiety_metal1 = cleaned_data_metal1["Anxiety"].mean()
    ave_dep_metal1 = cleaned_data_metal1["Depression"].mean()
    ave_insom_metal1 = cleaned_data_metal1["Insomnia"].mean()
    ave_ocd_metal1 = cleaned_data_metal1["OCD"].mean()
    
    ave_anxiety_metal2 = cleaned_data_metal2["Anxiety"].mean()
    ave_dep_metal2 = cleaned_data_metal2["Depression"].mean()
    ave_insom_metal2 = cleaned_data_metal2["Insomnia"].mean()
    ave_ocd_metal2 = cleaned_data_metal2["OCD"].mean()
    
    ave_anxiety_metal3 = cleaned_data_metal3["Anxiety"].mean()
    ave_dep_metal3 = cleaned_data_metal3["Depression"].mean()
    ave_insom_metal3 = cleaned_data_metal3["Insomnia"].mean()
    ave_ocd_metal3 = cleaned_data_metal3["OCD"].mean()
    
    ave_anxiety_metal4 = cleaned_data_metal4["Anxiety"].mean()
    ave_dep_metal4 = cleaned_data_metal4["Depression"].mean()
    ave_insom_metal4 = cleaned_data_metal4["Insomnia"].mean()
    ave_ocd_metal4 = cleaned_data_metal4["OCD"].mean()
    
    
    
    
    #making Folk subsets based on frequency
    cleaned_data_folk1 = cleaned_data[cleaned_data["Frequency [Folk]"] == 1]
    cleaned_data_folk2 = cleaned_data[cleaned_data["Frequency [Folk]"] == 2]
    cleaned_data_folk3 = cleaned_data[cleaned_data["Frequency [Folk]"] == 3]
    cleaned_data_folk4 = cleaned_data[cleaned_data["Frequency [Folk]"] == 4]
    
    #now get the average MH scores for each frequency
    ave_anxiety_folk1 = cleaned_data_folk1["Anxiety"].mean()
    ave_dep_folk1 = cleaned_data_folk1["Depression"].mean()
    ave_insom_folk1 = cleaned_data_folk1["Insomnia"].mean()
    ave_ocd_folk1 = cleaned_data_folk1["OCD"].mean()
    
    ave_anxiety_folk2 = cleaned_data_folk2["Anxiety"].mean()
    ave_dep_folk2 = cleaned_data_folk2["Depression"].mean()
    ave_insom_folk2 = cleaned_data_folk2["Insomnia"].mean()
    ave_ocd_folk2 = cleaned_data_folk2["OCD"].mean()
    
    ave_anxiety_folk3 = cleaned_data_folk3["Anxiety"].mean()
    ave_dep_folk3 = cleaned_data_folk3["Depression"].mean()
    ave_insom_folk3 = cleaned_data_folk3["Insomnia"].mean()
    ave_ocd_folk3 = cleaned_data_folk3["OCD"].mean()
    
    ave_anxiety_folk4 = cleaned_data_folk4["Anxiety"].mean()
    ave_dep_folk4 = cleaned_data_folk4["Depression"].mean()
    ave_insom_folk4 = cleaned_data_folk4["Insomnia"].mean()
    ave_ocd_folk4 = cleaned_data_folk4["OCD"].mean()
    
    
    
    
    
    #making Lofi subsets based on frequency
    cleaned_data_lofi1 = cleaned_data[cleaned_data["Frequency [Lofi]"] == 1]
    cleaned_data_lofi2 = cleaned_data[cleaned_data["Frequency [Lofi]"] == 2]
    cleaned_data_lofi3 = cleaned_data[cleaned_data["Frequency [Lofi]"] == 3]
    cleaned_data_lofi4 = cleaned_data[cleaned_data["Frequency [Lofi]"] == 4]
    
    #Now get the average MH scores for each frequency
    ave_anxiety_lofi1 = cleaned_data_lofi1["Anxiety"].mean()
    ave_dep_lofi1 = cleaned_data_lofi1["Depression"].mean()
    ave_insom_lofi1 = cleaned_data_lofi1["Insomnia"].mean()
    ave_ocd_lofi1 = cleaned_data_lofi1["OCD"].mean()
    
    ave_anxiety_lofi2 = cleaned_data_lofi2["Anxiety"].mean()
    ave_dep_lofi2 = cleaned_data_lofi2["Depression"].mean()
    ave_insom_lofi2 = cleaned_data_lofi2["Insomnia"].mean()
    ave_ocd_lofi2 = cleaned_data_lofi2["OCD"].mean()
    
    ave_anxiety_lofi3 = cleaned_data_lofi3["Anxiety"].mean()
    ave_dep_lofi3 = cleaned_data_lofi3["Depression"].mean()
    ave_insom_lofi3 = cleaned_data_lofi3["Insomnia"].mean()
    ave_ocd_lofi3 = cleaned_data_lofi3["OCD"].mean()
    
    ave_anxiety_lofi4 = cleaned_data_lofi4["Anxiety"].mean()
    ave_dep_lofi4 = cleaned_data_lofi4["Depression"].mean()
    ave_insom_lofi4 = cleaned_data_lofi4["Insomnia"].mean()
    ave_ocd_lofi4 = cleaned_data_lofi4["OCD"].mean()
    
    
    
    
    #making Gospel subsets based on frequency
    cleaned_data_gospel1 = cleaned_data[cleaned_data["Frequency [Gospel]"] == 1]
    cleaned_data_gospel2 = cleaned_data[cleaned_data["Frequency [Gospel]"] == 2]
    cleaned_data_gospel3 = cleaned_data[cleaned_data["Frequency [Gospel]"] == 3]
    cleaned_data_gospel4 = cleaned_data[cleaned_data["Frequency [Gospel]"] == 4]
    
    #now get the average MH scores for each frequency
    ave_anxiety_gospel1 = cleaned_data_gospel1["Anxiety"].mean()
    ave_dep_gospel1 = cleaned_data_gospel1["Depression"].mean()
    ave_insom_gospel1 = cleaned_data_gospel1["Insomnia"].mean()
    ave_ocd_gospel1 = cleaned_data_gospel1["OCD"].mean()
    
    ave_anxiety_gospel2 = cleaned_data_gospel2["Anxiety"].mean()
    ave_dep_gospel2 = cleaned_data_gospel2["Depression"].mean()
    ave_insom_gospel2 = cleaned_data_gospel2["Insomnia"].mean()
    ave_ocd_gospel2 = cleaned_data_gospel2["OCD"].mean()
    
    ave_anxiety_gospel3 = cleaned_data_gospel3["Anxiety"].mean()
    ave_dep_gospel3 = cleaned_data_gospel3["Depression"].mean()
    ave_insom_gospel3 = cleaned_data_gospel3["Insomnia"].mean()
    ave_ocd_gospel3 = cleaned_data_gospel3["OCD"].mean()
    
    ave_anxiety_gospel4 = cleaned_data_gospel4["Anxiety"].mean()
    ave_dep_gospel4 = cleaned_data_gospel4["Depression"].mean()
    ave_insom_gospel4 = cleaned_data_gospel4["Insomnia"].mean()
    ave_ocd_gospel4 = cleaned_data_gospel4["OCD"].mean()
    
    #create a dataframe for these values
    index = ["Classical", "Country", "EDM", "Folk", "Gospel", "Hip hop", "Jazz", "K pop", "Latin", "Lofi", 
             "Metal", "Pop", "R&B", "Rap", "Rock", "Video game music"]
    columns = ["Anxiety", "Depression", "Insomnia", "OCD"]
    
    mh_by_genre = pd.DataFrame(index=index, columns=columns)
    
    
    #add values
    average_anxiety = [ave_anxiety_classical4, ave_anxiety_country4, ave_anxiety_edm4,  ave_anxiety_folk4, ave_anxiety_gospel4, ave_anxiety_hiphop4, 
        ave_anxiety_jazz4, ave_anxiety_kpop4, ave_anxiety_latin4, ave_anxiety_lofi4, ave_anxiety_metal4, ave_anxiety_pop4, ave_anxiety_rnb4, 
        ave_anxiety_rap4, ave_anxiety_rock4, ave_anxiety_vgm4]
    
    average_depression = [ave_dep_classical4, ave_dep_country4, ave_dep_edm4, ave_dep_folk4, ave_dep_gospel4, ave_dep_hiphop4, ave_dep_jazz4, 
            ave_dep_kpop4, ave_dep_latin4, ave_dep_lofi4, ave_dep_metal4, ave_dep_pop4, ave_dep_rnb4, ave_dep_rap4, ave_dep_rock4, ave_dep_vgm4]
    
    average_ocd = [ave_ocd_classical4, ave_ocd_country4, ave_ocd_edm4, ave_ocd_folk4, ave_ocd_gospel4, ave_ocd_hiphop4, ave_ocd_jazz4, ave_ocd_kpop4, 
        ave_ocd_latin4, ave_ocd_lofi4, ave_ocd_metal4, ave_ocd_pop4, ave_ocd_rnb4, ave_ocd_rap4, ave_ocd_rock4, ave_ocd_vgm4]
    
    average_insomnia = [ave_insom_classical4, ave_insom_country4, ave_insom_edm4, ave_insom_folk4, ave_insom_gospel4, ave_insom_hiphop4, ave_insom_jazz4, 
            ave_insom_kpop4, ave_insom_latin4, ave_insom_lofi4, ave_insom_metal4, ave_insom_pop4, ave_insom_rnb4, ave_insom_rap4, ave_insom_rock4, 
            ave_insom_vgm4]
    
    mh_by_genre["Anxiety"] = average_anxiety
    mh_by_genre["Depression"] = average_depression
    mh_by_genre["Insomnia"] = average_insomnia
    mh_by_genre["OCD"] = average_ocd
    
    mh_by_genre["Dep Effect"] = np.where(mh_by_genre["Depression"] >= 5, 1, 0)
    mh_by_genre["Anx Effect"] = np.where(mh_by_genre["Anxiety"] >= 5, 1, 0)
    mh_by_genre["Ins Effect"] = np.where(mh_by_genre["Insomnia"] >= 5, 1, 0)
    mh_by_genre["OCD Effect"] = np.where(mh_by_genre["OCD"] >= 5, 1, 0)
    
    #This dataframe will be used to connect this analysis with the second dataset.
    effect_df = mh_by_genre.reset_index(names='Genre')
    effect_df.drop(["Anxiety", "Depression", "OCD", "Insomnia"], axis=1)
    
    
    # cleaned_data = cleaned_data.copy()
    # #I will say the max they could realistically listen to is 16 hrs
    # cleaned_data = cleaned_data[(cleaned_data["Hours per day"] < 16)]
    # #deleted 6 rows
    
    # #take away age outliers 
    # cleaned_data = cleaned_data[(cleaned_data["Age"] > 18) & (cleaned_data["Age"] < 64)]
    
    # #get the median frequency
    # values = cleaned_data["Fav genre"].value_counts()
    # #values.median()
    
    # #make the changes to rock
    # num = 21
    # length = len(cleaned_data[cleaned_data["Fav genre"] == "Rock"])
    # drop_these_many = length - num
    # random_idx = np.random.choice(cleaned_data[cleaned_data["Fav genre"] == "Rock"].index, drop_these_many, replace=False)
    # #drop the selected indices from the DataFrame
    # cleaned_data = cleaned_data.drop(random_idx)
    
    # #make the changes to metal
    # num = 21
    # length = len(cleaned_data[cleaned_data["Fav genre"] == "Metal"])
    # drop_these_many = length - num
    # random_idx = np.random.choice(cleaned_data[cleaned_data["Fav genre"] == "Metal"].index, drop_these_many, replace=False)
    # #drop the selected indices from the DataFrame
    # cleaned_data = cleaned_data.drop(random_idx)
    
    # #make the changes to pop
    # num = 21
    # length = len(cleaned_data[cleaned_data["Fav genre"] == "Pop"])
    # drop_these_many = length - num
    # random_idx = np.random.choice(cleaned_data[cleaned_data["Fav genre"] == "Pop"].index, drop_these_many, replace=False)
    # #drop the selected indices from the DataFrame
    # cleaned_data = cleaned_data.drop(random_idx)
    
    
    # ##############balance anxiety 
    # #reset index
    # cleaned_data.reset_index(drop=True, inplace=True)
    # cleaned_data["Anxiety_category"] = np.where(cleaned_data["Anxiety"] >= 5, 1, 0)
    # X = cleaned_data.drop(["Anxiety", "Anxiety_category"], axis=1)  
    # y = cleaned_data["Anxiety_category"] 
    
    # rus = RandomUnderSampler(random_state=42)
    # X_resampled, y_resampled = rus.fit_resample(X, y)
    
    # print(f"Before Undersampling: \n{y.value_counts()}")
    # print(f"After Undersampling: \n{y_resampled.value_counts()}")
    
    # resampled_indices = rus.sample_indices_
    
    # anxiety_resampled = cleaned_data.loc[resampled_indices, "Anxiety"]
    
    # cleaned_data = X_resampled.copy()  
    # cleaned_data["Anxiety"] = anxiety_resampled.values  
    
    # #reset index
    # cleaned_data.reset_index(drop=True, inplace=True)
    
    
    # ######now balance depression
    # cleaned_data["Depression_category"] = np.where(cleaned_data["Depression"] >= 5, 1, 0)
    # X = cleaned_data.drop(["Depression", "Depression_category"], axis=1)  
    # y = cleaned_data["Depression_category"] 
    
    # rus = RandomUnderSampler(random_state=42)
    # X_resampled, y_resampled = rus.fit_resample(X, y)
    
    # print(f"Before Undersampling: \n{y.value_counts()}")
    # print(f"After Undersampling: \n{y_resampled.value_counts()}")
    
    # resampled_indices = rus.sample_indices_
    
    # depression_resampled = cleaned_data.loc[resampled_indices, "Depression"]
    
    # cleaned_data = X_resampled.copy()  
    # cleaned_data["Depression"] = depression_resampled.values  
    
    # #reset index
    # cleaned_data.reset_index(drop=True, inplace=True)
    
    
    
    
    ###### adding dataset #2 edits from the Explore Data section so we can use dataset #2 in Get Recommendations
    
    songs = pd.read_csv("songs_normalize.csv")
    
    songs = songs[songs["explicit"] == False]
    
    
    #Some songs are categorized as multiple genres. Let's split that up so each song is listed once per genre that it classifies as. This will create duplicates. For example, I want a pop-rock song to be recommened for pop and rock recommedations.")
    songs["genre"] = songs["genre"].str.split(",")
    
    #explode the dataset so each genre gets its own row
    ######explode() expands the list of genres so each genre has its own row, duplicating other information about the song.
    #####reset_index(drop=True)  resets the index to keep things neat after exploding.
    songs_expanded = songs.explode("genre").reset_index(drop=True)
    
    
    
    #make sure genres are consistent
    #songs_expanded["genre"]==[" Folk/Acoustic"].replace("Folk/Acoustic")
    songs_expanded["genre"] = songs_expanded["genre"].replace(" Folk/Acoustic", "Folk/Acoustic")
    songs_expanded["genre"] = songs_expanded["genre"].replace(" Dance/Electronic", "Dance/Electronic")
    songs_expanded["genre"] = songs_expanded["genre"].replace(" pop", "pop")
    songs_expanded["genre"] = songs_expanded["genre"].replace(" hip hop", "hip hop")
    songs_expanded["genre"] = songs_expanded["genre"].replace(" country", "country")
    songs_expanded["genre"] = songs_expanded["genre"].replace(" metal", "metal")
    songs_expanded["genre"] = songs_expanded["genre"].replace(" R&B", "R&B")
    songs_expanded["genre"] = songs_expanded["genre"].replace(" rock", "rock")
    songs_expanded["genre"] = songs_expanded["genre"].replace(" easy listening", "easy listening")
    songs_expanded["genre"] = songs_expanded["genre"].replace(" latin", "latin")
    songs_expanded["genre"] = songs_expanded["genre"].replace(" classical", "classical")
    songs_expanded["genre"] = songs_expanded["genre"].replace(" blues", "blues")
    songs_expanded["genre"] = songs_expanded["genre"].replace(" jazz", "Jazz")
    
    #changing capitalization and wording
    songs_expanded["genre"] = songs_expanded["genre"].replace("pop", "Pop")
    songs_expanded["genre"] = songs_expanded["genre"].replace("rock", "Rock")
    songs_expanded["genre"] = songs_expanded["genre"].replace("country", "Country")
    songs_expanded["genre"] = songs_expanded["genre"].replace("metal", "Metal")
    songs_expanded["genre"] = songs_expanded["genre"].replace("hip hop", "Hip hop")
    songs_expanded["genre"] = songs_expanded["genre"].replace("Dance/Electronic", "EDM")
    songs_expanded["genre"] = songs_expanded["genre"].replace("Folk/Acoustic", "Folk")
    songs_expanded["genre"] = songs_expanded["genre"].replace("latin", "Latin")
    songs_expanded["genre"] = songs_expanded["genre"].replace("jazz", "Jazz")
    songs_expanded["genre"] = songs_expanded["genre"].replace("classical", "Classical")
    
    
    
    
    songs_expanded.reset_index(drop=True, inplace=True)
    songs_expanded["valence_category"] = np.where(songs_expanded["valence"] >= 0.5, 1, 0)
    #separate features (X) and target (y)
    #drop the continuous feature and the categorical version we just made
    X = songs_expanded.drop(["valence", "valence_category"], axis=1)  # Keep only non-target features
    #look at the categorical version as the target 
    y = songs_expanded["valence_category"]  # Target variable
    
    #apply RandomUnderSampler
    #initialize it
    rus = RandomUnderSampler(random_state=42)
    #apply it to X and y and store the changed versions
    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    #print the differences so we can see that the package did its job
    print(f"Before Undersampling: \n{y.value_counts()}")
    print(f"After Undersampling: \n{y_resampled.value_counts()}")
    
    #get the indices of the resampled data
    resampled_indices = rus.sample_indices_
    
    #use the indices to retrieve the original continuous valence values
    valence_resampled = songs_expanded.loc[resampled_indices, "valence"]
    
    #create the final resampled dataset with original continuous valence values
    songs_balanced = X_resampled.copy()  #start with resampled features
    songs_balanced["valence"] = valence_resampled.values  #add back continuous valence
    
    songs_balanced["valence_category"] = np.where(songs_balanced["valence"] >= 0.5, 1, 0)
    
    
    ####### adding code where I merge the datasets
    
    #delete valence_category 
    #songs_balanced = songs_balanced.drop("valence_category", axis = 1)
    
    
    #First I have to make sure the genre columns are capitalized the same
    songs_balanced.rename(columns={'genre': 'Genre'}, inplace=True)
    
    #merge them
    merged_df = pd.merge(songs_balanced, effect_df, on='Genre', how='left')
    
    
    
    
    ####################### done replicating the filtering done above

    # st.markdown("Recommendations are made based on genre first. For example, if a user's goal is to feel happy, genres associated with lower depression scores are recommended. Then, songs from those genres are chosen based on valence (perceived positivity) scores. Please see the 'Practice: Get Recommendations' page under 'Behind The Scenes' to get more info on how recommendations are made. As you view the graphs below, please know that the Gospel and Kpop genres were removed from consideration since they represented only a few participants. ")

    # st.subheader("Observe the df")
    # st.write(merged_df)

    # st.subheader("What's the balance between classes?")

    # #plot of Dep Effect

    # plt.figure(figsize=(10, 6))
    # plt.hist(merged_df["Dep Effect"], bins=11, edgecolor='black')
    
    # #set the title of the plot
    # plt.title('Distribution of Dep Effect')
    
    # #set the x-axis title
    # plt.xlabel('Depression Below 5 (0) and Above 5 (1)')
    # plt.xticks([0, 1])
    # st.pyplot(plt)

    # #plot of Anx Effect

    # plt.figure(figsize=(10, 6))
    # plt.hist(merged_df["Anx Effect"], bins=11, edgecolor='black')
    
    # #set the title of the plot
    # plt.title('Distribution of Anx Effect')
    
    # #set the x-axis title
    # plt.xlabel('Anxiety Below 5 (0) and Above 5 (1)')
    # plt.xticks([0, 1])
    # st.pyplot(plt)

    # #plot of OCD Effect

    # plt.figure(figsize=(10, 6))
    # plt.hist(merged_df["OCD Effect"], bins=11, edgecolor='black')
    
    # #set the title of the plot
    # plt.title('Distribution of OCD Effect')
    
    # #set the x-axis title
    # plt.xlabel('OCD Below 5 (0) and Above 5 (1)')
    # plt.xticks([0, 1])
    # st.pyplot(plt)

    # #plot of Ins Effect

    # plt.figure(figsize=(10, 6))
    # plt.hist(merged_df["Ins Effect"], bins=11, edgecolor='black')
    
    # #set the title of the plot
    # plt.title('Distribution of Ins Effect')
    
    # #set the x-axis title
    # plt.xlabel('Insomnia Below 5 (0) and Above 5 (1)')
    # plt.xticks([0, 1])
    # st.pyplot(plt)

    
    #I'm editing the code below so that it uses merged_df and not mh_by_genre
    ###mood_increase_genres = mh_by_genre[mh_by_genre["Dep Effect"] == 0]
    ###mood_decrease_genres = mh_by_genre[mh_by_genre["Dep Effect"] == 1]
    ###increase_recommendations = mood_increase_genres.index
    ###decrease_recommendations = mood_decrease_genres.index

    
    
    
    feel_happy = merged_df[merged_df["Dep Effect"] == 0]
    feel_sad = merged_df[merged_df["Dep Effect"] == 1]
    #select only rows where Anxiety score is less 5/10
    feel_calm = merged_df[merged_df["Anx Effect"] == 0]
    #get average dance score 
    avg_danceability = merged_df["danceability"].mean()
    high_danceability_df = merged_df[merged_df["danceability"] >= avg_danceability]
    low_danceability_df =  merged_df[merged_df["danceability"] < avg_danceability]
    
    feel_happy_recs = feel_happy[["artist", "song", "year"]]
    #feel_happy_recs = feel_happy
    feel_sad_recs = feel_sad[["artist", "song", "year"]]
    #feel_sad_recs = feel_sad
    feel_calm_recs = feel_calm[["artist", "song", "year"]]
    #feel_calm_recs = feel_calm
    feel_dancey_recs = high_danceability_df[["artist", "song", "year"]]
    #feel_dancey_recs = high_danceability_df

    st.markdown("Please choose a listening goal to recieve aligned recommendations based on genre and song features.")
    #dropdown menu
    categories = ["Happy", "Sad", "Calm", "Dance"]
    selected_category = st.selectbox("Choose a listening goal:", categories)
    
    if selected_category == "Happy":
    
        #display the selected category
        st.write(f"You selected: {selected_category}")
    
        st.markdown("Here are your recommended genres:")
        #st.write(feel_happy_recs)
        st.write(feel_happy["Genre"].unique())
        #st.markdown("Here are song suggestions from those genres. (Notice, there are sad songs in this list. I can trust valence more than danceability and energy. For example, 'What's Hurts The Most' by Rascal Flats, a sad song based on the lyrics, has high danceability and energy, but low valence. I'll use valence to improve these recommendations below.)")
        #st.write(feel_happy_recs)
    
        
        #st.markdown("This plot demonstrates the need to enhance recommendations further. There's too much variety in each genre to base recs off genre alone.")

        #plt.close(fig)
        
        ## Melt the DataFrame to a long format
        #happy_features = feel_happy[["valence", "danceability", "energy"]]
        #df_long = happy_features.melt(var_name="Feature", value_name="Value")
        
        ## Create the scatterplot
        #fig = px.scatter(
            #df_long, 
            #x=df_long.index,  # The index as x-axis
            #y="Value", 
            #color="Feature",
            #title="Scatterplot of Features",
            #labels={"index": "Row Index", "Value": "Feature Value"}
        #)
        
        ## Display the plot in Streamlit
        #import streamlit as st
        #st.plotly_chart(fig)

        st.markdown("Here are your recommended songs within those genres based on your listening goal:")
        #st.write(feel_happy_recs)
        avg_valence = merged_df["valence"].mean()
        above_avg_valence = feel_happy[feel_happy["valence"] >= avg_valence]
        st.write(above_avg_valence[["artist", "song", "year"]])


        #include a visualization
        # Set the plot style
        sns.set(style="whitegrid")
    
        # Create a figure and axis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
        # Create a bar plot for each mental health measure
        sns.barplot(x=mh_by_genre.index, y='Anxiety', data=mh_by_genre, ax=axes[0, 0], palette='viridis')
        axes[0, 0].set_title('Anxiety Levels by Genre')
        axes[0, 1].set_xlabel('Genre')
        axes[0, 0].set_ylabel('Anxiety Level')
        axes[0, 0].tick_params(axis='x', rotation=45)
            
        sns.barplot(x=mh_by_genre.index, y='Depression', data=mh_by_genre, ax=axes[0, 1], palette='viridis')
        axes[0, 1].set_title('Depression Levels by Genre')
        axes[0, 1].set_xlabel('Genre')
        axes[0, 1].set_ylabel('Depression Level')
        axes[0, 1].tick_params(axis='x', rotation=45)
            
        sns.barplot(x=mh_by_genre.index, y='Insomnia', data=mh_by_genre, ax=axes[1, 0], palette='viridis')
        axes[1, 0].set_title('Insomnia Levels by Genre')
        axes[1, 0].set_xlabel('Genre')
        axes[1, 0].set_ylabel('Insomnia Level')
        axes[1, 0].tick_params(axis='x', rotation=45)
            
        sns.barplot(x=mh_by_genre.index, y='OCD', data=mh_by_genre, ax=axes[1, 1], palette='viridis')
        axes[1, 1].set_title('OCD Levels by Genre')
        axes[1, 1].set_ylabel('OCD Level')
        axes[1, 1].set_xlabel('Genre')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.suptitle('Mental Health Levels Associated With Music Genres', fontsize=16, y=1.02)
    
        plt.tight_layout()
            
        #show the plot
        st.pyplot(plt)
        st.write("Genre recommendations are based on the above mental health levels.")
        #plt.close(fig)
        
        
        # Melt the DataFrame to a long format
        df_long = above_avg_valence[["valence", "danceability", "energy"]]
        df_long = df_long.melt(var_name="Feature", value_name="Value")
        
        # Create the scatterplot
        fig = px.scatter(
            df_long, 
            x=df_long.index,  # The index as x-axis
            y="Value", 
            color="Feature",
            title="Features of Recommended Songs",
            labels={"index": "Row Index", "Value": "Feature Value"}
        )

        # Update the y-axis range and ticks
        fig.update_yaxes(range=[0, 1], tick0=0, dtick=0.2)
        
        # Display the plot in Streamlit
        import streamlit as st
        st.plotly_chart(fig)
        st.write("Song recommendations are based on the above features.")

    
    if selected_category == "Sad":
    
        #display the selected category
        st.write(f"You selected: {selected_category}")
    
        st.markdown("Here are your recommended genres:")
        st.write(feel_sad["Genre"].unique())
        

        st.markdown("Here are your recommended songs within those genres based on your listening goal:")
        #changing from mean to median because upbeat songs are being recommended 
        avg_valence = merged_df["valence"].mean()
        avg_danceability = merged_df["danceability"].median()
        avg_energy = merged_df["energy"].median()
        # Filter rows where valence, danceability, and energy are below their respective averages
        below_avg_val_dan_ener = feel_sad[
            (feel_sad["valence"] < avg_valence) &
            (feel_sad["danceability"] < avg_danceability) &
            (feel_sad["energy"] < avg_energy)
        ]
        st.write(below_avg_val_dan_ener[["artist", "song", "year"]])


        #include a visualization
        # Set the plot style
        sns.set(style="whitegrid")
    
        # Create a figure and axis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
        # Create a bar plot for each mental health measure
        sns.barplot(x=mh_by_genre.index, y='Anxiety', data=mh_by_genre, ax=axes[0, 0], palette='viridis')
        axes[0, 0].set_title('Anxiety Levels by Genre')
        axes[0, 0].set_xlabel('Genre')
        axes[0, 0].set_ylabel('Anxiety Level')
        axes[0, 0].tick_params(axis='x', rotation=45)
            
        sns.barplot(x=mh_by_genre.index, y='Depression', data=mh_by_genre, ax=axes[0, 1], palette='viridis')
        axes[0, 1].set_title('Depression Levels by Genre')
        axes[0, 1].set_xlabel('Genre')
        axes[0, 1].set_ylabel('Depression Level')
        axes[0, 1].tick_params(axis='x', rotation=45)
            
        sns.barplot(x=mh_by_genre.index, y='Insomnia', data=mh_by_genre, ax=axes[1, 0], palette='viridis')
        axes[1, 0].set_title('Insomnia Levels by Genre')
        axes[1, 0].set_xlabel('Genre')
        axes[1, 0].set_ylabel('Insomnia Level')
        axes[1, 0].tick_params(axis='x', rotation=45)
            
        sns.barplot(x=mh_by_genre.index, y='OCD', data=mh_by_genre, ax=axes[1, 1], palette='viridis')
        axes[1, 1].set_title('OCD Levels by Genre')
        axes[1, 1].set_xlabel('Genre')
        axes[1, 1].set_ylabel('OCD Level')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.suptitle('Mental Health Levels Associated With Music Genres', fontsize=16, y=1.02)
    
        plt.tight_layout()
            
        #show the plot
        st.pyplot(plt)
        
        #plt.close(fig)
        st.write("Genre recommendations are based on the above mental health levels.")
        
        # Melt the DataFrame to a long format
        df_long = below_avg_val_dan_ener[["valence", "danceability", "energy"]]
        df_long = df_long.melt(var_name="Feature", value_name="Value")
        
        # Create the scatterplot
        fig = px.scatter(
            df_long, 
            x=df_long.index,  # The index as x-axis
            y="Value", 
            color="Feature",
            title="Features of Recommended Songs",
            labels={"index": "Row Index", "Value": "Feature Value"}
        )

        # Update the y-axis range and ticks
        fig.update_yaxes(range=[0, 1], tick0=0, dtick=0.2)
        
        # Display the plot in Streamlit
        import streamlit as st
        st.plotly_chart(fig)
        st.write("Song recommendations are based on the above features.")

    
    if selected_category == "Calm":
    
        #display the selected category
        st.write(f"You selected: {selected_category}")
    
        st.markdown("Here are your recommended genres:")
        st.write(feel_calm["Genre"].unique())
        

        st.markdown("Here are your recommended songs within those genres based on your listening goal:")
        #changing from mean to median because upbeat songs are being recommended 
        avg_valence = merged_df["valence"].mean()
        avg_danceability = merged_df["danceability"].median()
        avg_energy = merged_df["energy"].median()
        # Filter rows where valence, danceability are below 0.6 to recommend songs that are neutral on these features. Average seemed to high and to close to the Happy scatterplot
        below_point_six = feel_calm[
            (feel_calm["energy"] <= 0.6) &
            (feel_calm["danceability"] <= 0.6) &
            (feel_calm["valence"] <= 0.6) 
        ]
        
        st.write(below_point_six[["artist", "song", "year"]])


        #include a visualization
        # Set the plot style
        sns.set(style="whitegrid")
    
        # Create a figure and axis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
        # Create a bar plot for each mental health measure
        sns.barplot(x=mh_by_genre.index, y='Anxiety', data=mh_by_genre, ax=axes[0, 0], palette='viridis')
        axes[0, 0].set_title('Anxiety Levels by Genre')
        axes[0, 0].set_xlabel('Genre')
        axes[0, 0].set_ylabel('Anxiety Level')
        axes[0, 0].tick_params(axis='x', rotation=45)
            
        sns.barplot(x=mh_by_genre.index, y='Depression', data=mh_by_genre, ax=axes[0, 1], palette='viridis')
        axes[0, 1].set_title('Depression Levels by Genre')
        axes[0, 1].set_xlabel('Genre')
        axes[0, 1].set_ylabel('Depression Level')
        axes[0, 1].tick_params(axis='x', rotation=45)
            
        sns.barplot(x=mh_by_genre.index, y='Insomnia', data=mh_by_genre, ax=axes[1, 0], palette='viridis')
        axes[1, 0].set_title('Insomnia Levels by Genre')
        axes[1, 0].set_xlabel('Genre')
        axes[1, 0].set_ylabel('Insomnia Level')
        axes[1, 0].tick_params(axis='x', rotation=45)
            
        sns.barplot(x=mh_by_genre.index, y='OCD', data=mh_by_genre, ax=axes[1, 1], palette='viridis')
        axes[1, 1].set_title('OCD Levels by Genre')
        axes[1, 1].set_xlabel('Genre')
        axes[1, 1].set_ylabel('OCD Level')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.suptitle('Mental Health Levels Associated With Music Genres', fontsize=16, y=1.02)
        
        plt.tight_layout()
            
        #show the plot
        st.pyplot(plt)

        #plt.close(fig)
        st.write("Genre recommendations are based on the above mental health levels.")
        
        
        # Melt the DataFrame to a long format
        df_long = below_point_six[["valence", "danceability", "energy"]]
        df_long = df_long.melt(var_name="Feature", value_name="Value")
        
        # Create the scatterplot
        fig = px.scatter(
            df_long, 
            x=df_long.index,  # The index as x-axis
            y="Value", 
            color="Feature",
            title="Features of Recommended Songs",
            labels={"index": "Row Index", "Value": "Feature Value"}
        )

        # Update the y-axis range and ticks
        fig.update_yaxes(range=[0, 1], tick0=0, dtick=0.2)
        
        # Display the plot in Streamlit
        import streamlit as st
        st.plotly_chart(fig)
        st.write("Song recommendations are based on the above features.")


    if selected_category == "Dance":
    
        #display the selected category
        st.write(f"You selected: {selected_category}")
    
        st.markdown("Here are your recommended genres:")
        st.write(feel_happy["Genre"].unique())
        

        st.markdown("Here are your recommended songs within those genres based on your listening goal:")
        #changing from mean to median because upbeat songs are being recommended 
        avg_valence = merged_df["valence"].mean()
        avg_danceability = merged_df["danceability"].median()
        avg_energy = merged_df["energy"].median()
        high_dance = feel_happy[feel_happy["danceability"] >= avg_danceability]
        st.write(high_dance[["artist", "song", "year"]])


        #include a visualization
        # Set the plot style
        sns.set(style="whitegrid")
    
        # Create a figure and axis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
        # Create a bar plot for each mental health measure
        sns.barplot(x=mh_by_genre.index, y='Anxiety', data=mh_by_genre, ax=axes[0, 0], palette='viridis')
        axes[0, 0].set_title('Anxiety Levels by Genre')
        axes[0, 0].set_xlabel('Genre')
        axes[0, 0].set_ylabel('Anxiety Level')
        axes[0, 0].tick_params(axis='x', rotation=45)
            
        sns.barplot(x=mh_by_genre.index, y='Depression', data=mh_by_genre, ax=axes[0, 1], palette='viridis')
        axes[0, 1].set_title('Depression Levels by Genre')
        axes[0, 1].set_xlabel('Genre')
        axes[0, 1].set_ylabel('Depression Level')
        axes[0, 1].tick_params(axis='x', rotation=45)
            
        sns.barplot(x=mh_by_genre.index, y='Insomnia', data=mh_by_genre, ax=axes[1, 0], palette='viridis')
        axes[1, 0].set_title('Insomnia Levels by Genre')
        axes[1, 0].set_xlabel('Genre')
        axes[1, 0].set_ylabel('Insomnia Level')
        axes[1, 0].tick_params(axis='x', rotation=45)
            
        sns.barplot(x=mh_by_genre.index, y='OCD', data=mh_by_genre, ax=axes[1, 1], palette='viridis')
        axes[1, 1].set_title('OCD Levels by Genre')
        axes[1, 1].set_xlabel('Genre')
        axes[1, 1].set_ylabel('OCD Level')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.suptitle('Mental Health Levels Associated With Music Genres', fontsize=16, y=1.02)
    
        plt.tight_layout()
            
        #show the plot
        #st.pyplot(plt)
        st.pyplot(fig)


        
        #plt.close(fig)
        st.write("Genre recommendations are based on the above mental health levels.")
        
        # Melt the DataFrame to a long format
        df_long = high_dance[["valence", "danceability", "energy"]]
        df_long = df_long.melt(var_name="Feature", value_name="Value")
        
        # Create the scatterplot
        fig = px.scatter(
            df_long, 
            x=df_long.index,  # The index as x-axis
            y="Value", 
            color="Feature",
            title="Features of Recommended Songs",
            labels={"index": "Row Index", "Value": "Feature Value"}
        )

        # Update the y-axis range and ticks
        fig.update_yaxes(range=[0, 1], tick0=0, dtick=0.2)
        
        # Display the plot in Streamlit
        import streamlit as st
        st.plotly_chart(fig)
        st.write("Song recommendations are based on the above features.")


#add a footer
st.markdown(
    """
    <style>
        .footer {
            text-align: center;
            font-size: 14px;
            margin-top: 20px;
            color: #888888;
        }
    </style>
    <div class="footer">
        © 2024 Tunes By Mood | Designed by Kayla Bowman
    </div>
    """,
    unsafe_allow_html=True
)

