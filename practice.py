{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "aaeaa045-d50b-4daf-b30e-c7d4b7a51e1c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from imblearn.over_sampling import SMOTE\n",
    "import plotly.express as px\n",
    "#theres a spotify package\n",
    "#makes a correlation matrix\n",
    "import plotly.figure_factory as ff # HW 4\n",
    "#go.Figure makes a pair plot\n",
    "import plotly.graph_objs as go # HW4\n",
    "from imblearn.over_sampling import SMOTE\n",
    "from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "311ac4ae-0763-466c-94fb-f95b118c33b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "def main():\n",
    "    #title of the app\n",
    "    st.title(\"Music Therapy App\")\n",
    "\n",
    "    #dropdown menu\n",
    "    categories = [\"The Data\", \"Investigate The Data\", \"Clean The Data\", \"Explore The Data\"]\n",
    "    selected_category = st.selectbox(\"Choose one:\", categories)\n",
    "\n",
    "    #display the selected category\n",
    "    st.write(f\"You selected: {selected_category}\")\n",
    "\n",
    "    #markdown section\n",
    "    st.markdown(\"# Load The Data\")\n",
    "    st.markdown(\"* found this dataset on [Kaggle](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)\")\n",
    "\n",
    "     #load the Data\n",
    "    mxmh_survey_results = pd.read_csv(\"mxmh_survey_results.csv\")\n",
    "    \n",
    "    #display the data\n",
    "    st.write(mxmh_survey_results.head())  \n",
    "\n",
    "    #markdown section\n",
    "    st.markdown(\"The frequency columns are categorical. We'll have to recode those to be numeric so we can run correlations.\")\n",
    "    st.markdown(\"What are our features?\")\n",
    "\n",
    "    #number of features\n",
    "    num_features = len(mxmh_survey_results.columns)\n",
    "    st.write(f\"Number of features: {num_features}\")\n",
    "\n",
    "    #names of the features\n",
    "    st.write(\"Features in the dataset:\")\n",
    "    st.write(mxmh_survey_results.columns.tolist())  \n",
    "\n",
    "    if __name__ == \"__main__\":\n",
    "        main()\n",
    "    \n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "41f1d980-a7bc-4234-ad82-e9e6fc9928df",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
