
#YouTube Comments Sentiment Analyser


# 1. Data Extraction
import googleapiclient.discovery
import pandas as pd

def get_youtube_comments(api_key, video_id, max_comments):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )
    comments = []
    while request and len(comments) < max_comments:
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if len(comments) >= max_comments:
                break
        if "nextPageToken" in response and len(comments) < max_comments:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                pageToken=response["nextPageToken"],
                maxResults=100,
                textFormat="plainText"
            )
        else:
            break
    return comments

if __name__ == "__main__":
    api_key = 'AIzaSyDG1HEhYKO765JH3SY_CQARprIErYAMMpE'
    video_id = input("Enter YouTube video ID: ")
    saved = input("Enter the file name to be saved as: (e.g., comments.csv) ")
    max_comments = int(input("Enter the maximum number of comments to fetch: "))
    
    comments = get_youtube_comments(api_key, video_id, max_comments)
    
    df = pd.DataFrame(comments, columns=["Comment"])
    df.to_csv(saved, index=False)
    print(f"Comments saved to {saved}")


# 2.Data Transformation

#Libraries Required:

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#%matplotlib inline
import os

# Import functions for data preprocessing & data preparation
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import string
from string import punctuation
import nltk
import re


# 3. Reading Data
#code that reads a CSV and checks the columns before proceeding:

import pandas as pd

# Load the CSV file
file_path = saved
df = pd.read_csv(file_path)

# Print the columns to understand the structure
print("Columns in the DataFrame:")
print(df.columns)

# 4. Droping the unnecessary columns

data = pd.read_csv(file_path)
data.columns
#data1=data.drop(['Unnamed: 0','Likes','Time','user','UserLink'],axis=1)
data1=data

# 5. Data Labeling

nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
data1["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data1["Comment"]]
data1["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data1["Comment"]]
data1["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data1["Comment"]]
data1['Compound'] = [sentiments.polarity_scores(i)["compound"] for i in data1["Comment"]]
score = data1["Compound"].values
sentiment = []
for i in score:
    if i >= 0.05 :
        sentiment.append('Positive')
    elif i <= -0.05 :
        sentiment.append('Negative')
    else:
        sentiment.append('Neutral')
data1["Sentiment"] = sentiment
print("\n Total Comments fetched:- \n\n")
print(data1.head(max_comments))

#6. Final Data

data2=data1.drop(['Positive','Negative','Neutral','Compound'],axis=1)
data2 = pd.DataFrame(data2)
print("\n\nComments after droping few columns:-\n\n")
print(data2.head(max_comments))

#7. If the NLTK library is unable to find the "stopwords" resource. You can easily fix by downloading the required resource.Run the following command:

import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer

# 8. Data Transformation

stop_words = stopwords.words('english')
porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer() 
snowball_stemer = SnowballStemmer(language="english")
lzr = WordNetLemmatizer()

# 9. 

def text_processing(text):   
    # convert text into lowercase
    text = text.lower()

    # remove new line characters in text
    text = re.sub(r'\n',' ', text)
    
    # remove punctuations from text
    text = re.sub('[%s]' % re.escape(punctuation), "", text)
    
    # remove references and hashtags from text
    text = re.sub("^a-zA-Z0-9$,.", "", text)
    
    # remove multiple spaces from text
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    
    # remove special characters from text
    text = re.sub(r'\W', ' ', text)

    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
    
    text=' '.join([lzr.lemmatize(word) for word in word_tokenize(text)])

    return text

# 10. 

import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')

#Check for Download Location:
#Verify where NLTK is looking for its data files. You can print the paths it searches:
import nltk
print(nltk.data.path)

#If necessary, add your custom path where you downloaded the NLTK resources:
nltk.data.path.append('C:\\Users\\balaj\\AppData\\Roaming\\nltk_data')  # Replace with your actual path

#Confirm that the required resources are indeed available:
try:
    nltk.data.find('corpora/stopwords.zip')
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('corpora/omw-1.4.zip')
    print("\nResources found!")
except LookupError:
    print("Resources not found!")

#Ensure your text_processing function is defined correctly, and you’re using it properly. Here’s an example:
from nltk.corpus import stopwords
print("\n Assigning Sentiment values to the fetched comments\n")
# Ensure stopwords are loaded
stop_words = set(stopwords.words('english'))
def text_processing(text):
    # Remove stopwords from the text
    processed_text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    return processed_text

# Make a copy of your DataFrame
data_copy = data2.copy()
# Apply text processing
data_copy['Comment'] = data_copy['Comment'].apply(lambda text: text_processing(text))
#
le = LabelEncoder()
data_copy['Sentiment'] = le.fit_transform(data_copy['Sentiment'])
#
processed_data = {
    'Sentence':data_copy.Comment,
    'Sentiment':data_copy['Sentiment']
}

processed_data = pd.DataFrame(processed_data)
print(processed_data.head(max_comments))
#
print("\n\nTotal Comments:\n\nNegative--0\nNeutral--1\nPositive--2\n\n")
print(processed_data['Sentiment'].value_counts())

# Balancing Data

import pandas as pd
from sklearn.utils import resample
print("\nBalancing all the comments Equally based on the number of comments fetched\n")
# Assuming 'processed_data' is your original dataframe
# Find the class with the maximum number of samples
max_samples = max_comments//3 #processed_data['Sentiment'].value_counts().min()

# Function to upsample a dataframe
def upsample_minority(df, target_class, n_samples):
    return resample(
        df, 
        replace=True,    # sample with replacement
        n_samples=n_samples, # number of samples in the new dataframe
        random_state=42  # reproducible results
    )

# Separate data by sentiment
df_neutral = processed_data[processed_data['Sentiment'] == 1]
df_negative = processed_data[processed_data['Sentiment'] == 0]
df_positive = processed_data[processed_data['Sentiment'] == 2]

# Upsample minority classes to match the majority class
df_negative_upsampled = upsample_minority(df_negative, 0, max_samples)
df_neutral_upsampled = upsample_minority(df_neutral, 1, max_samples)
df_positive_upsampled = upsample_minority(df_positive, 2, max_samples)

# Concatenate the upsampled dataframes
final_data = pd.concat([df_negative_upsampled, df_neutral_upsampled, df_positive_upsampled])

# Display the shape of the final dataframe to confirm balancing
#print(final_data.shape)

# Optional: Display the count of each sentiment class in the final dataframe to confirm balancing
print(final_data['Sentiment'].value_counts())

#    
corpus = []
for sentence in final_data['Sentence']:
    corpus.append(sentence)
corpus[0:5]

#
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = final_data.iloc[:, -1].values

#Machine Learning Model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm

# Accuracy
nb_score = accuracy_score(y_test, y_pred)
print('Accuracy:',(nb_score*100))

