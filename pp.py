import pandas as pd
df=pd.read_csv('data/7817_1.csv')
#making the tables for summarizing the data set

df=df[["reviews.text","reviews.rating"]].dropna()#dropna to remove any missing text of the selected colums
#to make the sentiment labeling here
def label_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'
df["label"]=df["reviews.rating"].apply(label_sentiment)
df["reviewText"]=df["reviews.text"].str.lower().astype(str)#convert to lowercase and string type
df.to_csv('data/pp_7817.csv',index=False)#save the preprocessed data
print("preprocessing completes and the file is saved as pp_7817.csv")