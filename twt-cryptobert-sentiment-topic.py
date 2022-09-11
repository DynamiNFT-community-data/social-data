#%% Import libraries

from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import re

# %% Clean text
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        # t = "@user" if t.startswith("@") and len(t) > 1 else t
        # t = "http" if t.startswith("http") else t
        t = re.sub("\\n", "", t)
        new_text.append(t)
    return " ".join(new_text)


tweets_df["clean_text"] = tweets_df.text.apply(lambda x: preprocess(x))


#%%
######## Detect SPAM ############
tokenizer = AutoTokenizer.from_pretrained("mvonwyl/roberta-twitter-spam-classifier")

spam_model = AutoModelForSequenceClassification.from_pretrained("mvonwyl/roberta-twitter-spam-classifier")

pipe_spam = TextClassificationPipeline(model=spam_model, tokenizer=tokenizer)

tweets_df["spam"] = np.nan
tweets_df["spam_score"] = np.nan


def translate_spam_label(x):
    if x == "LABEL_0":
        return "ham"
    if x == "LABEL_1":
        return "spam"


for i, row in tweets_df.iterrows():
    score = pipe_spam(row["clean_text"])
    tweets_df.at[i, "spam"] = translate_spam_label(score[0]["label"])
    tweets_df.at[i, "spam_score"] = score[0]["score"]

print(tweets_df[["clean_text", "spam", "spam_score"]])

# %% Sentiment analysis


model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
pipe = TextClassificationPipeline(
    model=model, tokenizer=tokenizer, max_length=128, truncation=True, padding="max_length"
)

# %%

tweets_df["sentiment"] = np.nan
tweets_df["sentiment_score"] = np.nan

for i, row in tweets_df.iterrows():
    score = pipe(row["clean_text"])
    tweets_df.at[i, "sentiment"] = score[0]["label"]
    tweets_df.at[i, "sentiment_score"] = score[0]["score"]


tweets_df[["clean_text", "sentiment"]]
# %% Topic modelling

from top2vec import Top2Vec

# %%
docs = tweets_df.clean_text.tolist()
model = Top2Vec(documents=docs, speed="deep-learn")

# %%
print(model.get_num_topics())

(topic_words, word_scores, topic_nums) = model.get_topics(4)
for topic in topic_nums:
    model.generate_topic_wordcloud(topic)


# %%
