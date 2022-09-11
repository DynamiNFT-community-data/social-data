# %%
# Import libraries
import snscrape.modules.twitter as sn
import pandas as pd
import matplotlib.pyplot as plt
import requests
import math
import os


# Define functions
def getTweets(query, limit=10):
    tweets = []

    for tweet in sn.TwitterSearchScraper(query).get_items():
        if len(tweets) == limit:
            break

        else:
            tweets.append(
                [
                    tweet.date,
                    tweet.user,
                    tweet.user.username,
                    tweet.content,
                    tweet.id,
                    tweet.url,
                    tweet.replyCount,
                    tweet.retweetCount,
                    tweet.likeCount,
                    tweet.quoteCount,
                    tweet.mentionedUsers,
                    tweet.lang,
                    tweet.retweetedTweet,
                    tweet.quotedTweet,
                    tweet.media,
                    tweet.outlinks,
                    tweet.tcooutlinks,
                ]
            )

    #  Save twitter data to dataframe
    tweets_df = pd.DataFrame(
        tweets,
        columns=[
            "date",
            "user",
            "username",
            "text",
            "tweet_id",
            "url",
            "replies",
            "retweets",
            "likes",
            "quotes",
            "mentioned_users",
            "lang",
            "retweet_source",
            "quoted_source",
            "media",
            "outlinks",
            "tcooutlinks",
        ],
    )

    return tweets_df


# %% Scrape twitter data
query = 'boredapeyc OR bayc OR "bored ape" -is:retweet'
tweets_df = getTweets(query, 100)
tweets_df["mentioned_users_count"] = [len(x) if type(x) == list else 0 for x in tweets_df.mentioned_users]

print(tweets_df)

# %% Calculate social engagement metrics
resample_interval = "5min"
tweets_df["count"] = 1
rs = tweets_df.resample(resample_interval, on="date", origin="start_day")
tweets_df_agg = rs[["count", "replies", "likes", "retweets", "quotes"]].sum()

# Number of unique contributors
tweets_df_unique_users = tweets_df.drop_duplicates(subset="username")
tweets_df_agg["contributors"] = tweets_df_unique_users.resample(resample_interval, on="date", origin="start_day")[
    "count"
].sum()

tweets_df_agg

# %% Get most influential tweets

# %% ### Get nft twitter account stats
brand_user = getTweets("from:boredapeyc", 1)["user"][0]
brand_user

# %% #### Calulate % unique pfps on twitter ####
# Get all nfts metadata from collection
contract_address = "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d"

page_size = 50


def getNftsFromContract(contract_address, page_number, page_size=50):
    url = f"https://api.nftport.xyz/v0/nfts/{contract_address}"  # bayc contract address
    querystring = {"chain": "ethereum", "include": "metadata", "page_number": page_number, "page_size": page_size}
    headers = {"Content-Type": "application/json", "Authorization": "ae102b1f-3d0c-42dd-ba2f-b192b641f82f"}
    response = requests.request("GET", url, headers=headers, params=querystring)
    # print(response.text)
    return response.json()


total_nfts = getNftsFromContract(contract_address, 1)["total"]
pages = list(range(1, math.ceil(total_nfts / page_size) + 1, 1))

json_data = []

for page_number in pages:
    print(page_number)
    json_data.append(getNftsFromContract(contract_address, page_number))

# %%
# Pull out only nft data from json data
collection = json_data[0]["contract"]["name"]
nfts_data = list(map(lambda x: x["nfts"], json_data))
nfts_data = sum(nfts_data, [])
nfts_images = pd.DataFrame(nfts_data, columns=["token_id", "cached_file_url"])

nfts_images.head()
nfts_images.shape

# %%
## Get pfps of twitter users
# twitter_pfps = list(map(lambda x: x.profileImageUrl.replace("_normal", ''), tweets_df_unique_users.user))
twitter_pfps = pd.DataFrame(columns=["user_id", "profile_image_url"])
for row in tweets_df_unique_users.user:
    twitter_pfps = twitter_pfps.append(
        {"user_id": row.id, "profile_image_url": row.profileImageUrl.replace("_normal", "")}, ignore_index=True
    )

twitter_pfps


# %% Download all of collection's nft images to local folder

folder_name = "nft-images"
for i, row in nfts_images.iterrows():
    url = row["cached_file_url"]
    filename = f'{collection}-{row["token_id"]}.png'
    # filename = url.split('/')[-1]
    print(filename)

    if folder_name not in os.listdir():
        os.makedirs(folder_name)

    r = requests.get(url)
    with open(f"{folder_name}/{filename}", "wb") as f:
        f.write(r.content)
    # shutil.rmtree(dirpath)


# %% Download all of twitter pfps to local folder

folder_name = "pfp-images"
for i, row in twitter_pfps.iterrows():
    url = row["profile_image_url"]
    twitter_image_filename = url.split("/")[-1]
    user_id = row["user_id"]
    filename = f"{user_id}-{twitter_image_filename}"

    print(filename)
    if folder_name not in os.listdir():
        os.makedirs(folder_name)

    r = requests.get(url)
    with open(f"{folder_name}/{filename}", "wb") as f:
        f.write(r.content)
    # shutil.rmtree(dirpath)
