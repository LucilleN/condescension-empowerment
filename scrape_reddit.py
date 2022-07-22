from psaw import PushshiftAPI
import pandas as pd
import datetime as dt
import numpy as np

api = PushshiftAPI()

limit = 10000

subreddits = [
    "DecidingToBeBetter",
    "productivity",
    "selfimprovement",
    "GetMotivated",
    "LifeProTips",
    "ted",
    "ZenHabits",
    "YouShouldKnow"
]

data_fields = [
    'title', 
    'score', 
    'upvote_ratio', 
    'all_awardings'
]

if __name__ == "__main__":

    for subreddit in subreddits:

        for year in range(2005, 2022): # everything from Jan 1, 2005 (inclusive) to Jan 1, 2022 (exclusive)
        # for year in range(2020, 2022): # everything from Jan 1, 2005 (inclusive) to Jan 1, 2022 (exclusive)

            for month in range(1, 13): # every month from January (month 1) to December (month 12)

                start_epoch=int(dt.datetime(year, month, 1).timestamp())
                if month < 12:
                    end_epoch=int(dt.datetime(year, month + 1, 1).timestamp())
                else:
                    end_epoch=int(dt.datetime(year + 1, 1, 1).timestamp())

                api_request_generator = api.search_submissions(
                    subreddit=subreddit, 
                    after=start_epoch,
                    before=end_epoch, 
                    limit=limit)

                # let's see if theres anything in here
                # print(next(api_request_generator))

                submissions = pd.DataFrame([submission.d_ for submission in api_request_generator])
                if len(submissions) == 0:
                    continue
                if len(submissions) == limit:
                    print(f"\n!!!!!!!!!! {subreddit} {month}-{year} exceeded limit: {submissions.shape}\n")

                print(f"\n########## {subreddit} {month}-{year} submissions shape: {submissions.shape}\n")
                # print(DecidingToBeBetter_submissions[['title', 'score']].sample(10))
                # data_to_keep = submissions[['title', 'score']]
                
                # for column_name in data_fields:
                #     if column_name not in submissions: 
                #         submissions[column_name] = np.nan

                if 'upvote_ratio' not in submissions:
                    submissions['upvote_ratio'] = -1
                if 'all_awardings' not in submissions:
                    submissions['all_awardings'] = np.empty((len(submissions), 0)).tolist()

                data_to_keep = submissions[['title', 'score', 'upvote_ratio', 'all_awardings']]
                # print(data_to_keep['author_flair_css_class'])
                # print(data_to_keep['author_flair_text'])
                # print(data_to_keep['score'])
                # print(data_to_keep['link_flair_text'])
                # print(data_to_keep['all_awardings'])
                # for item in data_to_keep["all_awardings"]:
                #     if len(item) > 0:
                #         print(item)
                
                print(f"data_to_keep shape: {data_to_keep.shape}")
                for col in data_to_keep.columns:
                    print(col)

                data_to_keep.to_csv(f"data/reddit_scrape_2/{subreddit}.csv", mode='a', sep="\t", header=False, index=False)




