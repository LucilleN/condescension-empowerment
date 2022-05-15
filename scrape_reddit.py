from psaw import PushshiftAPI
import pandas as pd
import datetime as dt


api = PushshiftAPI()

limit = 10000

subreddits = [
    # "DecidingToBeBetter",
    # "productivity",
    # "selfimprovement",
    "GetMotivated",
    "LifeProTips",
    "ted",
    "ZenHabits",
    "YouShouldKnow"
]

if __name__ == "__main__":

    for subreddit in subreddits:

        for year in range(2005, 2022): # everything from Jan 1, 2005 (inclusive) to Jan 1, 2022 (exclusive)

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
                data_to_keep = submissions[['title', 'score']]
                print(f"data_to_keep shape: {data_to_keep.shape}")

                data_to_keep.to_csv(f"reddit_scrape/{subreddit}.csv", mode='a', sep="\t", index=False)


    ##################
    # print(DecidingToBeBetter_submissions['title'].sample(10))

    # all_dates = pd.date_range('2019-05-25', '2020-06-27', freq='D')
    # print(all_dates)




