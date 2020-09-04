import os
from langdetect import detect
from time import sleep

import GetOldTweets3 as got
import pandas as pd
import numpy as np

import warnings

from Company import Company

warnings.filterwarnings("ignore")


class TweetScrapper:
    def __init__(self, no_of_tweets=100):
        """
        Class used for performing the tweet scrapping task
        Parameters
        ----------
        no_of_tweets : int
            number of tweets that should be scrapped
        """
        self.no_of_tweets = no_of_tweets

    def scrap_for_tweets(self, comp):
        """
        The purpose of this function is to extract at most 10*no_of_tweets tweets that contain the string that is
        the name of the company(company belonging to the the industry, given as the second parameter) and
        that were tweeted between start_date and end_date.

        Parameters
        ----------
        comp: Company object
        no_of_tweets : int

        Returns
        ----------
        scrapped_df : pandas.DataFrame

        In the below link you may find the reference for got.manager.TweetCriteria:
        https://pypi.org/project/GetOldTweets3/
        """
        scrapped_df = pd.DataFrame(
            columns=[
                'Company',
                'Industry',
                'Id',
                'Tweet',
                'Year',
                'Month',
                'Date',
                'No. Of Retweets',
                'No. Of Favorites',
                'Influence Score',
                'Label'])
        for i in range(2):
            tweetCriteria = None
            if i % 2 == 1:
                print("location")
                tweetCriteria = got.manager.TweetCriteria().setQuerySearch(
                    comp.company) .setSince(
                    comp.start_date) .setUntil(
                    comp.end_date) .setNear("Fagaras") .setWithin(
                    "230" +
                    str(i) +
                    "mi") .setMaxTweets(
                    self.no_of_tweets -
                    i)
            else:
                print("no location")
                tweetCriteria = got.manager.TweetCriteria().setQuerySearch(
                    comp.company) .setSince(
                    comp.start_date) .setUntil(
                    comp.end_date) .setMaxTweets(
                    self.no_of_tweets -
                    i)

            """
            Fagaras was chosen as the point of reference because
            the city is located in the geographical center of Romania.
            Also, 250 miles(around 400 kilometers) is enough in order to cover the whole country.
            """
            tweets = None
            while True:
                try:
                    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
                    if tweets is not None:
                        break
                except Exception as e:
                    print("Please wait a little")
                    sleep(1)
                    continue
            for tweet in tweets:
                text = tweet.text
                try:
                    # The tweets starting with 'I'm at" are just tweets where
                    # people tag themselves in a location and do not convey any
                    # message, so they are eliminated
                    common_twitter_string = "I'm at"
                    language = detect(text)
                    if language == "ro" and common_twitter_string not in text[:10]:
                        should_be_added = True
                        if (comp.company in text) == False:
                            continue
                        id = tweet.id
                        if text in scrapped_df['Tweet']:
                            for j in range(len(scrapped_df)):
                                if text == scrapped_df.iloc[j]['Tweet']:
                                    if scrapped_df.iloc[j]['Id'] != id:
                                        should_be_added
                                    else:
                                        should_be_added = False
                                        break
                        if not should_be_added:
                            continue

                        retweets = tweet.retweets
                        favorites = tweet.favorites
                        influence_score = 3 * (retweets + 1) + (favorites + 1)
                        year = tweet.date.year
                        date = ""
                        month = ""
                        if tweet.date.month < 10:
                            date = str(tweet.date.year) + "-0" + \
                                str(tweet.date.month) + "-" + str(tweet.date.day)
                        else:
                            date = str(tweet.date.year) + "-" + \
                                str(tweet.date.month) + "-" + str(tweet.date.day)
                        if tweet.date.month < 10:
                            month = str(tweet.date.year) + "-0" + \
                                str(tweet.date.month)
                        else:
                            month = str(tweet.date.year) + "-" + \
                                str(tweet.date.month)
                        ith_tweet = [
                            comp.company,
                            comp.industry,
                            id,
                            text,
                            year,
                            month,
                            date,
                            retweets,
                            favorites,
                            influence_score,
                            '#']
                        scrapped_df.loc[len(scrapped_df)] = ith_tweet
                except Exception as e:
                    continue
        print(len(scrapped_df))
        return scrapped_df

    def scrap_for_tweets_in_a_industry(self, ind):
        """
        As the name suggests this function is looking for tweets where companies from a certain industry are mentioned
        Parameters
        ----------
        ind : Industry object

        Returns
        ----------
        industry_df : pandas.DataFrame
        """
        industry_df = pd.DataFrame(
            columns=[
                'Company',
                'Industry',
                'Id',
                'Tweet',
                'Year',
                'Date',
                'No. Of Retweets',
                'No. Of Favorites',
                'Influence Score',
                'Label'])
        # Iterating over the list of companies
        for company in ind.list_of_companies:
            # Creating the Company object
            comp = Company(company, ind.industry, ind.start_date, ind.end_date)

            # Scrapping tweets where the company previously created is
            # mentioned
            company_df = self.scrap_for_tweets(comp)

            # Concatenating the dataframes
            frames = [industry_df, company_df]
            result = pd.concat(frames)
            industry_df = result

        return industry_df
