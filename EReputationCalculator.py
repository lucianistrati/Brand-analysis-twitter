
from fasttext import load_model
from copy import deepcopy

import pandas as pd

import warnings

from Company import Company

warnings.filterwarnings("ignore")


class EReputationCalculator:

    def __init__(self):
        pass

    def get_e_reputation(self, ent, prediction_model, scrapped_df, my_dates):
        """
        Calculates the E-reputation matrix for the ent object which can be either a Company or Industry object
        Parameters
        ----------
        ent : Company or Industry object
        prediction_model : Model object
            an object of type Model used for predicting the scrapped tweets and the next elements are
            the paths with the files that have to be loaded in order to use the model
        scrapped_df : pandas.DataFrame
            dataframe that contains tweets about a certain company/industry
            together with infomations regarding the number of retweets & likes
        Returns
        ----------
        scores : list
            matrix shape (n,3) where for each columne, the three rational numbers add up to 1
            and where the first element is the impact of negative tweets, the second element for
            the impact of the neutral tweets and the third element the impact of the positive tweets
        """
        diff = len(my_dates)
        # Creating the list of labels for the barchart plot
        scores = [[0] * 3 for i in range(diff)]
        weights = {}
        weights_sum = {}
        """
        First element of the list "scores" is the number of
        negative tweets where the company/company from the industry is mentioned.
        The second and the third element of this list respectively
        represent the number of neutral and positive tweets in
        which the company is being mentioned.

        Example for the way of calculating the list scores for a company:

        Let's say that we have 60 tweets about this company called "Company A".
        From these 60 tweets: 10 are negative, 10 are neutral and 40 are positive.
        If we would consider just the number of tweets then we could
        say that the tweets about the company are divided like this:
        -> 16.66% is the negative e-reputation;
        -> 16.66% is the neutral e-reputation;
        -> 66.66% is the positive e-reputation.

        This should sum up to 100%.

        However, we will consider the impact of the tweets, not only their number.
        The impact of a tweet is determined after an heuristic function (also called "influence score"),
        which is going to be calculated as: 3 * number_of_retweets + 1 * number_of_likes.

        Now, we will calculate this score for each tweet and then we will sum the scores up
        for each category(negative/neutral/positive) and we will obtain the following sums:
        -> 200 being the sum of the influence scores for the negative tweets;
        -> 50 being the sum of the influence scores for the neutral tweets;
        -> 750 being the sum of the influence scores for the positive tweets;

        Then, we will calculate the total influence scores as sum of the (numbers of tweets per each category) * (the sum of influence score per each category).

        We will get the following number: 10 * 200 + 10 * 50 + 40 * 750 which equals to 32500.

        Then, in order to obtain the list scores we will just scale this for each category:
        -> (10 * 200) / 32500 for the negative tweets;
        -> (10 * 50) / 32500 for the negative tweets;
        -> (50 * 750) / 32500 for the negative tweets;

        Sorting out through the maths and we will have:
        -> 6.15% - negative e-reputation;
        -> 1.53% - neutral e-reputation;
        -> 92.3% - positive e-reputation.

        These e-reputation scores look staggeringly different from the (1/6, 1/6, 4/6) which we would have had
        if we calculated the e-reputation simply based on the number of tweets and completely ignoring
        their impact and their reach on social media.

        In the end the list scores for the company A will look like this: [0.0615, 0.0153, 0.923]
        """
        column_name = ""
        entity_name = ""
        if "Company" in str(type(ent)):
            # Can be either "Company" or "Industry", depends what we want to
            # analyse
            column_name = "Company"
            entity_name = ent.company
        else:
            column_name = "Industry"
            entity_name = ent.industry
        df = scrapped_df
        for i in range(len(df)):
            if df.iloc[i][column_name] == entity_name:
                text = ""

                # Modifying the tweet so that it becomes a string
                if "list" in str(type(df.iloc[i]['Tweet'])):
                    text = ' '.join(df.iloc[i]['Tweet'])
                if "str" in str(type(df.iloc[i]['Tweet'])):
                    text = df.iloc[i]['Tweet']

                # Predicting the sentiment score
                label, confidence_score = prediction_model.predict(text)
                # Labeling the tweet
                if confidence_score > 0.66:
                    if label == "__label__negative":
                        df.at[i, 'Label'] = -1
                    elif label == "__label__positive":
                        df.at[i, 'Label'] = 1
                    else:
                        print(
                            "An unexpected label was given for the tweet " +
                            df.iloc[i]['Tweet'])
                else:
                    df.at[i, 'Label'] = 0

                # Creating the weights and weigts_sum dictionaries weighted
                # after the influence score
                influence_score = df.iloc[i]['Influence Score']
                if df.iloc[i]['Month'] not in weights_sum.keys():
                    weights_sum[df.iloc[i]['Month']] = 0.0
                    weights_sum[df.iloc[i]['Month']] += influence_score
                else:
                    weights_sum[df.iloc[i]['Month']] += influence_score
                if df.iloc[i]['Month'] not in weights.keys():
                    weights[df.iloc[i]['Month']] = [0.0] * 3
                    weights[df.iloc[i]['Month']][df.iloc[i]
                                                 ['Label'] + 1] += influence_score
                else:
                    weights[df.iloc[i]['Month']][df.iloc[i]
                                                 ['Label'] + 1] += influence_score

        # Inputting the dictionaries with null values where no tweeets were
        # found
        for date in my_dates:
            if date not in weights.keys():
                weights[date] = [0.0] * 3
            if date not in weights_sum.keys():
                weights_sum[date] = 0.0

        # Determining the influence score matrix
        for j in range(diff):
            for i in range(3):
                if weights_sum[my_dates[j]] == 0:
                    scores[j][i] = 0.0
                else:
                    scores[j][i] = weights[my_dates[j]][i] / \
                        weights_sum[my_dates[j]]
        return scores

    def get_monthly_dates(self, ent):
        """
        Create a list of stringS where each element is a string in the format 'YYYY-MM'
        with the first element of the list being the first month included in the analysis
        and the last element being the last month included in the analysis
        Parameters
        ----------
        ent : Company or Industry object

        Returns
        ----------
        my_dates : list
        """
        start_year = int(ent.start_date[:4])
        end_year = int(ent.end_date[:4])
        start_month = int(ent.start_date[5:7])
        end_month = int(ent.end_date[5:7])
        diff = 0
        if end_year == start_year:
            diff = end_month - start_month + 1
        else:  # end_year!=start_year
            diff = (13 - start_month) + (end_year -
                                         start_year - 1) * 12 + (end_month)
        dates = {}
        month = ""
        # Creating the list of labels for the barchart plot
        my_dates = []
        if end_year != start_year:
            for i in range(start_month, 13):
                if i < 10:
                    month = str(start_year) + '-0' + str(i)
                else:
                    month = str(start_year) + '-' + str(i)
                my_dates.append(month)
        if end_year - start_year >= 2:
            for i in range(start_year + 1, end_year):
                for j in range(1, 13):
                    if j < 10:
                        month = str(i) + '-0' + str(j)
                    else:
                        month = str(i) + '-' + str(j)
                    my_dates.append(month)
        if end_year != start_year:
            for i in range(1, end_month + 1):
                if i < 10:
                    month = str(end_year) + '-0' + str(i)
                else:
                    month = str(end_year) + '-' + str(i)
                my_dates.append(month)
        if end_year == start_year:
            for i in range(start_month, end_month + 1):
                if i < 10:
                    month = str(end_year) + '-0' + str(i)
                else:
                    month = str(end_year) + '-' + str(i)
                my_dates.append(month)
        return my_dates

    def get_e_reputation_industry(
            self,
            ind,
            prediction_model,
            ind_df,
            my_dates):
        """
        Creating a dictionary that maps the name of all companies in an industry to their own E-reputation matrix
        Parameters
        ----------
        ind : Industry object
        prediction_model : Model object
        ind_df : pandas.DataFrame

        Returns
        ----------
        dict_comp_scores : dict
            dictionary that maps a company to the scores matrix
        """
        industry_df = deepcopy(ind_df)
        dict_comp_scores = {}
        companies = list(set(industry_df['Company']))
        for i in range(len(companies)):
            df = pd.DataFrame(columns=industry_df.columns)
            for j in range(len(industry_df)):
                if industry_df.iloc[j]['Company'] == companies[i]:
                    df.loc[len(df)] = industry_df.iloc[j]
            df['new_index'] = range(len(df))
            df = df.set_index('new_index')
            comp = Company(
                companies[i],
                ind.industry,
                ind.start_date,
                ind.end_date)
            scores = self.get_e_reputation(
                comp, prediction_model, df, my_dates)
            dict_comp_scores[companies[i]] = scores
        return dict_comp_scores

    def transpose_multi_year_scores_matrix(self, ent):
        """
        Below we are making the transpose of the matrix of scores for the company and the industry
        From the shape (n, 3) to (3, n)
        Parameters
        ----------
        ent : Company or Industry object

        Returns
        ----------
        aux : list of shape (3, n)
        """
        w = 3
        start_year = int(ent.start_date[:4])
        end_year = int(ent.end_date[:4])
        start_month = int(ent.start_date[5:7])
        end_month = int(ent.end_date[5:7])
        h = 0
        if end_year == start_year:
            h = end_month - start_month + 1
        else:  # end_year!=start_year
            h = (13 - start_month) + (end_year -
                                      start_year - 1) * 12 + (end_month)

        aux = [[0 for x in range(h)] for y in range(w)]
        for i in range(len(ent.multi_year_scores)):
            aux[0][i] = ent.multi_year_scores[i][0]
            aux[1][i] = ent.multi_year_scores[i][1]
            aux[2][i] = ent.multi_year_scores[i][2]
        return aux

    def calculate_yearly_averaged_scores(self, ent):
        """
        This method create a list of three elements weighted
        for each label(negative/neutral/positive) for every
        month of the analysing period
        Parameters
        ----------
        ent : Company or Industry object

        Returns
        ----------
        ent.yearly_averaged_scores : list of shape (1,3)
        """
        ent.yearly_averaged_scores = [0.0] * 3
        start_year = int(ent.start_date[:4])
        end_year = int(ent.end_date[:4])
        start_month = int(ent.start_date[5:7])
        end_month = int(ent.end_date[5:7])
        h = 0
        if end_year == start_year:
            h = end_month - start_month + 1
        else:  # end_year!=start_year
            h = (13 - start_month) + (end_year -
                                      start_year - 1) * 12 + (end_month)
        scores_sum = 0.0
        for i in range(3):
            for j in range(len(ent.multi_year_scores[i])):
                ent.yearly_averaged_scores[i] += ent.multi_year_scores[i][j]
                scores_sum += ent.multi_year_scores[i][j]
        for i in range(3):
            ent.yearly_averaged_scores[i] /= scores_sum
        return ent.yearly_averaged_scores
