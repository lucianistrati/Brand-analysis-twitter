import squarify
from TweetPreprocessor import TweetPreprocessor
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud
from PIL import Image

import pandas as pd
import numpy as np


import warnings
warnings.filterwarnings("ignore")


class Visualizer:
    def __init__(self):
        """
        Class used for plotting the data for the reports
        """
        self.stop_words_nltk = np.array(list(set(stopwords.words('romanian'))))

    def plot_industry_treemap(
            self,
            industry_name,
            industry_df,
            company_to_score,
            my_plot):
        """
        Parameters
        ----------
        industry_name : string
        industry_df : pandas.DataFrame
        company_to_score : dict
            maps the name of a company to a list of size 3 of e-reputation scores
        my_plot : matplotlib.axes._subplots.AxesSubplot

        Returns
        ----------
        my_plot : matplotlib.axes._subplots.AxesSubplot

        This method will be able to plot a TreeMap for a certain industry with some companies in it
        and alongside the TreeMap it will plot a ranking of the industries after the biggest score they obtain.
        Given that for each company we will know a list scores that has 3 elements:
        ->first element is the weight of negative tweets for that company;
        ->second element is the weight of neutral tweets for that company;
        ->third element is the weight of positive tweets for that company.
        The sum of all these weights should be 1.
        We will color the rectangle that belongs to a company in this way:
        {
        Red : scores[0] * 256 (the impact of negative tweets scaled to 256)
        Blue : 0,
        Green : scores[2] * 256 (the impact of positive tweets scaled to 256)
        }
        The score for which we will classify the e-reputation of the company within its industry will be:
        scores[2] - scores[0] (the impact of positive tweets minus the impact of negative tweets).(A)
        Know that we know how the color is supposed to look like, the following important step is
        to know how big should be its rectangle compared to the whole treemap. In order to have
        a pretty fair measure of popularity of a company within its industry, the corresponding rectangle in the
        treemap should match the fraction: (number of tweets about the company)/(number of tweets about the industry).

        On the right side of the Treemap a list of companies will be displayed according to score mentioned in the paragraph A.
        """
        total_num_of_tweets = len(industry_df)
        company_to_tweets_percentage = {}
        sizes = []
        colors = []

        # Creating a dictionary that stores for each company the proportion of tweets where the company is mentioned compared to the rest of the tweets found for the industry
        # This dictionary helps us decide the size of the square for each
        # company represented in the TreeMap
        for i in range(len(industry_df)):
            if industry_df.iloc[i]['Company'] not in company_to_tweets_percentage.keys(
            ):
                company_to_tweets_percentage[industry_df.iloc[i]
                                             ['Company']] = 1
            else:
                company_to_tweets_percentage[industry_df.iloc[i]
                                             ['Company']] += 1
        for company_name in company_to_tweets_percentage.keys():
            company_to_tweets_percentage[company_name] = company_to_tweets_percentage[company_name] / \
                total_num_of_tweets

        # Determining the companies present in the industry for which any
        # tweets were found
        companies = list(company_to_tweets_percentage.keys())
        N = len(companies)

        # Creating an dictionary where for each company we know its brand score
        # This dictionary helps us when choosing the color for the square of each company
        # The more green it is, better its brand score, the more red it is,
        # worse its brand score
        yearly_average_company_to_score = {}
        for comp in company_to_score.keys():
            scores_list = company_to_score[comp]
            yearly_average_scores = [0.0] * 3
            sum_scores = 0.0
            for i in range(len(scores_list)):
                for j in range(3):
                    yearly_average_scores[j] += scores_list[i][j]
                    sum_scores += scores_list[i][j]
            for j in range(3):
                yearly_average_scores[j] /= sum_scores
            yearly_average_company_to_score[comp] = yearly_average_scores
        company_to_score = yearly_average_company_to_score

        # Creating the list of colors and sizes
        company_neg_pos_scores = []
        for i in range(N):
            # If we found no tweets about a company in that certain period than
            # we will skip it
            if companies[i] not in company_to_tweets_percentage.keys():
                continue
            sizes.append(company_to_tweets_percentage[companies[i]])
            # Setting the neutral score to zero in order to prevent it from
            # influencing the final color
            company_to_score[companies[i]][1] = 0
            # Switching the green with blue in the order of the list
            tup = (companies[i], company_to_score[companies[i]]
                   [1] - company_to_score[companies[i]][0])
            company_neg_pos_scores.append(tup)
            if company_to_score[companies[i]][2] - \
                    company_to_score[companies[i]][0] > 0.3:
                company_to_score[companies[i]][2] = max(
                    1.0, (1.0 + company_to_score[companies[i]][2]) / 2)
            elif company_to_score[companies[i]][2] - company_to_score[companies[i]][0] < -0.3:
                company_to_score[companies[i]][0] = max(
                    1.0, (1.0 + company_to_score[companies[i]][0]) / 2)
            elif company_to_score[companies[i]][2] - company_to_score[companies[i]][0] > -0.3 and company_to_score[companies[i]][2] - company_to_score[companies[i]][0] <= 0.0:
                company_to_score[companies[i]][0] = max(
                    1.0, (0.6 + company_to_score[companies[i]][0]) / 2)
            else:
                company_to_score[companies[i]][2] = max(
                    1.0, (0.6 + company_to_score[companies[i]][2]) / 2)
            company_to_score[companies[i]][1], company_to_score[companies[i]
                                                                ][2] = company_to_score[companies[i]][2], company_to_score[companies[i]][1]
            colors.append(tuple(company_to_score[companies[i]]))
        # Sorting the final scores descendingly, using these final scores which
        # we will make the top later
        company_neg_pos_scores = sorted(
            company_neg_pos_scores, key=lambda x: (-1) * x[1])

        # Creating the TreeMap
        squarify.plot(sizes=sizes, label=companies, color=colors)
        my_plot.axis('off')
        my_plot.set_title(
            "Treemap of " +
            industry_name +
            " industry by no. of tweets")
        return my_plot, company_neg_pos_scores

    def plot_industry_top(
            self,
            industry_name,
            company_neg_pos_scores,
            my_plot):
        """
        This functions iterates in the dictionary company_neg_pos_scores and is creating the list of top 10 companies in the industry,
        or less than 10 if there are not that many companies in that industry. The scores are discretized in intervals of the form
        of (a-0.4,a] where a belongs to [-0.6,-0.2,0.2,0.6,1.0](with -1.0 belonging to the first interval). For each interval we assign
        the labels ['very bad','bad','neutral','good','very good']
        Parameters
        ----------
        industry_name : string
            name of the industry for which the top will be
        company_neg_pos_scores : dictionary
            dictionary that contains pairs of the form company-name:score
            where company-name is string and score is float
        my_plot : matplotlib.axes._subplots.AxesSubplot

        Returns
        ----------
        my_plot : matplotlib.axes._subplots.AxesSubplot
        """
        textstr = "Top companies in " + industry_name + \
            "\n industry by E-reputation score:\n\n"
        i = 1
        for comp, score in company_neg_pos_scores:
            if i == 11:
                break
            if score > 0.6:
                textstr += str(i) + "." + comp + ": very good\n"
            elif score > 0.2:
                textstr += str(i) + "." + comp + ": good\n"
            elif score > -0.2:
                textstr += str(i) + "." + comp + ": neutral\n"
            elif score > -0.6:
                textstr += str(i) + "." + comp + ": bad\n"
            else:
                textstr += str(i) + "." + comp + ": very bad\n"
            i += 1

        font = {'family': 'serif',
                'color': 'black',
                'weight': 'normal',
                'size': 9,
                }
        my_plot.text(
            0.5,
            0.5,
            textstr,
            horizontalalignment='center',
            verticalalignment='center',
            fontdict=font,
            transform=my_plot.transAxes)
        my_plot.axis('off')

        return my_plot

    def plot_barchart_tweet_distribution(self, comp, df, my_plot):
        """
        Parameters
        ----------
        comp : Company object
        df : pandas.DataFrame
        my_plot : matplotlib.axes._subplots.AxesSubplot

        Returns
        ----------
        my_plot : matplotlib.axes._subplots.AxesSubplot
        """
        start_year = int(comp.start_date[:4])
        start_month = int(comp.start_date[5:7])
        end_year = int(comp.end_date[:4])
        end_month = int(comp.end_date[5:7])
        dates = {}
        for i in range(len(df)):
            key = df.loc[i]['Date'][:7]
            if key in dates.keys():
                dates[key] += 1
            else:
                dates[key] = 1
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
                if month not in dates.keys():
                    dates[month] = 0
        if end_year - start_year >= 2:
            for i in range(start_year + 1, end_year):
                for j in range(1, 13):
                    if j < 10:
                        month = str(i) + '-0' + str(j)
                    else:
                        month = str(i) + '-' + str(j)
                    my_dates.append(month)
                    if month not in dates.keys():
                        dates[month] = 0
        if end_year != start_year:
            for i in range(1, end_month + 1):
                if i < 10:
                    month = str(end_year) + '-0' + str(i)
                else:
                    month = str(end_year) + '-' + str(i)
                my_dates.append(month)
                if month not in dates.keys():
                    dates[month] = 0
        if end_year == start_year:
            for i in range(start_month, end_month + 1):
                if i < 10:
                    month = str(end_year) + '-0' + str(i)
                else:
                    month = str(end_year) + '-' + str(i)
                my_dates.append(month)
                if month not in dates.keys():
                    dates[month] = 0
        reordered_dates = {k: dates[k] for k in my_dates}

        my_plot.bar(*zip(*reordered_dates.items()))
        s = 0
        for value in dates.values():
            s += value
        average_tweet_volume = s / len(dates.keys())
        my_plot.axhline(
            y=average_tweet_volume,
            color='purple',
            label="Average monthly tweet volume for " +
            comp.company)
        labels = ["Average monthly tweet volume for " + comp.company]
        handles, _ = my_plot.get_legend_handles_labels()
        my_plot.legend(handles=handles[1:], labels=labels, loc='best')

        return my_plot

    def pretty_formatter(self, text):
        """
        Parameters
        ----------
        text : string

        Returns
        ----------
        text : string
            well formatted, 25 character per row or slightly a little more than 25
        """
        n = 25
        s = ""
        newline = False
        for i in range(len(text)):
            if newline and text[i] == ' ':
                s += '\n'
                newline = False
            s += text[i]
            if i % n == 0 and i != 0:
                if text[i] == ' ':
                    s += '\n'
                    newline = False
                else:
                    newline = True

        return s

    def plot_most_popular_tweet(self, ent, scrapped_df, my_plot):
        """
        Parameters
        ----------
        ent : Company or Industry object
        scrapped_df : pandas.DataFrame
        my_plot : matplotlib.axes._subplots.AxesSubplot

        Returns
        ----------
        my_plot : matplotlib.axes._subplots.AxesSubplot
        """
        # Determining the tweet with the highest influence score
        max_score = -1
        idx = -1
        for i in range(len(scrapped_df)):
            if scrapped_df.iloc[i]['Influence Score'] > max_score:
                max_score = scrapped_df.iloc[i]['Influence Score']
                idx = i

        # Saving the row of the found tweet together with some additional
        # information
        record = scrapped_df.iloc[idx]
        if 'Industry' in str(type(ent)):
            textstr = "The most popular tweet in the " + ent.industry + " industry is: " + record['Tweet'] + " with " + str(
                record['No. Of Retweets']) + " retweets and " + str(record['No. Of Favorites']) + " likes."
        elif 'Company' in str(type(ent)):
            textstr = "The most popular tweet where the company " + ent.company + " is mentioned is: " + record[
                'Tweet'] + " with " + str(record['No. Of Retweets']) + " retweets and " + str(
                record['No. Of Favorites']) + " likes."

        # Defining the custom font for the text that will be displayed
        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 10,
                }

        # Formatting the string in order to make it look centered and then
        # plotting it
        textstr = self.pretty_formatter(textstr)
        my_plot.text(
            0.5,
            0.5,
            textstr,
            horizontalalignment='center',
            verticalalignment='center',
            fontdict=font,
            transform=my_plot.transAxes)
        my_plot.axis('off')

        return my_plot

    def plot_wordcloud(self, ent, scrapped_df, my_plot):
        """
        This function creates a wordcloud with the most frequent words from a dataframe about a company while excluding the words from the list final_sws.
        Parameters
        ----------
        ent : Company or Industry object
        scrapped_df : pandas.DataFrame
            dataframe that contains tweets about a certain company/industry
            together with informations regarding the number of retweets & likes
        my_plot : matplotlib.axes._subplots.AxesSubplot

        Returns
        ----------
        my_plot : matplotlib.axes._subplots.AxesSubplot

        The order of the scores is the following: negative, neutral and positive.
        """
        # Saving the generated WordClouds in the following files
        path = ""
        text = ""
        colors = ['Reds', 'Greys', 'Greens']
        idx = 0
        max = ent.yearly_averaged_scores[0]
        for i in range(1, len(ent.yearly_averaged_scores)):
            if ent.yearly_averaged_scores[i] > max:
                max = ent.yearly_averaged_scores[i]
                idx = i

        for i in range(len(scrapped_df)):
            text += scrapped_df.iloc[i]['Tweet']
            if 'Company' in str(type(ent)):
                path = "Data/WordClouds/Companies/" + ent.company + "_company_word_cloud.png"
            else:
                path = "Data/WordClouds/Industries/" + ent.industry + "_industry_word_cloud.png"

        # Building a custom "aggresive" stop-words list in order to prevent
        # these words from appearing in the wordcloud
        tp = TweetPreprocessor()
        custom_stop_words = tp.read_list_of_stop_words()
        companies_list = tp.read_list_of_companies()
        final_sws = np.concatenate(
            (custom_stop_words, companies_list, self.stop_words_nltk), axis=0)

        # Creating the WordCloud
        image1 = Image.open("wordcloud_mask.jpg")
        graph = np.array(image1)
        wordcloud = WordCloud(
            width=600,
            height=200,
            stopwords=final_sws,
            background_color='white',
            collocations=False,
            colormap=colors[idx],
            mask=graph).generate(text)
        my_plot.imshow(wordcloud, interpolation='bilinear')
        my_plot.axis("off")

        wordcloud.to_file(path)

        return my_plot

    def plot_piechart_with_scores(self, ent, my_plot):
        """
        Creates a piechart with 3 labels: Negative Tweets, Neutral Tweets and Positive Tweets, each with an according color.
        The size of one pie from the piechart corresponds to the Influence Score Weighted Average that those tweet have together.
        Parameters
        ----------
        ent : Company or Industry object
        my_plot : matplotlib.axes._subplots.AxesSubplot

        Returns
        ----------
        my_plot : matplotlib.axes._subplots.AxesSubplot
        """
        labels = ['Negative Tweets', 'Neutral Tweets', 'Positive Tweets']
        colors = ['red', 'gray', 'green']
        """
        Red coresponds to the negative tweets, gray to the neutral ones and green to the positive in the piechart.
        """
        explode = (0.1, 0.1, 0.1)
        """
        The explode parameters make the pie look more dynamic
        with its components being distanced from the center of the piechart.
        """
        my_plot.pie(ent.yearly_averaged_scores,
                    explode=explode,
                    labels=labels,
                    colors=colors,
                    startangle=90,
                    autopct='%.1f%%',
                    shadow=True)
        if "company" in str(type(ent)):
            my_plot.set_title(ent.company)
        else:
            my_plot.set_title(ent.industry)
        return my_plot

    def plot_companies_linechart_with_scores(
            self, first_comp, second_comp, labels, my_plot):
        """
        Parameters
        ----------
        first_company : Company object
        second_comp : Company object
        labels : list
        my_plot : matplotlib.axes._subplots.AxesSubplot

        Returns
        ----------
        my_plot : matplotlib.axes._subplots.AxesSubplot
        """
        start_year = int(first_comp.start_date[:4])
        end_year = int(first_comp.end_date[:4])
        start_month = int(first_comp.start_date[5:7])
        end_month = int(first_comp.end_date[5:7])
        h = 0
        if end_year == start_year:
            h = end_month - start_month + 1
        else:  # end_year!=start_year
            h = (13 - start_month) + (end_year -
                                      start_year - 1) * 12 + (end_month)

        df = pd.DataFrame({'x': range(h),
                           'y_comp_1_neg': first_comp.multi_year_scores[0],
                           'y_comp_1_neu': first_comp.multi_year_scores[1],
                           'y_comp_1_pos': first_comp.multi_year_scores[2],
                           'y_comp_2_neg': second_comp.multi_year_scores[0],
                           'y_comp_2_neu': second_comp.multi_year_scores[1],
                           'y_comp_2_pos': second_comp.multi_year_scores[2]})

        my_plot.plot(
            'x',
            'y_comp_1_neg',
            data=df,
            marker='o',
            markerfacecolor='darkred',
            markersize=12,
            color='darkred',
            linewidth=4,
            label=first_comp.company +
            " negative tweets ")
        my_plot.plot(
            'x',
            'y_comp_1_neu',
            data=df,
            marker='o',
            markerfacecolor='black',
            markersize=12,
            color='black',
            linewidth=4,
            label=first_comp.company +
            " neutral tweets ")
        my_plot.plot(
            'x',
            'y_comp_1_pos',
            data=df,
            marker='o',
            markerfacecolor='darkgreen',
            markersize=12,
            color='darkgreen',
            linewidth=4,
            label=first_comp.company +
            " positive tweets ")
        my_plot.plot(
            'x',
            'y_comp_2_neg',
            data=df,
            marker='o',
            markerfacecolor='red',
            markersize=12,
            color='red',
            linewidth=4,
            label=second_comp.company +
            " negative tweets ")
        my_plot.plot(
            'x',
            'y_comp_2_neu',
            data=df,
            marker='o',
            markerfacecolor='darkgrey',
            markersize=12,
            color='darkgrey',
            linewidth=4,
            label=second_comp.company +
            " neutral tweets ")
        my_plot.plot(
            'x',
            'y_comp_2_pos',
            data=df,
            marker='o',
            markerfacecolor='lime',
            markersize=12,
            color='lime',
            linewidth=4,
            label=second_comp.company +
            " positive tweets ")
        my_plot.legend(loc="best", borderaxespad=0)
        my_plot.set_xlabel("Month")
        my_plot.set_ylabel("Percentage of tweets")
        return my_plot

    def plot_company_linechart_with_scores(self, ent, labels, my_plot):
        """
        This function creates a plot with a timeframe based on months where for each month we know the E-Reputation Scores for negative, neutral and positive tweets.
        Knowing these values for each month allows us to understand how the E-Reputation evolved monthly in the analysing period.
        Parameters
        ----------
        ent : Company or Industry object
        labels : list
        my_plot : matplotlib.axes._subplots.AxesSubplot

        Returns
        ----------
        my_plot : matplotlib.axes._subplots.AxesSubplot
        The purpose of this method is to allow you to visualize the following cases:
        1.The e-reputation trend of an industry
        2.The e-reputation trend of a company
        """
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
        print(labels)
        print(ent.multi_year_scores)
        df = pd.DataFrame({'x': labels,
                           'y_neg': ent.multi_year_scores[0],
                           'y_neu': ent.multi_year_scores[1],
                           'y_pos': ent.multi_year_scores[2], })
        label = ""
        if "Company" in str(type(ent)):
            label = ent.company
        else:
            label = ent.industry
        my_plot.plot(
            'x',
            'y_neg',
            data=df,
            marker='o',
            markerfacecolor='red',
            markersize=12,
            color='red',
            linewidth=4,
            label=label +
            " negative tweets ")
        my_plot.plot(
            'x',
            'y_neu',
            data=df,
            marker='o',
            markerfacecolor='darkgrey',
            markersize=12,
            color='darkgrey',
            linewidth=4,
            label=label +
            " neutral tweets ")
        my_plot.plot(
            'x',
            'y_pos',
            data=df,
            marker='o',
            markerfacecolor='lime',
            markersize=12,
            color='lime',
            linewidth=4,
            label=label +
            " positive tweets ")
        my_plot.legend(loc="best", borderaxespad=0)
        my_plot.set_xlabel("Month")
        # my_plot.set_xticks(labels)
        my_plot.set_ylabel("Percentage of tweets")
        return my_plot, df
