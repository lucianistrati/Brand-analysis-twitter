import time
import os
import io
from EReputationCalculator import EReputationCalculator
from matplotlib.backends.backend_pdf import PdfPages
from TweetPreprocessor import TweetPreprocessor
from TweetScrapper import TweetScrapper
from matplotlib import pyplot as plt
from Visualizer import Visualizer
from Industry import Industry
from Company import Company
from copy import deepcopy
from Model import Model
from PyPDF2 import PdfFileWriter, PdfFileReader

import numpy as np

import warnings
warnings.filterwarnings("ignore")


NUMBER_OF_TWEETS = 100
COMPANIES_COUNTER = 0
INDUSTRIES_COUNTER = 0
A4_PORTRAIT_MEASUREMENTS = (8.3, 11.7)
A4_DOUBLE_PORTRAIT_MEASUREMENTS = (8.3, 23.4)
A4_LANDSCAPE_MEASUREMENTS = (11.7, 8.3)


def get_file_counters():
    global COMPANIES_COUNTER, INDUSTRIES_COUNTER
    if 'Brand-analysis-master' not in str(os.getcwd()):
        COMPANIES_COUNTER, INDUSTRIES_COUNTER = np.load(
            "Brand-analysis-master/FILE_COUNTERS.npy")
    else:
        COMPANIES_COUNTER, INDUSTRIES_COUNTER = np.load("FILE_COUNTERS.npy")


def update_file_counters():
    global COMPANIES_COUNTER, INDUSTRIES_COUNTER
    a = np.array([COMPANIES_COUNTER, INDUSTRIES_COUNTER])
    np.save("FILE_COUNTERS.npy", a)


def merger(output_path, input_paths):
    pdf_writer = PdfFileWriter()
    for path in input_paths:
        pdf_reader = PdfFileReader(path)
        for page in range(pdf_reader.getNumPages()):
            pdf_writer.addPage(pdf_reader.getPage(page))
    with open(output_path, 'wb') as fh:
        pdf_writer.write(fh)


def plot_dispatcher(dict_form):
    global COMPANIES_COUNTER, INDUSTRIES_COUNTER
    get_file_counters()
    """
    Receives the options selected by user and performs one out of 4 options according to what the user wants
    Parameters
    ----------
    dict_form : ImmutableMultiDict
        this dictionary contains the fields that specify whether the user wants to analyse a single tweet or wants to perform an analysis,
        also for an analysis it is specified the period of interest

    Returns
    ----------
    tuple : tuple
        can be either:
            -2,-2,-2,-2 -> in case of error
            -1,-1,-1,-2 -> in case of finding not enough tweets in order to perform a relevant analysis
            COMPANIES_COUNTER, bytes_image, len(df), entity_name -> int, _io.BytesIO, int, string-> the file counter for companies, the image that will be displayed further in the API, the number of tweets analysed, name of the company
            INDUSTRIES_COUNTER, bytes_image, len(df), entity_name -> int, _io.BytesIO, int, string -> the file counter for industries, the image that will be displayed further in the API, the number of tweets analysed, name of the industry
            COMPANIES_COUNTER, bytes_image, len(df), entity_name -> int, _io.BytesIO, int, string-> the file counter for companies, the image that will be displayed further in the API, the number of tweets analysed, name of the companies
    
    Plotting legend:
    # - empty space
    0 - Barchart tweet distribution
    1 - Piechart
    2 - WordCloud
    3 - LineChart
    4 - Treemap
    5 - Industry top
          
    Company plots - Gridspec sizes (12,4):
    0000
    0000
    ####
    3333
    3333
    ####
    #11#
    #11#
    #11#
    #11#
    2222
    2222
            
    Industry plots - Gridspec sizes (9,5):
    11#55
    11#55
    33333
    33333
    #####
    44444
    44444
    22222
    22222 

    Company-comparison plots - Gridspec sizes (30,4):
    0000
    0000
    0000
    ####
    3333
    3333
    3333
    ####
    0000
    0000
    0000
    ####
    3333
    3333
    3333
    ####
    1111
    1111
    1111
    1111
    1111
    1111
    ####
    1111
    1111
    1111
    1111
    1111
    1111
    """
    model = Model('FastText', 'model_tweets-0.99-0.01-0.0.bin')
    viz = Visualizer()
    preprocessor = TweetPreprocessor(allow_stemming=True)
    erep_calc = EReputationCalculator()
    bytes_image = None
    if "tweet-analysis" in dict_form.keys():
        text = dict_form['tweet-analysis']
        text = preprocessor.preprocess_tweet(text)
        text = " ".join(text)
        # Predicting the sentiment score
        label, confidence_score = model.predict(text)
        confidence_score = int(confidence_score * 100)
        if label == '__label__positive':
            return confidence_score, 0, 0, 0
        if label == '__label__negative':
            return (-1) * confidence_score, 0, 0, 0
    elif "company" in dict_form.keys():
        scrapper = TweetScrapper(1000)
        # Getting the options selected by the user
        company = dict_form['company']
        industry = dict_form['industry-name']
        start_date = dict_form['first_day']
        end_date = dict_form['last_day']

        comp = Company(company, industry, start_date, end_date)

        my_dates = erep_calc.get_monthly_dates(comp)

        # Performing the tweet scraping
        df = scrapper.scrap_for_tweets(comp)

        print(len(df))
        # If less than 30 tweets resulted than we don't have a representative
        # enough amount of tweets in order to perform any analysis
        if len(df) < 30:
            return -1, -1, -1, -1  # Not enough tweets

        # Making a copy of the dataframe in its raw form
        df_raw = deepcopy(df)
        # Saving the raw dataframe
        df_raw.to_csv("Scrapped dataframes/Companies/"+
            comp.company +
            "_comp_" +
            str(COMPANIES_COUNTER) +
            "_raw.csv")

        # Preprocessing stage
        df = preprocessor.preprocess_dataframe(df)

        # Determining the e-reputation scores
        comp.multi_year_scores = erep_calc.get_e_reputation(
            comp, model, df, my_dates)

        # Saving the dataframe after preprocessing and labeling
        df.to_csv("Scrapped dataframes/Companies/"+
            comp.company +
            "_comp_" +
            str(COMPANIES_COUNTER) +
            "_processed.csv")

        # Preprocessing the scores in order to create some plots
        comp.multi_year_scores = erep_calc.transpose_multi_year_scores_matrix(
            comp)
        comp.yearly_averaged_scores = erep_calc.calculate_yearly_averaged_scores(
            comp)
        comp.ereputation_score = sum(comp.yearly_averaged_scores) / 3

        # Designing the grid for the plots
        fig = plt.figure(figsize=A4_PORTRAIT_MEASUREMENTS)
        gs = fig.add_gridspec(12, 4)
        plt.axis('off')

        # Creating the piechart
        piechart = fig.add_subplot(gs[6:10, 1:3])
        piechart = viz.plot_piechart_with_scores(comp, piechart)
        piechart.plot()

        # Creating the wordcloud
        wordcloud = fig.add_subplot(gs[10:12, :])
        wordcloud = viz.plot_wordcloud(comp, df_raw, wordcloud)
        wordcloud.plot()

        # Inserting the first month
        start_year = int(comp.start_date[:4])
        start_month = int(comp.start_date[5:7])
        end_year = int(comp.end_date[:4])
        end_month = int(comp.end_date[5:7])
        if start_month == 1:
            start_month = 12
            start_year -= 1
        else:
            start_month -= 1
        """
                if start_month < 10:
                    my_dates.insert(0, str(start_year) + '-0' + str(start_month))
                else:
                    my_dates.insert(0, str(start_year) + '-' + str(start_month))
                """

        # Creating the linechart while making sure that we are not creating a
        # monthly plot if the analysis period is one month or less
        if start_month != end_month or (
                start_month == end_month and start_year != end_year):
            linechart = fig.add_subplot(gs[3:5, :])
            linechart, linechart_df = viz.plot_company_linechart_with_scores(
                comp, my_dates, linechart)
            linechart.plot()

        # Creating the barchart
        barchart = fig.add_subplot(gs[0:2, :])
        barchart = viz.plot_barchart_tweet_distribution(comp, df, barchart)
        barchart.plot()

        # Rotating the x-labels in order to make them look properly on the
        # barchart and linechart
        for ax in fig.axes:
            plt.sca(ax)
            plt.xticks(rotation=30)

        plt.title(company + " report\n", size=24)
        fig.subplots_adjust(wspace=0.75)
        # Updating file counters
        COMPANIES_COUNTER += 1
        update_file_counters()

        # Saving the plot as a pdf file
        pp = PdfPages(
            "App/static/" +
            company +
            "-report-no." +
            str(COMPANIES_COUNTER) +
            ".pdf")
        pp.savefig(fig)
        pp.close()

        # Saving the plot in the as an _io.BytesIO object
        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)

        return COMPANIES_COUNTER, bytes_image, len(df), comp.company
    elif "company-comparison-first" in dict_form.keys():
        scrapper = TweetScrapper(1000)
        # Getting the options selected by the user
        first_company = dict_form['company-comparison-first']
        second_company = dict_form['company-comparison-second']
        start_date = dict_form['first_day']
        end_date = dict_form['last_day']

        # Creating the first_compny objects
        first_comp = Company(first_company, "", start_date, end_date)
        second_comp = Company(second_company, "", start_date, end_date)

        my_dates = erep_calc.get_monthly_dates(first_comp)

        # Performing the tweet scraping
        df_1 = scrapper.scrap_for_tweets(first_comp)
        df_2 = scrapper.scrap_for_tweets(second_comp)

        # If less than 30 tweets resulted than we don't have a representative
        # enough amount of tweets in order to perform any analysis
        if len(df_1) + len(df_2) < 30:
            return -1, -1, -1, -1  # Not enough tweets

        # Saving the raw dataframes
        df_1.to_csv("Scrapped dataframes/Companies/"+
            first_comp.company +
            "_comp_" +
            str(COMPANIES_COUNTER) +
            "_raw.csv")
        df_2.to_csv("Scrapped dataframes/Companies/"+
            second_comp.company +
            "_comp_" +
            str(COMPANIES_COUNTER) +
            "_raw.csv")

        # Preprocessing stage
        df_1 = preprocessor.preprocess_dataframe(df_1)
        df_2 = preprocessor.preprocess_dataframe(df_2)

        # Determining the e-reputation scores
        first_comp.multi_year_scores = erep_calc.get_e_reputation(
            first_comp, model, df_1, my_dates)
        second_comp.multi_year_scores = erep_calc.get_e_reputation(
            second_comp, model, df_2, my_dates)

        # Saving the dataframe after preprocessing and labeling
        df_1.to_csv("Scrapped dataframes/Companies/"+
            first_comp.company +
            "_comp_" +
            str(COMPANIES_COUNTER) +
            "_processed.csv")
        df_2.to_csv("Scrapped dataframes/Companies/"+
            second_comp.company +
            "_comp_" +
            str(COMPANIES_COUNTER) +
            "_processed.csv")

        # Preprocessing the scores in order to create some plots
        first_comp.multi_year_scores = erep_calc.transpose_multi_year_scores_matrix(
            first_comp)
        first_comp.yearly_averaged_scores = erep_calc.calculate_yearly_averaged_scores(
            first_comp)
        first_comp.ereputation_score = sum(
            first_comp.yearly_averaged_scores) / 3
        second_comp.multi_year_scores = erep_calc.transpose_multi_year_scores_matrix(
            second_comp)
        second_comp.yearly_averaged_scores = erep_calc.calculate_yearly_averaged_scores(
            second_comp)
        second_comp.ereputation_score = sum(
            second_comp.yearly_averaged_scores) / 3

        # Designing the grid for the plots
        fig = plt.figure(figsize=A4_DOUBLE_PORTRAIT_MEASUREMENTS)
        gs = fig.add_gridspec(30, 4)
        plt.title(
            first_comp.company +
            "-VS-" +
            second_comp.company +
            " report\n",
            size=24)
        plt.axis('off')

        # Creating the barcharts
        barchart_1 = fig.add_subplot(gs[0:3, :])
        barchart_1 = viz.plot_barchart_tweet_distribution(
            first_comp, df_1, barchart_1)
        barchart_1.plot()
        barchart_2 = fig.add_subplot(gs[8:11, :])
        barchart_2 = viz.plot_barchart_tweet_distribution(
            second_comp, df_2, barchart_2)
        barchart_2.plot()

        # Inserting the first month
        start_year = int(first_comp.start_date[:4])
        start_month = int(first_comp.start_date[5:7])
        end_year = int(first_comp.end_date[:4])
        end_month = int(first_comp.end_date[5:7])
        if start_month == 1:
            start_month = 12
            start_year -= 1
        else:
            start_month -= 1
        """
        if start_month < 10:
            my_dates.insert(0, str(start_year) + '-0' + str(start_month))
        else:
            my_dates.insert(0, str(start_year) + '-' + str(start_month))
        """
        # Creating the linechart while making sure that we are not creating a
        # monthly plot if the analysis period is one month or less
        if start_month != end_month or (
                start_month == end_month and start_year != end_year):
            linechart_1 = fig.add_subplot(gs[4:7, :])
            linechart_1, linechart_df = viz.plot_company_linechart_with_scores(
                first_comp, my_dates, linechart_1)
            linechart_1.plot()
            linechart_2 = fig.add_subplot(gs[12:15, :])
            linechart_2, linechart_df = viz.plot_company_linechart_with_scores(
                second_comp, my_dates, linechart_2)
            linechart_2.plot()

        # Saving the plot as a pdf file
        pp = PdfPages("1.pdf")
        pp.savefig(fig)
        pp.close()
        os.system("pdf-crop-margins -u -p4 100 5 100 20 '1.pdf'")

        # Creating the piecharts
        piechart_1 = fig.add_subplot(gs[15:22, :])
        piechart_1 = viz.plot_piechart_with_scores(first_comp, piechart_1)
        piechart_1.plot()
        piechart_2 = fig.add_subplot(gs[23:, :])
        piechart_2 = viz.plot_piechart_with_scores(second_comp, piechart_2)
        piechart_2.plot()

        # Rotating the x-labels in order to make them look properly on the
        # barchart and linechart
        for ax in fig.axes:
            plt.sca(ax)
            plt.xticks(rotation=30)

        fig.subplots_adjust(wspace=1)

        # Updating file counters
        COMPANIES_COUNTER += 1
        update_file_counters()

        # Saving the plot in the as an _io.BytesIO object
        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)

        fig_pie = plt.figure(figsize=A4_PORTRAIT_MEASUREMENTS)
        gs = fig.add_gridspec(15, 4)

        # Creating the piecharts
        piechart_1 = fig_pie.add_subplot(gs[:7, :])
        piechart_1 = viz.plot_piechart_with_scores(first_comp, piechart_1)
        piechart_1.plot()
        piechart_2 = fig_pie.add_subplot(gs[8:, :])
        piechart_2 = viz.plot_piechart_with_scores(second_comp, piechart_2)
        piechart_2.plot()

        # Creating the second page of the report
        pp = PdfPages("2.pdf")
        pp.savefig((fig_pie))
        pp.close()

        merger("App/static/" +
               first_comp.company +
               "-VS-" +
               second_comp.company +
               "-report-no." +
               str(COMPANIES_COUNTER) +
               ".pdf", ['1_cropped.pdf', '2.pdf'])
        return COMPANIES_COUNTER, bytes_image, len(
            df_1) + len(df_2), first_comp.company + "-VS-" + second_comp.company
    elif "industry-name" in dict_form.keys():
        scrapper = TweetScrapper(250)
        # Getting the options selected by the user
        industry = dict_form['industry-name']
        start_date = dict_form['first_day']
        end_date = dict_form['last_day']

        # Creating the Industry object
        ind = Industry(industry, start_date, end_date)

        # Performing the tweet scraping
        df = scrapper.scrap_for_tweets_in_a_industry(ind)
        df['new_index'] = range(len(df))
        df = df.set_index('new_index')

        # If less than 30 tweets resulted than we don't have a representative
        # enough amount of tweets in order to perform any analysis
        if len(df) < 30:
            return -1, -1, -1, -1  # Not enough tweets

        # Making a copy of the dataframe in its raw form
        df_raw = deepcopy(df)

        # Saving the raw dataframe
        df_raw.to_csv("Scrapped dataframes/Industries/"+
            ind.industry +
            "_ind_" +
            str(INDUSTRIES_COUNTER) +
            "_raw.csv")

        # Preprocessing stage
        df = preprocessor.preprocess_dataframe(df)

        my_dates = erep_calc.get_monthly_dates(ind)
        # Determining the e-reputation scores
        ind.multi_year_scores = erep_calc.get_e_reputation(
            ind, model, df, my_dates)

        # Determining a dictionary with the pairs in the form of
        # {[company]-[its associated score]}
        company_to_score = erep_calc.get_e_reputation_industry(
            ind, model, df, my_dates)

        # Saving the dataframe after preprocessing and labeling
        df.to_csv("Scrapped dataframes/Industries/"+
            ind.industry +
            "_comp_" +
            str(INDUSTRIES_COUNTER) +
            "_processed.csv")

        # Preprocessing the scores in order to create some plots
        ind.multi_year_scores = erep_calc.transpose_multi_year_scores_matrix(
            ind)
        ind.yearly_averaged_scores = erep_calc.calculate_yearly_averaged_scores(
            ind)
        ind.ereputation_score = sum(ind.yearly_averaged_scores) / 3

        # Designing the grid for the plots
        fig = plt.figure(figsize=A4_PORTRAIT_MEASUREMENTS)
        gs = fig.add_gridspec(9, 5)
        plt.axis('off')
        plt.title(industry + " report\n", size=24)

        # Creating the treemap
        treemap = fig.add_subplot(gs[5:7, :])
        treemap, scores_dict = viz.plot_industry_treemap(
            ind.industry, df, company_to_score, treemap)
        treemap.plot()

        # Creating the top 10 in the industry
        top = fig.add_subplot(gs[0:2, 3:5])
        top = viz.plot_industry_top(ind.industry, scores_dict, top)
        top.plot()

        # Creating the piechart
        piechart = fig.add_subplot(gs[0:2, 0:2])
        piechart = viz.plot_piechart_with_scores(ind, piechart)
        piechart.plot()

        # Creating the wordcloud
        wordcloud = fig.add_subplot(gs[7:9, :])
        wordcloud = viz.plot_wordcloud(ind, df_raw, wordcloud)
        wordcloud.plot()

        # Inserting the first month
        start_year = int(ind.start_date[:4])
        start_month = int(ind.start_date[5:7])
        end_year = int(ind.end_date[:4])
        end_month = int(ind.end_date[5:7])
        if start_month == 1:
            start_month = 12
            start_year -= 1
        else:
            start_month -= 1
        """
                if start_month < 10:
                    my_dates.insert(0, str(start_year) + '-0' + str(start_month))
                else:
                    my_dates.insert(0, str(start_year) + '-' + str(start_month))
                """

        # Creating the linechart while making sure that we are not creating a
        # monthly plot if the analysis period is one month or less
        if start_month != end_month or (
                start_month == end_month and start_year != end_year):
            linechart = fig.add_subplot(gs[2:4, :])
            linechart, linechart_df = viz.plot_company_linechart_with_scores(
                ind, my_dates, linechart)
            linechart.plot()

        # Updating file counters
        INDUSTRIES_COUNTER += 1
        update_file_counters()

        # Rotating the x-labels in order to make them look properly on the
        # barchart and linechart
        for ax in fig.axes:
            plt.sca(ax)
            plt.xticks(rotation=30)

        # Saving the plot as a pdf file
        pp = PdfPages(
            "App/static/" +
            industry +
            "-report-no." +
            str(INDUSTRIES_COUNTER) +
            ".pdf")
        pp.savefig(fig)
        pp.close()

        # Saving the plot in the as an _io.BytesIO object
        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)

        return INDUSTRIES_COUNTER, bytes_image, len(df), ind.industry
    else:
        return -2, -2, -2, -2   # An error intervened during the process


def main():
    pass


if __name__ == "__main__":
    main()
