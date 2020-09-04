# Brand-analysis

This is a repository for a project entitled _"Monitorization and analysis of brands using AI algorithms for Natural Language processing, based on data extracted from Twitter"_. The purpose of this project is the development of an API that is going to tell us the general view about a certain brand based on the tweets where the brand is mentioned. The way the PR image of the brand looks like will be decided by a machine learning model that will determine this.

The data that is going to be used in this project for the AI part consists mainly of tweets together with some information regarding the tweet and a label that is going to tell us whether the tweet is negative, neutral or positive.

The project is structured in the following steps:

**1.Data Gathering** - finding enough tweets for training a good model;

**2.Data Cleansing** - getting rid of tweets that might be in other languages or might not give us any relevant information about a brand;

**3.Annotation** - manually deciding what is the polarity of the tweet;

**4.Data Preprocessing** - preprocessing the tweets and chosing a good way of representation so that the data can be ready for training;

**5.Training** - finding a suitable machine learning model that is trying to predict the general opinion regarding a brand based on the polarity of the tweets where the brand is mentioned;

**6.Fine-tuning** - finding the right parameters for the best model from the prior step;

**7.API development** - developping an API that will receive the name of a brand and will tell us what is the general public opinion of the brands based on the data directly collected from Twitter.
