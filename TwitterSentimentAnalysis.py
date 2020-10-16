import tkinter as tk

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import matplotlib.patches

import numpy as np

import tweepy
from tweepy import OAuthHandler

import random
import csv
import re

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.ensemble import RandomForestClassifier

import seaborn

from collections import Counter

search_term = "#AppleEvent"
num_tweets = 2000

fetched_tweets = []


#Function for removing web links beggining with http
def remove_words_starting_with(link_text, starting_text):
    while starting_text in link_text: 
        word_start = link_text.find(starting_text)
        if link_text.find(' ', word_start + 1) >=0:  
            word_end = link_text.find(' ', word_start + 1)
        else:
            word_end = len(link_text) + 1
        link_text = link_text[0:word_start] + link_text[word_end:]
    return link_text

#Function for cleaning text
def clean_text(text_in):

  #remove web links
  processed_tweet = remove_words_starting_with(text_in, "http")

  #remove user links
  processed_tweet = remove_words_starting_with(processed_tweet, "@")
      
  # Remove all the special characters
  processed_tweet = re.sub(r'\W', ' ', processed_tweet)

  # remove all single characters
  processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)

  # Remove single characters from the start
  processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 

  # Substituting multiple spaces with single space
  processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)

  # Removing prefixed 'b'
  processed_tweet = re.sub(r'^b\s+', '', processed_tweet)

  # Converting to Lowercase
  processed_tweet = processed_tweet.lower()

  return processed_tweet
#

class TweetFetcher:
    def __init__(self):
        consumer_api_key = 'nCSS9AOFEtjC6z3hSrfdgh760'
        consumer_api_secret = 'zj6ELUiIIycKmSEx6jzUgNIYBGLPk7uLnF241rGzmHvcgHQJ9H' 
        access_token = '1315955315314167808-3rEY0WXk5XasWU1AgCEX6Ahbkjonzs'
        access_token_secret ='rSc7JXXHqwbbqAHOp3qHUwSYpLC8ViXybnnCXs3M4DNWI'

        authorizer = OAuthHandler(consumer_api_key, consumer_api_secret)
        authorizer.set_access_token(access_token, access_token_secret)

        self.api = tweepy.API(authorizer ,timeout=15)

    def Fetch(self, term) :
        tweets = []
        for tweet_object in tweepy.Cursor(self.api.search,q=term+" -filter:retweets -@",lang='en',result_type='mixed',tweet_mode='extended').items(num_tweets):
            tweets.append(tweet_object)
        
        return tweets
            
class Model :
    def __init__(self):

        X = []
        Y = []
        
        #load csv file - format returns tweet as [0] tweet text, [1] sentiment
        with open('model_tweet_results.csv', newline='', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                if(row[0]=="tweet_text") : continue
                X.append(row[0])
                Y.append(row[1])

        #

        # clean input tweets
        
        processed_tweets = []

        for tweet in X:
            processed_tweets.append(clean_text(tweet))

        #

        self.tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
        X = self.tfidfconverter.fit_transform(processed_tweets).toarray()

        self.text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)  
        self.text_classifier.fit(X, Y)
        
        #
        
        return

    def Resolve(self, tweet):
        return self.text_classifier.predict(self.tfidfconverter.transform([tweet]).toarray())

class MainWindow(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.config(bg="grey")
        self.pack(expand=True) 
        self.create_widgets()

    def create_widgets(self):
        self.search_title = tk.Label(self, text="Sentiment Analysis", bg="grey", font=("Ariel",32),height=1)
        self.search_title.pack()
        
        self.search_request = tk.Text(self,font=("Ariel",32),height=1)
        self.search_request.pack(padx = 20, pady=10)
        self.search_request.bind("<Return>", self.confirm_search)
        
        self.search_button = tk.Button(self, text="Search", bg="green", font=("Ariel", 24), command=self.confirm_search)
        self.search_button.pack(fill=tk.X, side=tk.BOTTOM, expand=True, pady=10, padx=20)

    def show_results(self): 
        self.results_window = tk.Toplevel(self.master)
        self.results_window.geometry("900x500")
        ResultsWindow(master=self.results_window)

        self.graph_window = tk.Toplevel(self.master)
        self.graph_window.geometry("1200x600")
        GraphsWindow(master=self.graph_window)

    def confirm_search(self, event=None):
        global search_term
        search_term = self.search_request.get("1.0",tk.END)

        self.search_button["state"]="disabled"
        self.search_button["text"]= "Loading..."
        
        print(search_term)

        self.master.update()
        
        self.show_results()
        return 'break'




class ResultsWindow(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.title = tk.Label(self, text="Sentiment Analysis Results for " + search_term, width=1280, bg="grey", font=("Ariel", 24))
        self.title.pack(expand=True, fill="x")

        # build scrolling frame view

        self.parent_frame = tk.Frame(self, bg="grey")

        self.scrolling_canvas = tk.Canvas(self.parent_frame, height=600)

        self.scrollbar = tk.Scrollbar(self.parent_frame, orient="vertical", command=self.scrolling_canvas.yview)
        
        self.scrolling_frame = tk.Frame(self.scrolling_canvas)
        self.scrolling_frame.bind( "<Configure>", lambda e: self.scrolling_canvas.configure(scrollregion=self.scrolling_canvas.bbox("all")) )

        #

        self.scrolling_canvas.create_window((0,0), window=self.scrolling_frame, anchor="nw")
        self.scrolling_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        #


        self.parent_frame.pack(side="left",expand=True, fill="both")
        self.scrolling_canvas.pack(side="left",fill="both",expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.build_results()

        #
    def build_results(self) :
        global fetched_tweets
        tweets = twitter.Fetch(search_term)
        fetched_tweets = tweets
        
        i = 1

        result_table = [0,0,0]
        
        for tweet in tweets:
            result = int(model.Resolve(clean_text(tweet.full_text)))
            
            text="Neutral"
            bg = "grey"
            if result == 0:
                bg = "green"
                text="Positive"
            elif result == 2:
                bg = "red"
                text="Negative"
            
            t = tk.Text(self.scrolling_frame, height=3, wrap="word")
            t.insert(tk.END,tweet.full_text)
            t.grid(row=i, column=1)

            l = tk.Label(self.scrolling_frame, text=text, width=20, bg=bg)
            l.grid(row=i, column=2)
            
            
            i+=1
        
        
class GraphsWindow(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.title = tk.Label(self, text="Sentiment Analysis Results for " + search_term, width=1280, bg="grey", font=("Ariel", 24))
        self.title.pack(expand=True, fill="x")

        # build scrolling frame view

        self.parent_frame = tk.Frame(self, bg="grey")

        self.scrolling_canvas = tk.Canvas(self.parent_frame, height=600)

        self.scrollbar = tk.Scrollbar(self.parent_frame, orient="vertical", command=self.scrolling_canvas.yview)
        
        self.scrolling_frame = tk.Frame(self.scrolling_canvas)
        self.scrolling_frame.bind( "<Configure>", lambda e: self.scrolling_canvas.configure(scrollregion=self.scrolling_canvas.bbox("all")) )

        #

        self.scrolling_canvas.create_window((0,0), window=self.scrolling_frame, anchor="nw")
        self.scrolling_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        #


        self.parent_frame.pack(side="left",expand=True, fill="both")
        self.scrolling_canvas.pack(side="left",fill="both",expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.build_results()

        #
    def build_results(self):
        result_table = [0,0,0]
        
        for tweet in fetched_tweets:
            result = int(model.Resolve(clean_text(tweet.full_text)))

            result_table[result] += 1
        
        fig = Figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.pie(result_table, colors=["green", "grey", "red"])
        ax.legend(["Positive","Neutral","Negative"], title="Sentiment")
        ax.set_title("Overall Sentiment from "+str(num_tweets)+" Tweets")

        circle = matplotlib.patches.Circle( (0,0),0.7,color="white")
        ax.add_artist(circle)
        
        canvas = FigureCanvasTkAgg(fig, master=self.scrolling_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.build_country_graph()

    def build_country_graph(self):
        country_data = {}

        # build data for graph
        
        for tweet in fetched_tweets:
            if tweet.place is None:
                continue
            
            tweet_country = tweet.place.country
            tweet_country_code = tweet.place.country_code
            # convert from 0=positive 2=negative to -1=negative and 1=positive
            tweet_sentiment = -(int(model.Resolve(clean_text(tweet.full_text)))-1)

            if tweet_country_code not in country_data :
                country_data[tweet_country_code] = {"tweets_count":0, "net_sentiment":0, "country":tweet_country}

            country_data[tweet_country_code]["tweets_count"]+=1
            country_data[tweet_country_code]["net_sentiment"]+=tweet_sentiment

        sorted_country_data = sorted(country_data.values(), key= lambda i: i['net_sentiment'])
        
        #

        labels = []
        sizes= []
        explodes = []
        colors = []

        max_sentiment = sorted_country_data[0]["net_sentiment"]
        min_sentiment = sorted_country_data[len(sorted_country_data)-1]["net_sentiment"]

        if abs(min_sentiment) > (max_sentiment):
            index_shift = abs(min_sentiment)
            number_of_colours = abs(min_sentiment)*2
            if number_of_colours % 2 == 0: 
                number_of_colours = number_of_colours + 1
        else: 
            index_shift = abs(max_sentiment)
            number_of_colours = abs(max_sentiment)*2
            if number_of_colours % 2 == 0:
                number_of_colours = number_of_colours + 1

        color_palette = seaborn.color_palette("RdYlGn", number_of_colours)
        
        #create lists for sorted_country_data chart parameters
        for country in sorted_country_data:
            print(country)
            labels.append(country["country"])
            sizes.append(country["tweets_count"])
            explodes.append(0.05)
            color_value = color_palette[country["net_sentiment"] + index_shift]
            colors.append(color_value)
        
        fig = Figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        ax.pie(sizes, labels=labels, explode=explodes, colors=colors, startangle=-90)
        ax.set_title("Country Net-Sentiment from "+str(num_tweets)+" Tweets")

        circle = matplotlib.patches.Circle( (0,0),0.3,color="white")
        ax.add_artist(circle)
        
        canvas = FigureCanvasTkAgg(fig, master=self.scrolling_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        #
        self.build_word_graph()

    def build_word_graph(self):
        word_counter = Counter()
        for tweet in fetched_tweets:
            result = int(model.Resolve(clean_text(tweet.full_text)))

            words = [x for x in clean_text(tweet.full_text).split(" ") if len(x) >= 4 and x.lower() not in ["with","this","they","next","from","will","have"]]

            word_counter.update(words)


        for label, value in word_counter.most_common(10):
            tk.Label(self.scrolling_frame, text=label+" : " + str(value)).pack()

            

twitter = TweetFetcher()
model = Model()

root = tk.Tk()
root.geometry("600x200")
root.config(bg="grey")
app = MainWindow(master=root)

app.mainloop()
