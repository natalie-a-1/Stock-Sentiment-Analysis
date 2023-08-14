from nltk.corpus import stopwords
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import requests
import config

# Initialize stopwords
stop_words = set(stopwords.words('english'))


# Process data
def preprocess_text(text):
    blob = TextBlob(text)
    tokens = [word for word in blob.words if word not in stop_words]
    text = ' '.join(tokens)
    return text


def get_sentiment(text, analyzer):
    blob = TextBlob(text, analyzer=analyzer)
    return blob.sentiment.classification


def get_stock_sentiment(stock, start, end):
    api_key = config.api_key
    url = f'https://newsapi.org/v2/everything?q={stock}&from=start&to=end&apiKey={api_key}'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        articles = data['articles']
        sentiments = []

        sentiment_analyzer = NaiveBayesAnalyzer()
        for article in articles:
            text = article['title'] + ' ' + article['description']
            text = preprocess_text(text)
            sentiment = get_sentiment(text, sentiment_analyzer)
            if sentiment == 'pos':
                sentiments.append(1)
            elif sentiment == 'neg':
                sentiments.append(-1)

        if len(sentiments) > 0:
            avg_sentiment = sum(sentiments) / len(sentiments)
            print(f'Average sentiment for {stock} is {avg_sentiment}')
        else:
            print(f'No articles found for {stock} in the specified date range.')

    except requests.exceptions.RequestException as e:
        print(f'An error occurred: {e}')


stock = input("Enter a stock to analyze: ")
start_date = input("Enter a start date (format = 1999-01-01): ")
end_date = input("Enter an end date (format = 1999-01-01): ")

get_stock_sentiment(stock, start_date, end_date)