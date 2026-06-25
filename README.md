# 🎬 Movie Rating Predictor (Joker 2019) — Twitter Sentiment Analysis

An NLP project that predicts a movie's star rating by analyzing real-time Twitter sentiment. Built using Python, NLTK, TextBlob, and Scikit-Learn, this project demonstrates a full text classification pipeline from live data collection through model evaluation.

## Problem Statement

Movie ratings on platforms like Rotten Tomatoes or IMDb are often delayed or biased. Can we predict how audiences feel about a movie in real time — directly from what they're tweeting? This project answers that question for the **Joker (2019)** movie using #jokermovie tweets.

## Data Collection

- **Source:** Twitter API (via Tweepy) — live tweets scraped using `#jokermovie` hashtag
- **Volume:** 100+ tweets collected in real time
- **Fields captured:** username, date, tweet text, favorite count

## Approach

### Text Preprocessing
- Removed mentions (@user), hashtags (#), retweets (RT), URLs, and special characters using regex
- Converted all text to lowercase
- Tokenized tweets using NLTK's `word_tokenize`
- Visualized most frequent terms using **WordCloud**

### Sentiment Analysis
- Computed **polarity** and **subjectivity** scores using **TextBlob**
- Classified each tweet as `positive`, `negative`, or `neutral` based on polarity
- Visualized sentiment distribution with scatter plots and bar charts

### NLP Pipeline & Classification
Used **TF-IDF vectorization** (CountVectorizer → TfidfTransformer) with two classifiers:

| Model | Type |
|-------|------|
| Multinomial Naive Bayes | Probabilistic text classifier |
| Logistic Regression | Linear text classifier |

Evaluated using: confusion matrix, classification report, and accuracy score.

### Rating Derivation
Final star rating (1–5 stars) derived from the ratio of positive to total (positive + negative) tweets — translating sentiment distribution into a familiar rating scale.

## Project Structure

```
PredictingMovieRating/
├── SentimentAnalysis.py   # Full NLP pipeline
└── README.md
```

## Programmming Language and Used Libraries

- **Language:** Python
- **Libraries:** Pandas, NumPy, NLTK, TextBlob, Scikit-Learn, Tweepy, Matplotlib, Seaborn, WordCloud

## Key Takeaways

- Real-time Twitter data provides a fast, crowd-sourced signal for audience sentiment before formal ratings are published
- TF-IDF + Logistic Regression outperformed Naive Bayes on tweet classification
- TextBlob polarity provided a lightweight but effective baseline for sentiment scoring
- Sentiment ratio is a practical proxy for star ratings when ground truth labels are unavailable

## Note on API Keys

This project was built using the Twitter v1.1 API (Tweepy). Twitter's API access has since changed. To run this project, you will need to obtain your own API credentials from the [Twitter Developer Portal](https://developer.twitter.com/).

## How to Run

```bash
# Install dependencies
pip install pandas numpy nltk textblob scikit-learn tweepy wordcloud matplotlib seaborn

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Add your Twitter API credentials to SentimentAnalysis.py
# Run the script
python SentimentAnalysis.py
```
