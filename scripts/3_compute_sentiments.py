from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import pandas as pd
import ast
import json
from tqdm import tqdm
import logging
import os
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model(batch_size=32):
    """
    Load the pre-trained FinBERT model and tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
    return pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, batch_size=batch_size)

def parse_articles(articles_str):
    """
    Parse the 'Articles' string from the CSV into a list.
    Attempts JSON parsing first, then falls back to ast.literal_eval.
    """
    try:
        # Attempt to parse as JSON first
        return json.loads(articles_str) if pd.notna(articles_str) else []
    except json.JSONDecodeError:
        try:
            # Fallback to literal_eval
            return ast.literal_eval(articles_str) if pd.notna(articles_str) else []
        except (ValueError, SyntaxError) as e:
            logging.error(f"Error parsing Articles: {e}")
            return []

def analyze_sentiments(nlp, articles):
    """
    Perform sentiment analysis on a list of articles.
    Returns the proportion of positive, negative, and neutral sentiments.
    """
    if not articles:
        return 0.0, 0.0, 0.0
    try:
        results = nlp(articles)
        labels = [res['label'].lower() for res in results]
        pos = labels.count('positive') / len(labels)
        neg = labels.count('negative') / len(labels)
        neu = labels.count('neutral') / len(labels)
        return pos, neg, neu
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        return 0.0, 0.0, 0.0

def compute_sentiment(input_file, output_file, batch_size=32):
    """
    Main function to compute sentiment scores per date.
    Aggregates all articles for each date before analysis.
    """
    nlp = load_model(batch_size)
    
    # Check if input file exists
    if not os.path.isfile(input_file):
        logging.error(f"Input file not found: {input_file}")
        return
    
    try:
        # Read the CSV with appropriate parameters
        news_data = pd.read_csv(
            input_file,
            engine='python',               # Use Python engine for flexibility
            delimiter=',',                 # Specify delimiter
            quoting=csv.QUOTE_MINIMAL,     # Adjust based on your CSV
            quotechar='"',                 
            escapechar='\\',               # If applicable
            on_bad_lines='warn',           # Handle bad lines
            encoding='utf-8'               # Adjust encoding if needed
        )
        logging.info(f"Successfully read input file: {input_file}")
    except pd.errors.ParserError as e:
        logging.error(f"ParserError: {e}")
        return
    except Exception as e:
        logging.error(f"Unexpected error reading CSV: {e}")
        return
    
    logging.info(f"Columns in CSV: {news_data.columns.tolist()}")
    
    required_columns = {'Date', 'Articles'}
    if not required_columns.issubset(news_data.columns):
        missing = required_columns - set(news_data.columns)
        logging.error(f"Input file is missing required columns: {missing}")
        return
    
    # Optional: Log a sample of the data
    if not news_data.empty:
        sample_articles = news_data['Articles'].iloc[0]
        logging.info(f"Sample 'Articles' data: {sample_articles}")
    
    # Parse the 'Articles' column into lists
    news_data['Parsed_Articles'] = news_data['Articles'].apply(parse_articles)
    
    # Group by 'Date' and aggregate all articles for each date
    grouped = news_data.groupby('Date')['Parsed_Articles'].apply(lambda lists: [article for sublist in lists for article in sublist]).reset_index()
    
    logging.info("Successfully grouped articles by date.")
    
    sentiments = []
    for _, row in tqdm(grouped.iterrows(), total=grouped.shape[0], desc="Processing dates"):
        date = row['Date']
        articles = row['Parsed_Articles']
        pos, neg, neu = analyze_sentiments(nlp, articles)
        sentiments.append({
            'Date': date,
            'Positive': pos,
            'Negative': neg,
            'Neutral': neu
        })
    
    sentiment_df = pd.DataFrame(sentiments)
    sentiment_df.to_csv(output_file, index=False)
    logging.info(f"Sentiment data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    compute_sentiment("data/news_data.csv", "data/news_sentiment.csv")

