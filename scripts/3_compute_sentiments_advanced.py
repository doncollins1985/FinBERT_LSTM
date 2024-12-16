#!/usr/bin/env python3
# compute_sentiments.py

import os
import csv
import json
import ast
import logging
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import pandas as pd
from tqdm import tqdm
from multiprocessing import cpu_count, Pool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("compute_sentiments.log"),
        logging.StreamHandler()
    ]
)

def load_model(batch_size=32):
    """
    Load the pre-trained FinBERT model and tokenizer.
    
    Args:
        batch_size (int): Number of samples per batch for sentiment analysis.
    
    Returns:
        pipeline: Hugging Face transformers pipeline for sentiment analysis.
    """
    logging.info("Loading FinBERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, batch_size=batch_size)
    logging.info("Model and tokenizer loaded successfully.")
    return nlp

def clean_csv(input_file, cleaned_file):
    """
    Clean the CSV by ensuring proper quoting and removing malformed lines.
    
    Args:
        input_file (str): Path to the original CSV file.
        cleaned_file (str): Path to save the cleaned CSV file.
    """
    logging.info(f"Starting CSV cleaning: {input_file} -> {cleaned_file}")
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(cleaned_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
        header = next(reader, None)
        if header:
            writer.writerow(header)
        for i, row in enumerate(reader, start=2):  # Start at 2 considering header
            if len(row) >= 2:
                writer.writerow(row)
            else:
                logging.warning(f"Skipping malformed line {i}: {row}")
    logging.info("CSV cleaning completed.")

def parse_articles(articles_str):
    """
    Parse the 'Articles' string from the CSV into a list.
    Attempts JSON parsing first, then falls back to ast.literal_eval.
    
    Args:
        articles_str (str): String representation of articles.
    
    Returns:
        list: List of articles.
    """
    if pd.isna(articles_str):
        return []
    try:
        # Attempt to parse as JSON
        return json.loads(articles_str)
    except json.JSONDecodeError:
        try:
            # Fallback to literal_eval
            return ast.literal_eval(articles_str)
        except (ValueError, SyntaxError) as e:
            logging.error(f"Error parsing Articles: {e} | Data: {articles_str}")
            return []

def analyze_sentiments(nlp, articles):
    """
    Perform sentiment analysis on a list of articles.
    Returns the proportion of positive, negative, and neutral sentiments,
    along with an aggregate score.
    
    Args:
        nlp (pipeline): Sentiment analysis pipeline.
        articles (list): List of article texts.
    
    Returns:
        tuple: (positive_ratio, negative_ratio, neutral_ratio, aggregate_score)
    """
    if not articles:
        return 0.0, 0.0, 0.0, 0.0
    try:
        results = nlp(articles)
        labels = [res['label'].lower() for res in results]
        pos = labels.count('positive') / len(labels)
        neg = labels.count('negative') / len(labels)
        neu = labels.count('neutral') / len(labels)
        aggregate_score = pos - neg  # Example: Positive minus Negative
        return pos, neg, neu, aggregate_score
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        return 0.0, 0.0, 0.0, 0.0

def remove_duplicates(news_data):
    """
    Remove duplicate articles based on 'Date' and 'Articles'.
    
    Args:
        news_data (DataFrame): Pandas DataFrame containing news data.
    
    Returns:
        DataFrame: Deduplicated DataFrame.
    """
    initial_count = news_data.shape[0]
    news_data = news_data.drop_duplicates(subset=['Date', 'Articles'])
    final_count = news_data.shape[0]
    logging.info(f"Removed {initial_count - final_count} duplicate rows.")
    return news_data

def handle_missing_values(news_data):
    """
    Handle missing values in the DataFrame by removing rows with missing 'Date' or 'Articles'.
    
    Args:
        news_data (DataFrame): Pandas DataFrame containing news data.
    
    Returns:
        DataFrame: Cleaned DataFrame with no missing 'Date' or 'Articles'.
    """
    initial_count = news_data.shape[0]
    news_data = news_data.dropna(subset=['Date', 'Articles'])
    final_count = news_data.shape[0]
    logging.info(f"Removed {initial_count - final_count} rows with missing 'Date' or 'Articles'.")
    return news_data

def compute_sentiment(input_file, output_file, cleaned_file=None, batch_size=32, parallel=False):
    """
    Main function to compute sentiment scores per date.
    Aggregates all articles for each date before analysis.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the output sentiment CSV.
        cleaned_file (str, optional): Path to save the cleaned CSV. Defaults to None.
        batch_size (int, optional): Batch size for sentiment analysis. Defaults to 32.
        parallel (bool, optional): Whether to use parallel processing. Defaults to False.
    """
    nlp = load_model(batch_size)
    
    # Step 1: Clean the CSV if a cleaned_file path is provided
    if cleaned_file:
        clean_csv(input_file, cleaned_file)
        data_file = cleaned_file
    else:
        data_file = input_file
    
    # Step 2: Read the CSV with appropriate parameters
    try:
        news_data = pd.read_csv(
            data_file,
            engine='python',
            delimiter=',',
            quoting=csv.QUOTE_MINIMAL,
            quotechar='"',
            escapechar='\\',
            on_bad_lines='warn',
            encoding='utf-8'
        )
        logging.info(f"Successfully read input file: {data_file}")
    except pd.errors.ParserError as e:
        logging.error(f"ParserError: {e}")
        return
    except Exception as e:
        logging.error(f"Unexpected error reading CSV: {e}")
        return
    
    logging.info(f"Columns in CSV: {news_data.columns.tolist()}")
    
    # Step 3: Validate required columns
    required_columns = {'Date', 'Articles'}
    if not required_columns.issubset(news_data.columns):
        missing = required_columns - set(news_data.columns)
        logging.error(f"Input file is missing required columns: {missing}")
        return
    
    # Step 4: Data Cleaning
    news_data = remove_duplicates(news_data)
    news_data = handle_missing_values(news_data)
    
    # Step 5: Parse the 'Articles' column into lists
    news_data['Parsed_Articles'] = news_data['Articles'].apply(parse_articles)
    
    # Step 6: Group by 'Date' and aggregate all articles for each date
    grouped = news_data.groupby('Date')['Parsed_Articles'].apply(
        lambda lists: [article for sublist in lists for article in sublist]
    ).reset_index()
    
    logging.info(f"Aggregated articles by date. Total unique dates: {grouped.shape[0]}")
    
    # Optional: Log a sample of the aggregated data
    if not grouped.empty:
        sample_date = grouped['Date'].iloc[0]
        sample_articles = grouped['Parsed_Articles'].iloc[0]
        logging.info(f"Sample 'Articles' for {sample_date}: {sample_articles[:2]}{'...' if len(sample_articles) > 2 else ''}")
    
    # Step 7: Perform Sentiment Analysis
    sentiments = []
    
    if parallel:
        logging.info("Starting sentiment analysis with parallel processing...")
        # Define a helper function for parallel processing
        def process_row(row):
            date = row['Date']
            articles = row['Parsed_Articles']
            pos, neg, neu, agg = analyze_sentiments(nlp, articles)
            return {'Date': date, 'Positive': pos, 'Negative': neg, 'Neutral': neu, 'Aggregate_Score': agg}
        
        with Pool(processes=cpu_count()) as pool:
            sentiments = list(tqdm(pool.imap(process_row, [row for _, row in grouped.iterrows()]), 
                                   total=grouped.shape[0], desc="Processing dates"))
    else:
        logging.info("Starting sentiment analysis with sequential processing...")
        for _, row in tqdm(grouped.iterrows(), total=grouped.shape[0], desc="Processing dates"):
            date = row['Date']
            articles = row['Parsed_Articles']
            pos, neg, neu, agg = analyze_sentiments(nlp, articles)
            sentiments.append({
                'Date': date,
                'Positive': pos,
                'Negative': neg,
                'Neutral': neu,
                'Aggregate_Score': agg
            })
    
    # Step 8: Create DataFrame and Save Results
    sentiment_df = pd.DataFrame(sentiments)
    
    # Optional: Reorder columns
    sentiment_df = sentiment_df[['Date', 'Positive', 'Negative', 'Neutral', 'Aggregate_Score']]
    
    # Save to CSV
    try:
        sentiment_df.to_csv(output_file, index=False)
        logging.info(f"Sentiment data saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving sentiment data to CSV: {e}")

def main():
    """
    Entry point of the script. Parses command-line arguments and initiates sentiment computation.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Compute sentiment scores from news articles CSV.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input CSV file (e.g., data/news_data.csv).")
    parser.add_argument('--output', type=str, required=True, help="Path to save the output sentiment CSV (e.g., data/news_sentiment.csv).")
    parser.add_argument('--cleaned', type=str, default=None, help="Path to save the cleaned CSV file. If not provided, cleaning is skipped.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for sentiment analysis.")
    parser.add_argument('--parallel', action='store_true', help="Enable parallel processing for sentiment analysis.")
    
    args = parser.parse_args()
    
    compute_sentiment(
        input_file=args.input,
        output_file=args.output,
        cleaned_file=args.cleaned,
        batch_size=args.batch_size,
        parallel=args.parallel
    )

if __name__ == "__main__":
    main()

