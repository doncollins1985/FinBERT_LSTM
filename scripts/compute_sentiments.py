from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import pandas as pd
import ast
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(batch_size=32):
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = BertForSequenceClassification.from_pretrained(
        'yiyanghkust/finbert-tone')
    return pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, batch_size=batch_size)


def parse_articles(articles_str):
    try:
        return ast.literal_eval(articles_str) if pd.notna(articles_str) else []
    except (ValueError, SyntaxError) as e:
        logging.error(f"Error parsing Articles: {e}")
        return []


def analyze_sentiments(nlp, articles):
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
    nlp = load_model(batch_size)
    news_data = pd.read_csv(input_file)

    # Verify required columns
    required_columns = {'Date', 'Articles'}
    if not required_columns.issubset(news_data.columns):
        missing = required_columns - set(news_data.columns)
        raise ValueError(f"Input file is missing required columns: {missing}")

    sentiments = []
    for _, row in tqdm(news_data.iterrows(), total=news_data.shape[0], desc="Processing rows"):
        articles = parse_articles(row['Articles'])
        pos, neg, neu = analyze_sentiments(nlp, articles)
        sentiments.append({
            'Date': row['Date'],
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
