from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import pandas as pd

def compute_sentiment(input_file, output_file):
    # Load FinBERT
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    # Load news data
    news_data = pd.read_csv(input_file)
    sentiments = []

    for _, row in news_data.iterrows():
        articles = eval(row['Articles'])  # Convert string list back to list
        pos, neg, neu = 0, 0, 0
        if articles:
            scores = [nlp(article)[0]['label'] for article in articles]
            pos = scores.count('positive') / len(scores)
            neg = scores.count('negative') / len(scores)
            neu = scores.count('neutral') / len(scores)
        sentiments.append({'Date': row['Date'], 'Positive': pos, 'Negative': neg, 'Neutral': neu})
    
    sentiment_df = pd.DataFrame(sentiments)
    sentiment_df.to_csv(output_file, index=False)
    print(f"Sentiment data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    compute_sentiment("data/news_data.csv", "data/news_sentiment.csv")

