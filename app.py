import re
from flask import Flask , render_template , request
import pickle

tokenizer=pickle.load(open("models/tfidf.pkl","rb"))
model=pickle.load(open("models/clf.pkl","rb"))
def clean_text(text):
    text = str(text)  # Ensure text is a string
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)    # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation, keep only letters and spaces
    text = text.lower()  # Convert to lowercase
    return text


app=Flask(__name__)

@app.route("/",methods=["GET","POST"])

def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    news_text=request.form.get("news-content")
    cleaned_news = clean_text(news_text) 
    print(clean_text)
    
    tokenized_news=tokenizer.transform([cleaned_news])
    print(tokenized_news)
    predictions=model.predict(tokenized_news)
    print(predictions)
    predictions ="True" if predictions=="True" else "Fake"
    print(predictions)
    return render_template("index.html",predictions=predictions,news_text=news_text)
                                
                                
                            

if __name__ == "__main__":
    app.run(debug=True)