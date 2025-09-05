# Circular Statistics and Emotion Dynamics Twitter Demos

This project is a **Flask web app** that demonstrates two interactive tools:

---

## Emotion Dynamics of Sentiment in Tweets

In the first demo, you can provide the text content of any Tweet, and a pre-trained Hugging Face model (**RoBERTa**) will provide the probability this is expressing positive or negative emotion.  

These statistics are plotted onto a line graph, which is continuously updated as you input more Tweets.  

The demo also generates three key **emotion dynamics scores** using the Tweets you have inputted:

1. **Mean** positive and negative emotion score.  
2. **Variability** (standard deviation) in the expression of positive and negative emotion.  
3. **Instability** (mean-squared successive difference, MSSD) in the expression of positive and negative emotion.  

Our measurement of instability is **time-adjusted**, meaning that it takes into account the difference in time between each successive Tweet.  

By default, the Tweets you input will be treated as **60 minutes apart**, but a **slider** lets you change this interval for all Tweets you input.  

---

## Circular Statistics for Timing of Tweet Posting

In the second demo, we provide an **interactive clock**, where you can click to add times of the day when a Tweet was posted.  

As you add times, the **circular mean** and **circular median** of Tweet posting hours are continuously updated.  
We also provide a **circular density plot** of the distribution of your inputted hours.  

You also have the option to generate a **random selection of times** of your chosen number.  

---

## Running These Demos

These demos can be run from a command line terminal:

### 1. Clone the repository
```bash
git clone https://github.com/JoinsonCoding/Circular-Emo_Dynamics-Tweets-Demos.git
cd Circular-Emo_Dynamics-Tweets-Demos
```

**### 2. Install dependencies**
```bash
pip install -r requirements.txt
```

**### 3. Run the app**
```bash
python app.py
```

Once running, visit the local webpage shown in your terminal (usually:
http://127.0.0.1:5000)

