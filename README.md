# Sentiment Analysis using LSTM model
<div style="padding-left:10px; padding-right:15px;">A LSTM model for sentiment analysis and API provided for those who wants to use LSTM sentiment analysis for their applications.</div>
<hr>
 
### Overview:
<div style="padding-left:10px; padding-right:15px;"> This is an LSTM model that allows users to do basic sentiment analysis. 

- Provides the users the ability to prepare data for the ML model
- Provides users the ability to create complex mini-batches of large data.
- Allows users to train model, plot graphs and saves checkpoint for early stopping.</div>


### Sentiment Analysis API:
<div style="padding-left:10px; padding-right:15px;"> The LSTM model is trained on dataset provided by <a href="http://help.sentiment140.com/for-students">sentiment140</a>. You can access the REST API created using the LSTM model in this repo. The location for REST API server is:

`www.sentiment-analysis-api.site`

The REST API sends 3 responses to a request made:

- 0 : If the sentiment of a sentence is _negative_
- 1 : If sentiment of a sentence is _positive_

You can access the API as follows:

__Single Sentence:__

<div style="padding-left:10px; padding-right:15px;"> 

If you want to parse a single sentence, to determine it's sentiment, we can send a basic JSON object using curl as follows:

```sh
>> curl -d '{"data":"this is awesome"}' -H "Content-Type: application/json" -X POST https://www.sentiment-analysis-api.site/one
1

>> curl -d '{"data":"I need a hug"}' -H "Content-Type: application/json" -X POST https://www.sentiment-analysis-api.site/one
0

>> curl -d '{"data":"i wish i was happy"}' -H "Content-Type: application/json" -X POST https://www.sentiment-analysis-api.site/one
0
```

</div>

__Bulk Sentences__

<div style="padding-left:10px; padding-right:15px;"> 

If you want to parse bulk of sentences in order to determine it's sentiment, you can send a basic JSON object using curl as follows:


```sh
>> curl -d '{"data":["This is awesome", "I need a hug", "i wish i was happy","My name is Beant"]}' -H "Content-Type: application/json" -X POST https://www.sentiment-analysis-api.site/bulk
{
  "I need a hug": 0, 
  "My name is Beant": 1, 
  "This is awesome": 1, 
  "avg": 0.5, 
  "i wish i was happy": 0
}
```
where `avg` is average score of the sentences.

</div>
</div>


### Model Training:









