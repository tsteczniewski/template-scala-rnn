# Template description.

This template provides sentiment analysis algorithm [RNN](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf).

# Installation.

Follow [installation guide for PredictionIO](http://docs.prediction.io/install/).

After installation start all PredictionIO vendors and check pio status:

```bash
pio-start-all
pio status
```

Copy this template to your local directory with:

```bash
pio template get ts335793/template-scala-spark-dl4j-word2vec-rnn <TemplateName>
```

Download [en-parser-chunking.bin](http://opennlp.sourceforge.net/models-1.5/en-parser-chunking.bin) and place it in `<TemplateDirectory>/src/main/resources/`.

# Importing training data.

You can import example training data from [kaggle](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data). It is collection of the Rotten Tomatoes movie reviews with sentiment labels.

In order to use this data, create new app:

```bash
pio app new <ApplicationName> # prints out ApplicationAccessKey
```

set appName in engine.json to ApplicationName and import data with:

```bash
python data/import_eventserver.py --access_key <ApplicationAccessKey> --file train.tsv
```

You can always remind your application id and key with:

```bash
pio app list
```

# Build, train, deploy.

You might build template, train it and deploy by typing:

```bash
pio build
pio train
pio deploy
```

# Sending requests to server.

In order to send a query run in template directory:

```bash
python data/send_query_interactive.py
```
and type phrase you want sentiment to be predicted. The result will be predicted sentiment for the phrase.