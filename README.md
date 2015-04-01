# Template description.

Goal of this template is implement RNN algorithm, as RNTN algorithm in deeplearning4j library does not work yet.

Recursive Neural Network algorithm might be used in supervised learning algorithm used to predict sentiment of sentences.

# Installation.

Follow [installation guide for PredictionIO](http://docs.prediction.io/install/).

After installation start all PredictionIO vendors and check pio status:
```bash
pio-start-all
pio status
```

This template depends on deeplearning4j 0.0.3.3.3.alpha1-SNAPSHOT with implementation of RNN algorithm. In order to install it run:
```bash
git clone git@github.com:ts335793/deeplearning4j.git
cd deeplearning4j
git checkout rnn
chmod a+x setup.sh
./setup.sh
```

Copy this template to your local directory with:
```bash
pio template get ts335793/template-scala-parallel-word2vec-rnn <TemplateName>
```

# Build, train, deploy.

You might build template, train it and deploy by typing:
```bash
pio build
pio train -- --executor-memory=4GB --driver-memory=4GB
pio deploy -- --executor-memory=4GB --driver-memory=4GB
```
Those pio train options are used to avoid problems with java garbage collector. In case they appear increase executor memory and driver memory.

# Importing training data.

You can import example training data from [kaggle](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data). It is collection of the Rotten Tomatoes movie reviews with sentiment labels.

In order to use this data, create new app:
```bash
pio app new <ApplicationName> # prints out ApplicationAccessKey and ApplicationId
```
set appId in engine.json to ApplicationId and import data with:
```bash
python data/import_eventserver.py --access_key <ApplicationAccessKey> --file train.tsv
```

You can always remind your application id and key with:
```bash
pio app list
```

# Sending requests to server.

In order to send a query run in template directory:
```bash
python data/send_query_interactive.py
```
and type phrase you want sentiment to be predicted. The result will be a list of predicted sentiments for all sentences in phrase.

# Algorithm overview.

At first Word2Vec is trained (it creates mapping from words to vectors).
```scala
val (vocabCache, weightLookupTable) = {
  val result = new SparkWord2Vec().train(data.phrases)
  (result.getFirst, result.getSecond)
}
```

Training phrases are converted to trees.
```scala
val rawTrees = data.labeledPhrases.mapPartitions(labeledPhrases => {
  val treeVectorizer = new TreeVectorizer()
  labeledPhrases.map(labeledPhrase => (treeVectorizer.getTreesWithLabels(labeledPhrase.phrase, data.labels), labeledPhrase.sentiment))
})
val convertedTrees = rawTrees.flatMap(rawTreesWithSentiment => {
  val (rawTrees, sentiment) = rawTreesWithSentiment
  rawTrees.map(rawTree => new Pair[Tree, Integer](Tree.fromTreeVectorizer(rawTree, vocabCache, weightLookupTable), sentiment))
})
```

On every spark data partition model is trained with RNN algorithm.
```scala
val collectedConvertedTrees = convertedTrees.glom()
val judgesAndCombinators = collectedConvertedTrees.map(convertedTrees => {
  val rnnSettings = new RNN.Settings(ap.inSize, data.labels.length)
  val rnn = new RNN(rnnSettings)
  val convertedTreesList = convertedTrees.toList
  rnn.stochasticGradientDescent(convertedTreesList)
  (rnn.judge, rnn.combinator, 1.0)
})
```

Average of all trained models is created, as in paper about [parallel stochastic gradient descent](http://www.research.rutgers.edu/~lihong/pub/Zinkevich11Parallelized.pdf).
```scala
val (judge, combinator, _) = judgesAndCombinators.reduce({
  case ((jl, cl, ql), (jr, cr, qr)) =>
    val sum = ql + qr
    val jlq = jl.mul(ql)
    val jrq = jr.mul(qr)
    val j = (jlq.add(jrq)).mul(1.0 / sum)
    val clq = cl.mul(ql)
    val crq = cr.mul(qr)
    val c = (clq.add(crq)).mul(1.0 / sum)
    (j, c, sum)
})
```

Model is saved.
```scala
new Model(
  vocabCache = vocabCache,
  weightLookupTable = weightLookupTable,
  judge = judge,
  combinator = combinator,
  labels = data.labels
)
```

# Serving overview.

List of trees for sentences in query is created.
```scala
val rawTrees = new TreeVectorizer().getTreesWithLabels(query.content, model.labels)
val convertedTrees = rawTrees.map(Tree.fromTreeVectorizer(_, model.vocabCache, model.weightLookupTable))
```

Sentiment for each sentence is being predicted.
```scala
val rnnSettings = new RNN.Settings(ap.inSize, model.labels.length)
val rnn = new RNN(rnnSettings, model.combinator, model.judge)
val sentiments = convertedTrees.map(rnn.predictClass(_))
```

Result is returned.
```scala
PredictedResult(sentiments = sentiments.toList)
```