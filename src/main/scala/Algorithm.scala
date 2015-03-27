package org.template.vanilla

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import org.apache.spark.SparkContext

import org.deeplearning4j.berkeley.Pair
import org.deeplearning4j.models.embeddings.WeightLookupTable
import org.deeplearning4j.models.rnn.RNN
import org.deeplearning4j.models.rnn.Tree
import org.deeplearning4j.models.word2vec.wordstore.VocabCache
import org.deeplearning4j.spark.models.word2vec.{Word2Vec => SparkWord2Vec}
import org.deeplearning4j.text.corpora.treeparser.TreeVectorizer

import scala.collection.JavaConversions._

import grizzled.slf4j.Logger

case class AlgorithmParams(
  val inSize: Integer
) extends Params

class Algorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): Model = {
    val (vocabCache, weightLookupTable) = {
      val result = new SparkWord2Vec().train(data.phrases)
      (result.getFirst, result.getSecond)
    }
    val rawTreess = data.labeledPhrases.mapPartitions(labeledPhrases => {
      val treeVectorizer = new TreeVectorizer()
      labeledPhrases.map(labeledPhrase => (treeVectorizer.getTreesWithLabels(labeledPhrase.phrase, data.labels), labeledPhrase.sentiment))
    })
    val convertedTreess = rawTreess.map(rawTrees => {
      val (rawTrees_, sentiment) = rawTrees
      rawTrees_.map(rawTree => new Pair[Tree, Integer](Tree.fromTreeVectorizer(rawTree, vocabCache, weightLookupTable), sentiment))
    })
    val convertedTrees = convertedTreess.reduce(_ ++ _)
    val rnnSettings = new RNN.Settings(ap.inSize, data.labels.length)
    val rnn = new RNN(rnnSettings)
    rnn.fit(convertedTrees)
    new Model(
      vocabCache = vocabCache,
      weightLookupTable = weightLookupTable,
      rnn = rnn,
      labels = data.labels
    )
  }

  def predict(model: Model, query: Query): PredictedResult = {
    val rawTrees = new TreeVectorizer().getTreesWithLabels(query.content, model.labels)
    val convertedTrees = rawTrees.map(Tree.fromTreeVectorizer(_, model.vocabCache, model.weightLookupTable))
    val sentiments = convertedTrees.map(model.rnn.predictClass(_))
    PredictedResult(sentiments = sentiments.toList)
  }
}

class Model(
  val vocabCache: VocabCache,
  val weightLookupTable: WeightLookupTable,
  val rnn: RNN,
  val labels: List[String]
) extends Serializable
