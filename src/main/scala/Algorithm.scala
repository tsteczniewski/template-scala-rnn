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
import org.nd4j.linalg.api.ndarray.INDArray

import scala.collection.JavaConversions._

import grizzled.slf4j.Logger

case class AlgorithmParams(
  val inSize: Integer
) extends Params

class Algorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def weightedMean(a: INDArray, b: INDArray, aq: Int, bq: Int): INDArray = {
    val a_aq = a.mul(aq)
    val b_bq = a.mul(bq)
    val sum = a_aq.add(b_bq)
    sum.mul(1.0 / (aq + bq))
  }

  def train(sc: SparkContext, data: PreparedData): Model = {
    val (vocabCache, weightLookupTable) = {
      val result = new SparkWord2Vec().train(data.phrases)
      (result.getFirst, result.getSecond)
    }
    val rawTrees = data.labeledPhrases.mapPartitions(labeledPhrases => {
      val treeVectorizer = new TreeVectorizer()
      labeledPhrases.map(labeledPhrase => (treeVectorizer.getTreesWithLabels(labeledPhrase.phrase, data.labels), labeledPhrase.sentiment))
    })
    val convertedTrees = rawTrees.flatMap(rawTreesWithSentiment => {
      val (rawTrees, sentiment) = rawTreesWithSentiment
      rawTrees.map(rawTree => new Pair[Tree, Integer](Tree.fromTreeVectorizer(rawTree, vocabCache, weightLookupTable), sentiment))
    })
    val collectedConvertedTrees = convertedTrees.glom()
    val judgesAndCombinators = collectedConvertedTrees.map(convertedTrees => {
      val rnnSettings = new RNN.Settings(ap.inSize, data.labels.length)
      val rnn = new RNN(rnnSettings)
      val convertedTreesList = convertedTrees.toList
      rnn.stochasticGradientDescent(convertedTreesList)
      (rnn.judge, rnn.combinator, 1)
    })
    val (judge, combinator, _) = judgesAndCombinators.reduce({
      case ((jl, cl, ql), (jr, cr, qr)) =>
        val j = weightedMean(jl, jr, ql, qr)
        val c = weightedMean(cl, cr, ql, qr)
        (j, c, ql + qr)
    })
    new Model(
      vocabCache = vocabCache,
      weightLookupTable = weightLookupTable,
      judge = judge,
      combinator = combinator,
      labels = data.labels
    )
  }

  def predict(model: Model, query: Query): PredictedResult = {
    val rawTrees = new TreeVectorizer().getTreesWithLabels(query.content, model.labels)
    val convertedTrees = rawTrees.map(Tree.fromTreeVectorizer(_, model.vocabCache, model.weightLookupTable))
    val rnnSettings = new RNN.Settings(ap.inSize, model.labels.length)
    val rnn = new RNN(rnnSettings, model.combinator, model.judge)
    val sentiments = convertedTrees.map(rnn.predictClass)
    PredictedResult(sentiments = sentiments.toList)
  }
}

class Model(
  val vocabCache: VocabCache,
  val weightLookupTable: WeightLookupTable,
  val judge: INDArray,
  val combinator: INDArray,
  val labels: List[String]
) extends Serializable
