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
      val rnnSettings = new RNN.Settings(ap.inSize)
      val rnn = new RNN(rnnSettings)
      val convertedTreesList = convertedTrees.toList
      rnn.stochasticGradientDescent(convertedTreesList)
      (rnn.judge, rnn.combinator, 1.0)
    })
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
    val rnnSettings = new RNN.Settings(ap.inSize)
    val rnn = new RNN(rnnSettings, model.combinator, model.judge)
    val convertedTrees = rawTrees.map(Tree.fromTreeVectorizer(_, model.vocabCache, model.weightLookupTable))
    val sentiments = convertedTrees.map(rnn.predict(_))
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
