package org.template.rnn

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import org.apache.spark.SparkContext

import grizzled.slf4j.Logger


case class AlgorithmParams(
  inSize: Int,
  outSize: Int,
  alpha: Double,
  regularizationCoeff: Double,
  steps: Int
) extends Params

class Algorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): Model = {
    val rnn = new RNN(ap.inSize, ap.outSize, ap.alpha, ap.regularizationCoeff, data.labeledTrees)
    for(i <- 0 until ap.steps) {
      logger.info(s"Iteration $i: ${rnn.forwardPropagateError(data.labeledTrees)}")
      rnn.fit()
    }
    Model(rnn)
  }

  def predict(model: Model, query: Query): PredictedResult = {
    val parser = Parser(query.content.length)
    val pennFormatted = parser.pennFormatted(query.content)
    val tree = Tree.fromPennTreeBankFormat(pennFormatted)
    val forwardPropagatedTree = model.rnn.forwardPropagateTree(tree)
    val judgement = model.rnn.forwardPropagateJudgment(forwardPropagatedTree)
    PredictedResult(RNN.maxClass(judgement))
  }
}

case class Model(
  rnn: RNN
) extends Serializable
