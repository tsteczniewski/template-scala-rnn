package org.template.rntn

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params
import opennlp.tools.cmdline.parser.ParserTool
import opennlp.tools.parser.{ParserFactory, ParserModel}

import org.apache.spark.SparkContext

import grizzled.slf4j.Logger

case class AlgorithmParams(
  inSize: Int,
  outSize: Int,
  alpha: Double,
  regularizationCoeff: Double,
  useAdaGrad: Boolean,
  steps: Int
) extends Params

class Algorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): Model = {
    val rntn = new RNTN(ap.inSize, ap.outSize, ap.alpha, ap.regularizationCoeff, ap.useAdaGrad)
    for(i <- 0 until ap.steps) {
      logger.info(s"Iteration $i: ${rntn.forwardPropagateError(data.labeledTrees)}")
      rntn.fit(data.labeledTrees)
    }

    /*val rntns = data.labeledTrees.mapPartitions(labeledTrees => {
      val rntn = new RNTN(ap.inSize, ap.outSize, ap.alpha, ap.regularizationCoeff)
      val labeledTreesVector = labeledTrees.toVector
      var lastError = 0.0
      var currentError = 100000000.0
      var i = 0
      do {
      //for(i <- 0 to ap.steps) {
        logger.info(s"Iteration $i: ${rntn.forwardPropagateError(labeledTreesVector)}")
        i += 1
        rntn.fit(labeledTreesVector)
        lastError = currentError
        currentError = rntn.forwardPropagateError(labeledTreesVector)
      } while (currentError < lastError)
      Iterator((rntn, 1))
    })
    val (rntn, _) = rntns.reduce({case ((a, qa), (b, qb)) => (RNTN.weightedMean(a, b, qa, qb), qa + qb)})*/
    Model(rntn)
  }

  def predict(model: Model, query: Query): PredictedResult = {
    // parser
    val parser = Parser(query.content.length)
    val pennFormatted = parser.pennFormatted(query.content)
    val tree = Tree.fromPennTreeBankFormat(pennFormatted)
    println(tree)
    val forwardPropagatedTree = model.rntn.forwardPropagateTree(tree)
    val judgement = model.rntn.forwardPropagateJudgment(forwardPropagatedTree)
    PredictedResult(RNTN.maxClass(judgement))
  }
}

case class Model(
  rntn: RNTN
) extends Serializable
