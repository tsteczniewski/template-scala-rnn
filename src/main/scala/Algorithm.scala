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
  steps: Int
) extends Params

class Algorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): Model = {
    val rntns = data.labeledTrees.mapPartitions(labeledTrees => {
      val rntn = new RNTN(ap.inSize, ap.outSize, ap.alpha, ap.regularizationCoeff)
      val labeledTreesVector = labeledTrees.toVector
      for(i <- 0 to ap.steps) {
        logger.info(s"Iteration $i: ${rntn.forwardPropagateError(labeledTreesVector)}")
        rntn.fit(labeledTreesVector)
      }
      Iterator((rntn, 1))
    })
    val (rntn, _) = rntns.reduce({case ((a, qa), (b, qb)) => (RNTN.weightedMean(a, b, qa, qb), qa + qb)})
    Model(rntn)
  }

  def predict(model: Model, query: Query): PredictedResult = {
    val stream = getClass.getResource("/en-parser-chunking.bin").openStream()
    val parserModel = new ParserModel(stream)
    stream.close()
    // create buffer
    val buffer = new StringBuffer(20 * query.content.length)
    // create parser
    val parser = ParserFactory.create(parserModel)
    ParserTool.parseLine(query.content, parser, 1)(0).show(buffer)
    val pennTreeBankFormattedPhrase = buffer.toString
    val tree = Tree.fromPennTreeBankFormat(pennTreeBankFormattedPhrase)
    val forwardPropagatedTree = model.rntn.forwardPropagateTree(tree)
    val judgement = model.rntn.forwardPropagateJudgment(forwardPropagatedTree)
    PredictedResult(RNTN.maxClass(judgement))
  }
}

case class Model(
  rntn: RNTN
) extends Serializable
