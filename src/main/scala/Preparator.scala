package org.template.rntn

import grizzled.slf4j.Logger
import io.prediction.controller.PPreparator

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class Preparator
  extends PPreparator[TrainingData, PreparedData] {

  @transient lazy val logger = Logger[this.type]

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    // collect
    val labeledPhrases = trainingData.labeledPhrases.collect
    // length
    val maxLength = labeledPhrases.maxBy(_.phrase.length).phrase.length
    // parser
    val parser = Parser(maxLength)
    val maybeLabeledTrees: Array[Option[(Tree, Int)]] = labeledPhrases.map(labeledPhrase => {
      var tree: Tree = null
      try {
        val pennFormatted = parser.pennFormatted(labeledPhrase.phrase)
        tree = Tree.fromPennTreeBankFormat(pennFormatted)
      } catch {
        case e => logger.info(s"Couldn't parse ${labeledPhrase.phrase}.")
      }
      if (tree != null) Some((tree, labeledPhrase.sentiment))
      else None
    })

    val labeledTrees = maybeLabeledTrees.filter(_ != None).map(_.get)

    PreparedData(labeledTrees.toVector)
    //PreparedData(sc.parallelize(labeledTrees))
  }
}

case class PreparedData(
  labeledTrees : Vector[(Tree, Int)]
  //labeledTrees: RDD[(Tree, Int)]
) extends Serializable
