package org.template.rntn

import io.prediction.controller.PPreparator

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class Preparator
  extends PPreparator[TrainingData, PreparedData] {

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    // collect
    val labeledPhrases = trainingData.labeledPhrases.collect
    // length
    val maxLength = labeledPhrases.maxBy(_.phrase.length).phrase.length
    // parser
    val parser = Parser(maxLength)
    val labeledTrees = labeledPhrases.map(labeledPhrase => {
      val pennFormatted = parser.pennFormatted(labeledPhrase.phrase)
      val tree = Tree.fromPennTreeBankFormat(pennFormatted)
      (tree, labeledPhrase.sentiment)
    })

    //PreparedData(labeledTrees.toVector)
    PreparedData(sc.parallelize(labeledTrees))
  }
}

case class PreparedData(
  //labeledTrees : Vector[(Tree, Int)]
  labeledTrees: RDD[(Tree, Int)]
) extends Serializable
