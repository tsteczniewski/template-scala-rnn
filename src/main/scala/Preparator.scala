package org.template.rntn

import io.prediction.controller.PPreparator
import io.prediction.data.storage.Event
import opennlp.tools.cmdline.parser.ParserTool
import opennlp.tools.parser.{ParserFactory, ParserModel}

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

class Preparator
  extends PPreparator[TrainingData, PreparedData] {

  val lengthCoeff = 20

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    val labeledTrees = trainingData.labeledPhrases.mapPartitions(labeledPhrases => {
      // get model
      val stream = getClass.getResource("/en-parser-chunking.bin").openStream()
      val parserModel = new ParserModel(stream)
      stream.close()
      // create buffer
      val maxLength = labeledPhrases.maxBy(_.phrase.length).phrase.length
      val buffer = new StringBuffer(1000)
      // create parser
      val parser = ParserFactory.create(parserModel)
      labeledPhrases.map(labeledPhrase => {
        ParserTool.parseLine(labeledPhrase.phrase, parser, 1)(0).show(buffer)
        val pennTreeBankFormattedPhrase = buffer.toString
        buffer.delete(0,  buffer.length())
        (Tree.fromPennTreeBankFormat(pennTreeBankFormattedPhrase), labeledPhrase.sentiment)
      })
    })

    PreparedData(labeledTrees)
  }
}

case class PreparedData(
  labeledTrees : RDD[(Tree, Int)]
) extends Serializable
