package org.template.rntn

import opennlp.tools.cmdline.parser.ParserTool
import opennlp.tools.parser.{ParserFactory, ParserModel}

object Parser {
  val stream = getClass.getResource("/en-parser-chunking.bin").openStream()
  val parserModel = new ParserModel(stream)
  stream.close()

  val parser = ParserFactory.create(Parser.parserModel)

  val resizeCoeff = 100

  def buffer(length: Int) = new StringBuffer(resizeCoeff * length)
}

case class Parser(
  maxLength: Int
) {
  val buffer = Parser.buffer(maxLength)

  def pennFormatted(phrase: String): String = {
    ParserTool.parseLine(phrase, Parser.parser, 1)(0).show(buffer)
    val pennFormattedPhrase = buffer.toString
    buffer.delete(0,  buffer.length())
    pennFormattedPhrase
  }
}