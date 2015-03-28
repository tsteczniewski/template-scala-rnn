package org.template.vanilla

import io.prediction.controller.IEngineFactory
import io.prediction.controller.Engine

case class Query(content: String) extends Serializable

case class PredictedResult(sentiments: List[Double]) extends Serializable

object VanillaEngine extends IEngineFactory {
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("rnn" -> classOf[Algorithm]),
      classOf[Serving])
  }
}
