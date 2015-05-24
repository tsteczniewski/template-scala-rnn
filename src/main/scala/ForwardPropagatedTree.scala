package org.template.rnn

import breeze.linalg.DenseVector

abstract class ForwardPropagatedTree(
  val label: String,
  val value: DenseVector[Double],
  val derivative: DenseVector[Double]
)

object ForwardPropagatedTree {
  def unapply(fpd: ForwardPropagatedTree) = Some((fpd.label, fpd.value, fpd.derivative))
}

case class ForwardPropagatedNode(
  children: List[ForwardPropagatedTree],
  override val label: String,
  override val value: DenseVector[Double],
  override val derivative: DenseVector[Double]
) extends ForwardPropagatedTree(label, value, derivative)

case class ForwardPropagatedLeaf(
  word: String,
  override val label: String,
  override val value: DenseVector[Double],
  override val derivative: DenseVector[Double]
) extends ForwardPropagatedTree(label, value, derivative)