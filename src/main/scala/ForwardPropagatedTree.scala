package org.template.rnn

import breeze.linalg.DenseVector

abstract class ForwardPropagatedTree(
  val value: DenseVector[Double],
  val derivative: DenseVector[Double]
)

object ForwardPropagatedTree {
  def unapply(fpd: ForwardPropagatedTree) = Some((fpd.value, fpd.derivative))
}

case class ForwardPropagatedNode(
  left: ForwardPropagatedTree,
  right: ForwardPropagatedTree,
  override val value: DenseVector[Double],
  override val derivative: DenseVector[Double]
) extends ForwardPropagatedTree(value, derivative)

case class ForwardPropagatedLeaf(
  word: String,
  override val value: DenseVector[Double],
  override val derivative: DenseVector[Double]
) extends ForwardPropagatedTree(value, derivative)