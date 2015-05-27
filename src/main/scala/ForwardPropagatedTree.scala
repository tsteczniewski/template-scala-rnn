package org.template.rnn

import breeze.linalg.DenseVector

abstract class ForwardPropagatedTree(
  val label: String,
  val value: DenseVector[Double],
  val gradient: DenseVector[Double]
) extends Serializable

object ForwardPropagatedTree {
  def unapply(fpd: ForwardPropagatedTree) = Some((fpd.label, fpd.value, fpd.gradient))
}

case class ForwardPropagatedNode(
  children: List[ForwardPropagatedTree],
  override val label: String,
  override val value: DenseVector[Double],
  override val gradient: DenseVector[Double]
) extends ForwardPropagatedTree(label, value, gradient)

case class ForwardPropagatedLeaf(
  word: String,
  override val label: String,
  override val value: DenseVector[Double],
  override val gradient: DenseVector[Double]
) extends ForwardPropagatedTree(label, value, gradient)