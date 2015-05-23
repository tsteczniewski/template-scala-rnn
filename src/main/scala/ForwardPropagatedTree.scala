package org.template.rnn

import breeze.linalg.DenseMatrix

abstract case class ForwardPropagatedTree(
  value: DenseMatrix[Double],
  derivative: DenseMatrix[Double]
)

case class ForwardPropagatedNode(
  left: ForwardPropagatedTree,
  right: ForwardPropagatedTree,
  override val value: DenseMatrix[Double],
  override val derivative: DenseMatrix[Double]
) extends ForwardPropagatedTree(value, derivative)

case class ForwardPropagatedLeaf(
  word: String,
  override val value: DenseMatrix[Double],
  override val derivative: DenseMatrix[Double]
) extends ForwardPropagatedTree(value, derivative)