package org.template.rnn

import breeze.linalg.{sum, norm, DenseVector, DenseMatrix}
import breeze.stats.distributions.Uniform
import scala.collection.mutable.Map
import scala.math.{exp, log}

object RNN {
  val randomDistribution = new Uniform(-1, 1)
  def randomMatrix(rows: Int, cols: Int) = DenseMatrix.rand(rows, cols, randomDistribution)
  def randomVector(rows: Int) = DenseVector.rand(rows, randomDistribution)

  def sigmoid(x: Double) = 1 / (1 + exp(-x))
  def sigmoidDerivative(x: Double) = sigmoid(x) * (1 - sigmoid(x))

  def logDerivative(x: Double) = 1 / x

  def regularization(m: DenseMatrix[Double]) = sum(m.map(x => x * x))
}

case class RNN(
  inSize: Int,
  outSize: Int,
  alpha: Double,
  regularizationCoeff: Double
) {
  var judge = RNN.randomMatrix(outSize, inSize + 1)
  var combinator = RNN.randomMatrix(inSize, 2 * inSize + 1)
  var wordToVecMap = Map[String, DenseVector[Double]]()

  var judgeDerivative: DenseMatrix[Double] = null
  var combinatorDerivative: DenseMatrix[Double] = null
  var wordToVecDerivativeMap: Map[String, DenseVector[Double]] = null

  def clearCache() = {
    judgeDerivative = DenseMatrix.zeros(judge.rows, judge.cols)
    combinatorDerivative = DenseMatrix.zeros(combinator.rows, combinator.cols)
    wordToVecDerivativeMap = Map[String, DenseVector[Double]]()
  }

  def label(i: Int) = {
    val m = DenseVector.zeros[Double](outSize)
    m(i) = 1
    m
  }

  def forwardPropagateTree(tree: Tree): ForwardPropagatedTree = tree match {
    case Node(l, r) =>
      val fpl@ForwardPropagatedTree(vl, _) = forwardPropagateTree(l)
      val fpr@ForwardPropagatedTree(vr, _) = forwardPropagateTree(r)
      val biased: DenseVector[Double] = DenseVector.vertcat(vl, vr, DenseVector.ones[Double](1))
      val transformed: DenseVector[Double] = combinator * biased
      ForwardPropagatedNode(fpl, fpr, transformed.map(RNN.sigmoid), transformed.map(RNN.sigmoidDerivative))
    case Leaf(w) =>
      val vec = wordToVecMap.getOrElseUpdate(w, RNN.randomVector(inSize))
      ForwardPropagatedLeaf(w, vec.map(RNN.sigmoid), vec.map(RNN.sigmoidDerivative))
  }

  def backwardPropagateTree(tree: ForwardPropagatedTree, y: DenseVector[Double]): Unit = tree match {
    case ForwardPropagatedNode(fpl@ForwardPropagatedTree(vl, _), fpr@ForwardPropagatedTree(vr, _), v, d) =>
      val z = y :* d
      val biased = DenseVector.vertcat(vl, vr, DenseVector.ones[Double](1))
      combinatorDerivative += z * biased.t
      val biasedDerivative: DenseVector[Double] = combinator.t * z
      backwardPropagateTree(fpl, biasedDerivative(0 to inSize - 1))
      backwardPropagateTree(fpr, biasedDerivative(inSize to 2 * inSize - 1))
    case ForwardPropagatedLeaf(w, _, d) =>
      val vecDerivative = wordToVecDerivativeMap.getOrElseUpdate(w, DenseVector.zeros(inSize))
      vecDerivative += y :* d
  }

  def forwardPropagateJudgment(tree: ForwardPropagatedTree) = tree match {
    case ForwardPropagatedTree(v, _) =>
      val biased = DenseVector.vertcat(v, DenseVector.ones[Double](1))
      val judged: DenseVector[Double] = judge * biased
      val activated = judged.map(RNN.sigmoid)
      activated
  }

  def backwardPropagateJudgement(tree: ForwardPropagatedTree, y: DenseVector[Double]): Unit = tree match {
    case ForwardPropagatedTree(v, _) =>
      val biased = DenseVector.vertcat(v, DenseVector.ones[Double](1))
      val judged: DenseVector[Double] = judge * biased
      val derivative: DenseVector[Double] = judged.map(RNN.sigmoidDerivative)
      val z = y :* derivative
      judgeDerivative += z * biased.t
      val biasedDerivative: DenseVector[Double] = judge.t * z
      backwardPropagateTree(tree, biasedDerivative(0 to inSize - 1))
  }

  def forwardPropagateError(tree: ForwardPropagatedTree, expected: DenseVector[Double]): Double = {
    val oneMinusExpected = DenseVector.ones[Double](outSize) - expected
    val actual = forwardPropagateJudgment(tree)
    val logActual = actual.map(log)
    val oneMinusActual = DenseVector.ones[Double](outSize) - actual
    val logOneMinusActual = oneMinusActual.map(log)
    -(expected.t * logActual + oneMinusExpected.t * logOneMinusActual)
  }

  def backwardPropagateError(tree: ForwardPropagatedTree, expected: DenseVector[Double]): Unit = {
    val oneMinusExpected = DenseVector.ones[Double](outSize) - expected
    val actual = forwardPropagateJudgment(tree)
    val logActualDerivative = actual.map(RNN.logDerivative)
    val oneMinusActual = DenseVector.ones[Double](outSize) - actual
    val logOneMinusActualDerivative = - oneMinusActual.map(RNN.logDerivative)
    val judgementDerivative = - ((expected :* logActualDerivative) + (oneMinusExpected :* logOneMinusActualDerivative))
    backwardPropagateJudgement(tree, judgementDerivative)
  }

  def forwardPropagateError(labeledTrees: Vector[(Tree, Int)]): Double =
    labeledTrees.foldLeft(0.0)((acc, labeledTree) => {
      val (t, i) = labeledTree
      acc + forwardPropagateError(forwardPropagateTree(t), label(i))
    })

  def forwardPropagateRegularizationError(): Double = {
    var regularization = RNN.regularization(judge) + RNN.regularization(combinator)
    for(vec <- wordToVecMap.values) regularization += RNN.regularization(vec.asDenseMatrix)
    regularizationCoeff * regularization
  }

  def backwardPropagateRegularizationError(): Unit = {
    val coeff = regularizationCoeff * 2.0
    judgeDerivative += coeff * judge
    combinatorDerivative += coeff * combinator
    for((word, vec) <- wordToVecMap) wordToVecDerivativeMap.get(word) match {
      case Some(vecDerivative) => vecDerivative += coeff * vec
    }
  }

  def fit(labeledTrees: Vector[(Tree, Int)]): Unit = {
    clearCache()
    for((t, i) <- labeledTrees) backwardPropagateError(forwardPropagateTree(t), label(i))
    backwardPropagateRegularizationError()
    judge -= alpha * judgeDerivative
    combinator -= alpha * combinatorDerivative
    for((word, vec) <- wordToVecMap) vec -= alpha * wordToVecDerivativeMap.getOrElse(word, DenseVector.zeros[Double](inSize))
  }
}