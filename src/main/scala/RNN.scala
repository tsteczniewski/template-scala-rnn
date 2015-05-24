package org.template.rnn

import breeze.linalg.{sum, DenseVector, DenseMatrix}
import breeze.stats.distributions.Uniform
import scala.collection.mutable.Map
import scala.math.{exp, log}

object RNTN {
  val randomDistribution = new Uniform(-1, 1)
  def randomMatrix(rows: Int, cols: Int) = DenseMatrix.rand(rows, cols, randomDistribution)
  def randomVector(rows: Int) = DenseVector.rand(rows, randomDistribution)

  def sigmoid(x: Double) = 1 / (1 + exp(-x))
  def sigmoidDerivative(x: Double) = sigmoid(x) * (1 - sigmoid(x))

  def logDerivative(x: Double) = 1 / x

  def regularization(m: DenseMatrix[Double]) = sum(m.map(x => x * x))
}

case class RNTN(
  inSize: Int,
  outSize: Int,
  alpha: Double,
  regularizationCoeff: Double
) {
  var judge = RNTN.randomMatrix(outSize, inSize + 1)
  var labelToCombinatorMap = Map[(String, Int), DenseMatrix[Double]]()
  var wordToVecMap = Map[(String, String), DenseVector[Double]]()

  var judgeDerivative: DenseMatrix[Double] = null
  var labelToCombinatorDerivativeMap: Map[(String, Int), DenseMatrix[Double]] = null
  var wordToVecDerivativeMap: Map[(String, String), DenseVector[Double]] = null

  def clearCache() = {
    judgeDerivative = DenseMatrix.zeros(judge.rows, judge.cols)
    labelToCombinatorDerivativeMap = Map[(String, Int), DenseMatrix[Double]]()
    wordToVecDerivativeMap = Map[(String, String), DenseVector[Double]]()
  }

  def label(i: Int) = {
    val m = DenseVector.zeros[Double](outSize)
    m(i) = 1
    m
  }

  def forwardPropagateTree(tree: Tree): ForwardPropagatedTree = tree match {
    case Node(children, label) =>
      val fpChildren = for(child <- children) yield forwardPropagateTree(child)
      val vsChildren = for(fpChild <- fpChildren) yield fpChild.value
      val joined: DenseVector[Double] = DenseVector.vertcat(vsChildren:_*)
      val biased: DenseVector[Double] = DenseVector.vertcat(joined, DenseVector.ones(1))
      val combinator = labelToCombinatorMap.getOrElseUpdate((label, children.length), RNTN.randomMatrix(inSize, inSize * children.length + 1))
      val transformed: DenseVector[Double] = combinator * biased
      ForwardPropagatedNode(fpChildren, label, transformed.map(RNTN.sigmoid), transformed.map(RNTN.sigmoidDerivative))
    case Leaf(word, label) =>
      val vec = wordToVecMap.getOrElseUpdate((word, label), RNTN.randomVector(inSize))
      ForwardPropagatedLeaf(word, label, vec.map(RNTN.sigmoid), vec.map(RNTN.sigmoidDerivative))
  }

  def backwardPropagateTree(tree: ForwardPropagatedTree, y: DenseVector[Double]): Unit = tree match {
    case ForwardPropagatedNode(children, label, _, d) =>
      val z = y :* d
      val vsChildren = for(child <- children) yield child.value
      val joined: DenseVector[Double] = DenseVector.vertcat(vsChildren:_*)
      val biased: DenseVector[Double] = DenseVector.vertcat(joined, DenseVector.ones(1))
      val combinator = labelToCombinatorMap.get((label, children.length)).get
      val combinatorDerivative = labelToCombinatorDerivativeMap.getOrElseUpdate((label, children.length), DenseMatrix.zeros(inSize, inSize * children.length + 1))
      combinatorDerivative += z * biased.t
      val biasedDerivative: DenseVector[Double] = combinator.t * z
      for(i <- children.indices) backwardPropagateTree(children(i), biasedDerivative(i * inSize to (i + 1) * inSize - 1))
    case ForwardPropagatedLeaf(word, label, _, d) =>
      val vecDerivative = wordToVecDerivativeMap.getOrElseUpdate((word, label), DenseVector.zeros(inSize))
      vecDerivative += y :* d
  }

  def forwardPropagateJudgment(tree: ForwardPropagatedTree) = tree match {
    case ForwardPropagatedTree(_, v, _) =>
      val biased = DenseVector.vertcat(v, DenseVector.ones[Double](1))
      val judged: DenseVector[Double] = judge * biased
      val activated = judged.map(RNTN.sigmoid)
      activated
  }

  def backwardPropagateJudgement(tree: ForwardPropagatedTree, y: DenseVector[Double]): Unit = tree match {
    case ForwardPropagatedTree(_, v, _) =>
      val biased = DenseVector.vertcat(v, DenseVector.ones[Double](1))
      val judged: DenseVector[Double] = judge * biased
      val derivative: DenseVector[Double] = judged.map(RNTN.sigmoidDerivative)
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
    val logActualDerivative = actual.map(RNTN.logDerivative)
    val oneMinusActual = DenseVector.ones[Double](outSize) - actual
    val logOneMinusActualDerivative = - oneMinusActual.map(RNTN.logDerivative)
    val judgementDerivative = - ((expected :* logActualDerivative) + (oneMinusExpected :* logOneMinusActualDerivative))
    backwardPropagateJudgement(tree, judgementDerivative)
  }

  def forwardPropagateError(labeledTrees: Vector[(Tree, Int)]): Double =
    labeledTrees.foldLeft(0.0)((acc, labeledTree) => {
      val (t, i) = labeledTree
      acc + forwardPropagateError(forwardPropagateTree(t), label(i))
    })

  def forwardPropagateRegularizationError(): Double = {
    var regularization = RNTN.regularization(judge)
    for(combinator <- labelToCombinatorMap.values) regularization += RNTN.regularization(combinator)
    for(vec <- wordToVecMap.values) regularization += RNTN.regularization(vec.asDenseMatrix)
    regularizationCoeff * regularization
  }

  def backwardPropagateRegularizationError(): Unit = {
    val coeff = regularizationCoeff * 2.0
    judgeDerivative += coeff * judge
    for((key, combinator) <- labelToCombinatorMap) labelToCombinatorDerivativeMap.get(key).get += coeff * combinator
    for((word, vec) <- wordToVecMap) wordToVecDerivativeMap.get(word).get += coeff * vec
  }

  def fit(labeledTrees: Vector[(Tree, Int)]): Unit = {
    clearCache()
    for((t, i) <- labeledTrees) backwardPropagateError(forwardPropagateTree(t), label(i))
    backwardPropagateRegularizationError()
    judge -= alpha * judgeDerivative
    for((key, combinator) <- labelToCombinatorMap) combinator -= alpha * labelToCombinatorDerivativeMap.get(key).get
    for((word, vec) <- wordToVecMap) vec -= alpha * wordToVecDerivativeMap.get(word).get
  }
}