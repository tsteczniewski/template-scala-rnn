package org.template.rnn

import breeze.linalg.{argmax, sum, DenseVector, DenseMatrix}
import breeze.stats.distributions.Uniform
import scala.collection.mutable.Map
import scala.collection.mutable.HashMap
import scala.math.{exp, log, sqrt}

object RNN {
  val randomDistribution = new Uniform(-1, 1)
  def randomMatrix(rows: Int, cols: Int) = DenseMatrix.rand(rows, cols, randomDistribution)
  def randomVector(rows: Int) = DenseVector.rand(rows, randomDistribution)

  def sigmoid(x: Double) = 1 / (1 + exp(-x))
  def sigmoidDerivative(x: Double) = sigmoid(x) * (1 - sigmoid(x))

  def logDerivative(x: Double) = 1 / x

  def regularization(m: DenseMatrix[Double]) = sum(m.map(x => x * x))

  def weightedMean(a: RNN, b: RNN, aq: Double, bq: Double): RNN = {
    assert(a.inSize == b.inSize && a.outSize == b.outSize && a.alpha == b.alpha && a.regularizationCoeff == b.regularizationCoeff && a.useAdaGrad == b.useAdaGrad)
    val sum = aq + bq
    val rnn = new RNN(a.inSize, a.outSize, a.alpha, a.regularizationCoeff, a.useAdaGrad)
    // judge
    rnn.judge = ((aq * a.judge) + (bq * b.judge)) / sum
    // combinator
    if (true) {
      val aKeySet = a.labelToCombinatorMap.keySet
      val bKeySet = b.labelToCombinatorMap.keySet
      for (key <- aKeySet.diff(bKeySet)) rnn.labelToCombinatorMap.put(key, (aq * a.labelToCombinatorMap(key)) / sum)
      for (key <- bKeySet.diff(bKeySet)) rnn.labelToCombinatorMap.put(key, (bq * b.labelToCombinatorMap(key)) / sum)
      for (key <- aKeySet.intersect(bKeySet)) rnn.labelToCombinatorMap.put(key, ((aq * a.labelToCombinatorMap(key)) + (bq * b.labelToCombinatorMap(key))) / sum)
    }
    // word vec
    if (true) {
      val aKeySet = a.wordToVecMap.keySet
      val bKeySet = b.wordToVecMap.keySet
      for (key <- aKeySet.diff(bKeySet)) rnn.wordToVecMap.put(key, (aq * a.wordToVecMap(key)) / sum)
      for (key <- bKeySet.diff(bKeySet)) rnn.wordToVecMap.put(key, (bq * b.wordToVecMap(key)) / sum)
      for (key <- aKeySet.intersect(bKeySet)) rnn.wordToVecMap.put(key, (aq * a.wordToVecMap(key) + bq * b.wordToVecMap(key)) / sum)
    }
    rnn
  }

  def maxClass(v: DenseVector[Double]): Int = argmax(v)

  def removeNans(m: DenseMatrix[Double]): DenseMatrix[Double] = m.map(x => if (x.isNaN) 0 else x)
  def removeNans(v: DenseVector[Double]): DenseVector[Double] = v.map(x => if (x.isNaN) 0 else x)

  val fudgeFactor = 1e-6
  def adaGrad(g: DenseMatrix[Double], gh: DenseMatrix[Double]): DenseMatrix[Double] = {
    gh += removeNans(g :* g)
    g :/ gh.map(x => fudgeFactor + sqrt(x))
  }
  def adaGrad(g: DenseVector[Double], gh: DenseVector[Double]): DenseVector[Double] = {
    gh += removeNans(g :* g)
    g :/ gh.map(x => fudgeFactor + sqrt(x))
  }
}

case class RNN (
  inSize: Int,
  outSize: Int,
  alpha: Double,
  regularizationCoeff: Double,
  useAdaGrad: Boolean
) extends Serializable {

  var judge = RNN.randomMatrix(outSize, inSize + 1)
  var labelToCombinatorMap = HashMap[(String, Int), DenseMatrix[Double]]()
  var wordToVecMap = HashMap[(String, String), DenseVector[Double]]()

  @transient var judgeGradient: DenseMatrix[Double] = null
  @transient var labelToCombinatorGradientMap: Map[(String, Int), DenseMatrix[Double]] = null
  @transient var wordToVecGradientMap: Map[(String, String), DenseVector[Double]] = null

  @transient var judgeGradientHistory = RNN.randomMatrix(outSize, inSize + 1)
  @transient var labelToCombinatorGradientHistoryMap = HashMap[(String, Int), DenseMatrix[Double]]()
  @transient var wordToVecGradientHistoryMap = HashMap[(String, String), DenseVector[Double]]()

  def clearCache() = {
    judgeGradient = DenseMatrix.zeros(judge.rows, judge.cols)
    labelToCombinatorGradientMap = Map[(String, Int), DenseMatrix[Double]]()
    wordToVecGradientMap = Map[(String, String), DenseVector[Double]]()
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
      val combinator = labelToCombinatorMap.getOrElseUpdate((label, children.length), RNN.randomMatrix(inSize, inSize * children.length + 1))
      val transformed: DenseVector[Double] = combinator * biased
      ForwardPropagatedNode(fpChildren, label, transformed.map(RNN.sigmoid), transformed.map(RNN.sigmoidDerivative))
    case Leaf(word, label) =>
      val vec = wordToVecMap.getOrElseUpdate((word, label), RNN.randomVector(inSize))
      ForwardPropagatedLeaf(word, label, vec.map(RNN.sigmoid), vec.map(RNN.sigmoidDerivative))
  }

  def backwardPropagateTree(tree: ForwardPropagatedTree, y: DenseVector[Double]): Unit = tree match {
    case ForwardPropagatedNode(children, label, _, d) =>
      val z = y :* d
      val vsChildren = for(child <- children) yield child.value
      val joined: DenseVector[Double] = DenseVector.vertcat(vsChildren:_*)
      val biased: DenseVector[Double] = DenseVector.vertcat(joined, DenseVector.ones(1))
      val combinator = labelToCombinatorMap.get((label, children.length)).get
      val combinatorGradient = labelToCombinatorGradientMap.getOrElseUpdate((label, children.length), DenseMatrix.zeros(inSize, inSize * children.length + 1))
      combinatorGradient += z * biased.t
      val biasedGradient: DenseVector[Double] = combinator.t * z
      for(i <- children.indices) backwardPropagateTree(children(i), biasedGradient(i * inSize to (i + 1) * inSize - 1))
    case ForwardPropagatedLeaf(word, label, _, d) =>
      val vecGradient = wordToVecGradientMap.getOrElseUpdate((word, label), DenseVector.zeros(inSize))
      vecGradient += y :* d
  }

  def forwardPropagateJudgment(tree: ForwardPropagatedTree) = tree match {
    case ForwardPropagatedTree(_, v, _) =>
      val biased = DenseVector.vertcat(v, DenseVector.ones[Double](1))
      val judged: DenseVector[Double] = judge * biased
      val activated = judged.map(RNN.sigmoid)
      activated
  }

  def backwardPropagateJudgement(tree: ForwardPropagatedTree, y: DenseVector[Double]): Unit = tree match {
    case ForwardPropagatedTree(_, v, _) =>
      val biased = DenseVector.vertcat(v, DenseVector.ones[Double](1))
      val judged: DenseVector[Double] = judge * biased
      val gradient: DenseVector[Double] = judged.map(RNN.sigmoidDerivative)
      val z = y :* gradient
      judgeGradient += z * biased.t
      val biasedGradient: DenseVector[Double] = judge.t * z
      backwardPropagateTree(tree, biasedGradient(0 to inSize - 1))
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
    val logActualGradient = actual.map(RNN.logDerivative)
    val oneMinusActual = DenseVector.ones[Double](outSize) - actual
    val logOneMinusActualGradient = - oneMinusActual.map(RNN.logDerivative)
    val judgementGradient = - ((expected :* logActualGradient) + (oneMinusExpected :* logOneMinusActualGradient))
    backwardPropagateJudgement(tree, judgementGradient)
  }

  def forwardPropagateError(labeledTrees: Vector[(Tree, Int)]): Double =
    labeledTrees.foldLeft(0.0)((acc, labeledTree) => {
      val (t, i) = labeledTree
      acc + forwardPropagateError(forwardPropagateTree(t), label(i))
    })

  def forwardPropagateRegularizationError(influence: Double): Double = {
    var regularization = RNN.regularization(judge)
    for(combinator <- labelToCombinatorMap.values) regularization += RNN.regularization(combinator)
    for(vec <- wordToVecMap.values) regularization += RNN.regularization(vec.asDenseMatrix)
    regularizationCoeff * influence * regularization
  }

  // calculates gradient only on touched matrices
  def backwardPropagateRegularizationError(influence: Double): Unit = {
    val coeff = regularizationCoeff * influence * 2.0
    judgeGradient += coeff * judge
    for((key, combinatorGradient) <- labelToCombinatorGradientMap) combinatorGradient += coeff * labelToCombinatorMap.get(key).get
    for((key, vecGradient) <- wordToVecGradientMap) vecGradient += coeff * wordToVecMap.get(key).get
  }

  def applyGradientWithoutAdaGrad() = {
    judge -= RNN.removeNans(alpha * judgeGradient)
    for((key, combinatorGradient) <- labelToCombinatorGradientMap) labelToCombinatorMap.get(key).get -= RNN.removeNans(alpha * combinatorGradient)
    for((key, vecGradient) <- wordToVecGradientMap) wordToVecMap.get(key).get -= RNN.removeNans(alpha * vecGradient)
  }

  def applyGradientWithAdaGrad() = {
    judge -= RNN.removeNans(alpha * RNN.adaGrad(judgeGradient, judgeGradientHistory))
    for((key, combinatorGradient) <- labelToCombinatorGradientMap) {
      val gradientHistory = labelToCombinatorGradientHistoryMap.getOrElseUpdate(key, DenseMatrix.zeros(combinatorGradient.rows, combinatorGradient.cols))
      labelToCombinatorMap.get(key).get -= RNN.removeNans(alpha * RNN.adaGrad(combinatorGradient, gradientHistory))
    }

    for((key, vecGradient) <- wordToVecGradientMap) {
      val gradientHistory = wordToVecGradientHistoryMap.getOrElseUpdate(key, DenseVector.zeros(inSize))
      wordToVecMap.get(key).get -= RNN.removeNans(alpha * RNN.adaGrad(vecGradient, gradientHistory))
    }
  }

  def fit(labeledTrees: Vector[(Tree, Int)]): Unit = {
    clearCache()
    for((t, i) <- labeledTrees) backwardPropagateError(forwardPropagateTree(t), label(i))
    backwardPropagateRegularizationError(1)
    if (useAdaGrad) applyGradientWithAdaGrad()
    else applyGradientWithoutAdaGrad()
  }

  def stochasticGradientDescent(labeledTrees: Vector[(Tree, Int)]): Unit = {
    for((t, i) <- labeledTrees) {
      clearCache()
      backwardPropagateError(forwardPropagateTree(t), label(i))
      backwardPropagateRegularizationError(1 / labeledTrees.length)
      if (useAdaGrad) applyGradientWithAdaGrad()
      else applyGradientWithoutAdaGrad()
    }
  }
}