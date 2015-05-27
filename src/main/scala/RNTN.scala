package org.template.rntn

import breeze.linalg.{argmax, sum, DenseVector, DenseMatrix}
import breeze.stats.distributions.Uniform
import scala.collection.mutable.Map
import scala.collection.mutable.HashMap
import scala.math.{exp, log, sqrt}

object RNTN {
  val randomDistribution = new Uniform(-1, 1)
  def randomMatrix(rows: Int, cols: Int) = DenseMatrix.rand(rows, cols, randomDistribution)
  def randomVector(rows: Int) = DenseVector.rand(rows, randomDistribution)

  def sigmoid(x: Double) = 1 / (1 + exp(-x))
  def sigmoidDerivative(x: Double) = sigmoid(x) * (1 - sigmoid(x))

  def logDerivative(x: Double) = 1 / x

  def regularization(m: DenseMatrix[Double]) = sum(m.map(x => x * x))

  def weightedMean(a: RNTN, b: RNTN, aq: Double, bq: Double): RNTN = {
    assert(a.inSize == b.inSize && a.outSize == b.outSize && a.alpha == b.alpha && a.regularizationCoeff == b.regularizationCoeff && a.useAdaGrad == b.useAdaGrad)
    val sum = aq + bq
    val rntn = new RNTN(a.inSize, a.outSize, a.alpha, a.regularizationCoeff, a.useAdaGrad)
    // judge
    rntn.judge = ((aq * a.judge) + (bq * b.judge)) / sum
    // combinator
    if (true) {
      val aKeySet = a.labelToCombinatorMap.keySet
      val bKeySet = b.labelToCombinatorMap.keySet
      for (key <- aKeySet.diff(bKeySet)) rntn.labelToCombinatorMap.put(key, (aq * a.labelToCombinatorMap(key)) / sum)
      for (key <- bKeySet.diff(bKeySet)) rntn.labelToCombinatorMap.put(key, (bq * b.labelToCombinatorMap(key)) / sum)
      for (key <- aKeySet.intersect(bKeySet)) rntn.labelToCombinatorMap.put(key, ((aq * a.labelToCombinatorMap(key)) + (bq * b.labelToCombinatorMap(key))) / sum)
    }
    // word vec
    if (true) {
      val aKeySet = a.wordToVecMap.keySet
      val bKeySet = b.wordToVecMap.keySet
      for (key <- aKeySet.diff(bKeySet)) rntn.wordToVecMap.put(key, (aq * a.wordToVecMap(key)) / sum)
      for (key <- bKeySet.diff(bKeySet)) rntn.wordToVecMap.put(key, (bq * b.wordToVecMap(key)) / sum)
      for (key <- aKeySet.intersect(bKeySet)) rntn.wordToVecMap.put(key, (aq * a.wordToVecMap(key) + bq * b.wordToVecMap(key)) / sum)
    }
    rntn
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

case class RNTN (
  inSize: Int,
  outSize: Int,
  alpha: Double,
  regularizationCoeff: Double,
  useAdaGrad: Boolean
) extends Serializable {

  var judge = RNTN.randomMatrix(outSize, inSize + 1)
  var labelToCombinatorMap = HashMap[(String, Int), DenseMatrix[Double]]()
  var wordToVecMap = HashMap[(String, String), DenseVector[Double]]()

  @transient var judgeDerivative: DenseMatrix[Double] = null
  @transient var labelToCombinatorDerivativeMap: Map[(String, Int), DenseMatrix[Double]] = null
  @transient var wordToVecDerivativeMap: Map[(String, String), DenseVector[Double]] = null

  @transient var judgeDerivativeHistory = RNTN.randomMatrix(outSize, inSize + 1)
  @transient var labelToCombinatorDerivativeHistoryMap = HashMap[(String, Int), DenseMatrix[Double]]()
  @transient var wordToVecDerivativeHistoryMap = HashMap[(String, String), DenseVector[Double]]()

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

  def forwardPropagateRegularizationError(influence: Double): Double = {
    var regularization = RNTN.regularization(judge)
    for(combinator <- labelToCombinatorMap.values) regularization += RNTN.regularization(combinator)
    for(vec <- wordToVecMap.values) regularization += RNTN.regularization(vec.asDenseMatrix)
    regularizationCoeff * influence * regularization
  }

  // calculates gradient only on touched matrices
  def backwardPropagateRegularizationError(influence: Double): Unit = {
    val coeff = regularizationCoeff * influence * 2.0
    judgeDerivative += coeff * judge
    for((key, combinatorDerivative) <- labelToCombinatorDerivativeMap) combinatorDerivative += coeff * labelToCombinatorMap.get(key).get
    for((key, vecDerivative) <- wordToVecDerivativeMap) vecDerivative += coeff * wordToVecMap.get(key).get
  }

  def applyGradientWithoutAdaGrad() = {
    judge -= RNTN.removeNans(alpha * judgeDerivative)
    for((key, combinatorDerivative) <- labelToCombinatorDerivativeMap) labelToCombinatorMap.get(key).get -= RNTN.removeNans(alpha * combinatorDerivative)
    for((key, vecDerivative) <- wordToVecDerivativeMap) wordToVecMap.get(key).get -= RNTN.removeNans(alpha * vecDerivative)
  }

  def applyGradientWithAdaGrad() = {
    judge -= RNTN.removeNans(alpha * RNTN.adaGrad(judgeDerivative, judgeDerivativeHistory))
    for((key, combinatorDerivative) <- labelToCombinatorDerivativeMap) {
      val derivativeHistory = labelToCombinatorDerivativeHistoryMap.getOrElseUpdate(key, DenseMatrix.zeros(combinatorDerivative.rows, combinatorDerivative.cols))
      labelToCombinatorMap.get(key).get -= RNTN.removeNans(alpha * RNTN.adaGrad(combinatorDerivative, derivativeHistory))
    }

    for((key, vecDerivative) <- wordToVecDerivativeMap) {
      val derivativeHistory = wordToVecDerivativeHistoryMap.getOrElseUpdate(key, DenseVector.zeros(inSize))
      wordToVecMap.get(key).get -= RNTN.removeNans(alpha * RNTN.adaGrad(vecDerivative, derivativeHistory))
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