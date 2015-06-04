package org.template.rnn

import breeze.linalg.{Vector => _, _}
import org.apache.spark.rdd.RDD
import scala.collection.mutable.Map
import breeze.stats.distributions.Uniform
import scala.math.{exp, log, sqrt}

object RNN {
  val randomDistribution = new Uniform(-1, 1)
  def randomMatrix(rows: Int, cols: Int) = DenseMatrix.rand(rows, cols, randomDistribution)
  def randomVector(rows: Int) = DenseVector.rand(rows, randomDistribution)

  def sigmoid(x: Double) = 1 / (1 + exp(-x))
  def sigmoidDerivative(x: Double) = sigmoid(x) * (1 - sigmoid(x))

  def logDerivative(x: Double) = 1 / x

  def regularization(m: DenseMatrix[Double]) = sum(m.map(x => x * x))
  def regularization(m: DenseVector[Double]) = sum(m.map(x => x * x))

  def maxClass(v: DenseVector[Double]): Int = argmax(v)

  def clearNans(m: DenseVector[Double]): DenseVector[Double] = m.map(x => if (x.isNaN) 0 else x)
  def clearNans(m: DenseMatrix[Double]): DenseMatrix[Double] = m.map(x => if (x.isNaN) 0 else x)

  val fudgeFactor = 1e-6
  def adaGrad(g: DenseMatrix[Double], gh: DenseMatrix[Double]): DenseMatrix[Double] = {
    gh += clearNans(g :* g)
    g :/ gh.map(x => fudgeFactor + sqrt(x))
  }
  def adaGrad(g: DenseVector[Double], gh: DenseVector[Double]): DenseVector[Double] = {
    gh += clearNans(g :* g)
    g :/ gh.map(x => fudgeFactor + sqrt(x))
  }

  case class Gradient(
    judgeGradient: DenseMatrix[Double],
    labelToCombinatorGradientMap: Map[(String, Int), DenseMatrix[Double]],
    wordToVecGradientMap: Map[(String, String), DenseVector[Double]]
  ) extends Serializable

  def mergeCombinatorGradientMaps(a: Map[(String, Int), DenseMatrix[Double]], b: Map[(String, Int), DenseMatrix[Double]]): Map[(String, Int), DenseMatrix[Double]] = {
    // let b be smaller map
    if(a.size < b.size) mergeCombinatorGradientMaps(b, a)
    for (key <- b.keySet intersect a.keySet) a.get(key).get += b.get(key).get
    for (key <- b.keySet diff a.keySet) a.put(key, b.get(key).get)
    a
  }
  def mergeVecGradientMaps(a: Map[(String, String), DenseVector[Double]], b: Map[(String, String), DenseVector[Double]]): Map[(String, String), DenseVector[Double]] = {
    // let b be smaller map
    if(a.size < b.size) mergeVecGradientMaps(b, a)
    for (key <- b.keySet intersect a.keySet) a.get(key).get += b.get(key).get
    for (key <- b.keySet diff a.keySet) a.put(key, b.get(key).get)
    a
  }
  def mergeGradients(a: Gradient, b: Gradient) =
    Gradient(
      a.judgeGradient + b.judgeGradient,
      mergeCombinatorGradientMaps(a.labelToCombinatorGradientMap, b.labelToCombinatorGradientMap),
      mergeVecGradientMaps(a.wordToVecGradientMap, b.wordToVecGradientMap)
    )
}

case class RNN (
  inSize: Int,
  outSize: Int,
  alpha: Double,
  regularizationCoeff: Double,
  @transient labeledTrees: RDD[(Tree, Int)]
) extends Serializable {

  var judge = RNN.randomMatrix(outSize, inSize + 1)
  var labelToCombinatorMap = Map[(String, Int), DenseMatrix[Double]]()
  var wordToVecMap = Map[(String, String), DenseVector[Double]]()

  @transient var judgeGradientHistory = RNN.randomMatrix(outSize, inSize + 1)
  @transient var labelToCombinatorGradientHistoryMap = Map[(String, Int), DenseMatrix[Double]]()
  @transient var wordToVecGradientHistoryMap = Map[(String, String), DenseVector[Double]]()

  def initializeMaps(trees: Array[Tree]): Unit = {
    val nodeTypes = trees.map(Tree.nodeTypes(_)).reduce((a, b) => a union b)
    val leafTypes = trees.map(Tree.leafTypes(_)).reduce((a, b) => a union b)
    for ((label, childrenLength) <- nodeTypes) {
      labelToCombinatorMap.put((label, childrenLength), RNN.randomMatrix(inSize, inSize * childrenLength + 1))
      labelToCombinatorGradientHistoryMap.put((label, childrenLength), DenseMatrix.zeros[Double](inSize, inSize * childrenLength + 1))
    }
    for ((word, label) <- leafTypes) {
      wordToVecMap.put((word, label), RNN.randomVector(inSize))
      wordToVecGradientHistoryMap.put((word, label), DenseVector.zeros[Double](inSize))
    }
  }

  // initialize maps
  if(labeledTrees != null) // for tests
    initializeMaps(labeledTrees.map(_._1).collect())

  def label(i: Int) = {
    val m = DenseVector.zeros[Double](outSize)
    m(i) = 1
    m
  }

  def forwardPropagateTree(tree: Tree): ForwardPropagatedTree = tree match {
    case Node(children, label) =>
      val fpChildren = for (child <- children) yield forwardPropagateTree(child)
      val vsChildren = for (fpChild <- fpChildren) yield fpChild.value
      val joined: DenseVector[Double] = DenseVector.vertcat(vsChildren: _*)
      val biased: DenseVector[Double] = DenseVector.vertcat(joined, DenseVector.ones(1))
      val combinator = labelToCombinatorMap.getOrElse((label, children.length), RNN.randomMatrix(inSize, inSize * children.length + 1))
    val transformed: DenseVector[Double] = combinator * biased
      ForwardPropagatedNode(fpChildren, label, transformed.map(RNN.sigmoid), transformed.map(RNN.sigmoidDerivative))
    case Leaf(word, label) =>
      val vec = wordToVecMap.getOrElse((word, label), RNN.randomVector(inSize))
      ForwardPropagatedLeaf(word, label, vec.map(RNN.sigmoid), vec.map(RNN.sigmoidDerivative))
  }

  def backwardPropagateTree(tree: ForwardPropagatedTree, y: DenseVector[Double], gradient: RNN.Gradient): Unit = tree match {
    case ForwardPropagatedNode(children, label, _, d) =>
      val z = y :* d
      // combinator
      val vsChildren = for (child <- children) yield child.value
      val joined: DenseVector[Double] = DenseVector.vertcat(vsChildren: _*)
      val biased: DenseVector[Double] = DenseVector.vertcat(joined, DenseVector.ones(1))
      val combinator = labelToCombinatorMap.get((label, children.length)).get
      // update gradient
      val combinatorGradient = gradient.labelToCombinatorGradientMap.getOrElseUpdate((label, children.length), DenseMatrix.zeros(inSize, inSize * children.length + 1))
      // backward propagate children
      val biasedGradient: DenseVector[Double] = combinator.t * z
      for (i <- children.indices) backwardPropagateTree(children(i), biasedGradient(i * inSize to (i + 1) * inSize - 1), gradient)
      combinatorGradient += z * biased.t
    case ForwardPropagatedLeaf(word, label, _, d) =>
      val vecGradient = gradient.wordToVecGradientMap.getOrElseUpdate((word, label), DenseVector.zeros[Double](inSize))
      vecGradient += y :* d
  }

  def forwardPropagateJudgment(tree: ForwardPropagatedTree) = tree match {
    case ForwardPropagatedTree(_, v, _) =>
      val biased = DenseVector.vertcat(v, DenseVector.ones[Double](1))
      val judged: DenseVector[Double] = judge * biased
      val activated = judged.map(RNN.sigmoid)
      activated
  }

  def backwardPropagateJudgement(tree: ForwardPropagatedTree, y: DenseVector[Double], gradient: RNN.Gradient) = tree match {
    case ForwardPropagatedTree(_, v, _) =>
      val biased = DenseVector.vertcat(v, DenseVector.ones[Double](1))
      val judged: DenseVector[Double] = judge * biased
      val activatedGradient: DenseVector[Double] = judged.map(RNN.sigmoidDerivative)
      val z = y :* activatedGradient
      gradient.judgeGradient += z * biased.t
      val biasedGradient: DenseVector[Double] = judge.t * z
      backwardPropagateTree(tree, biasedGradient(0 to inSize - 1), gradient)
  }

  def forwardPropagateError(tree: ForwardPropagatedTree, expected: DenseVector[Double]) = {
    val oneMinusExpected = DenseVector.ones[Double](outSize) - expected
    val actual = forwardPropagateJudgment(tree)
    val logActual = actual.map(log)
    val oneMinusActual = DenseVector.ones[Double](outSize) - actual
    val logOneMinusActual = oneMinusActual.map(log)
    -(expected.t * logActual + oneMinusExpected.t * logOneMinusActual)
  }

  def backwardPropagateError(tree: ForwardPropagatedTree, expected: DenseVector[Double], gradient: RNN.Gradient) = {
    val oneMinusExpected = DenseVector.ones[Double](outSize) - expected
    val actual = forwardPropagateJudgment(tree)
    val logActualGradient = actual.map(RNN.logDerivative)
    val oneMinusActual = DenseVector.ones[Double](outSize) - actual
    val logOneMinusActualGradient = -oneMinusActual.map(RNN.logDerivative)
    val judgementGradient = -((expected :* logActualGradient) + (oneMinusExpected :* logOneMinusActualGradient))
    backwardPropagateJudgement(tree, judgementGradient, gradient)
  }

  def forwardPropagateError(labeledTrees: RDD[(Tree, Int)]): Double =
    labeledTrees.map(labeledTree => forwardPropagateError(forwardPropagateTree(labeledTree._1), label(labeledTree._2))).reduce((x, y) => x + y)

  def forwardPropagateRegularizationError(influence: Double): Double = {
    var regularization = RNN.regularization(judge)
    for (combinator <- labelToCombinatorMap.values) regularization += RNN.regularization(combinator)
    for (vec <- wordToVecMap.values) regularization += RNN.regularization(vec)
    regularizationCoeff * influence * regularization
  }

  def backwardPropagateRegularizationError(gradient: RNN.Gradient): Unit = {
    val coeff = regularizationCoeff * 2.0
    gradient.judgeGradient += coeff * judge
    for ((key, combinatorGradient) <- gradient.labelToCombinatorGradientMap) combinatorGradient += coeff * labelToCombinatorMap.get(key).get
    for ((key, vecGradient) <- gradient.wordToVecGradientMap) vecGradient += coeff * wordToVecMap.get(key).get
  }

  def applyGradient(gradient: RNN.Gradient): Unit = {
    judge -= RNN.clearNans(alpha * RNN.adaGrad(gradient.judgeGradient, judgeGradientHistory))
    for ((key, combinatorGradient) <- gradient.labelToCombinatorGradientMap) {
      val gradientHistory = labelToCombinatorGradientHistoryMap.get(key).get
      labelToCombinatorMap.get(key).get -= RNN.clearNans(alpha * RNN.adaGrad(combinatorGradient, gradientHistory))
    }

    for ((key, vecGradient) <- gradient.wordToVecGradientMap) {
      val gradientHistory = wordToVecGradientHistoryMap.get(key).get
      wordToVecMap.get(key).get -= RNN.clearNans(alpha * RNN.adaGrad(vecGradient, gradientHistory))
    }
  }

  def fit(): Unit = {
    val gradient = labeledTrees.mapPartitions(labeledTrees => {
      val gradient = RNN.Gradient(DenseMatrix.zeros[Double](outSize, inSize + 1), Map.empty, Map.empty)
      labeledTrees.foreach(labeledTree => backwardPropagateError(forwardPropagateTree(labeledTree._1), label(labeledTree._2), gradient))
      Iterator(gradient)
    }).reduce(RNN.mergeGradients)
    backwardPropagateRegularizationError(gradient)
    applyGradient(gradient)
  }
}