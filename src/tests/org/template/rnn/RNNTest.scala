package org.template.rnn

import breeze.linalg.{DenseVector, DenseMatrix}

class RNNTest extends org.scalatest.FunSuite {

  def equal(x: Double, y: Double): Boolean = {
    val eps = 0.000001
    (x - eps <= y && y <= x + eps) || (1.0 - eps <= x / y && x / y <= 1.0 + eps)
  }

  test("test sigmoid") {
    val xs = List(-1.0, 0.0, 1.0, 2.0, 3.0)

    // 1 / (1 + e^(-x)) for x in -1, 0, 1, 2, 3
    val sigmoidsOfXs = List(0.268941, 0.5, 0.731059, 0.880797, 0.952574)
    for((x, sigmoidOfX) <- xs.zip(sigmoidsOfXs)) {
      assert(equal(RNN.sigmoid(x), sigmoidOfX))
    }
  }

  test("test sigmoid derivative") {
    val xs = List(-1.0, 0.0, 1.0, 2.0, 3.0)

    // differentiate 1 / (1 + e^(-x)) wrt x for x in -1, 0, 1, 2, 3
    val sigmoidDerivativesOfXs = List(0.196612, 0.25, 0.196612, 0.104994, 0.0451767)
    for((x, sigmoidDerivativeOfX) <- xs.zip(sigmoidDerivativesOfXs)) {
      assert(equal(RNN.sigmoidDerivative(x), sigmoidDerivativeOfX))
    }
  }

  test("fold tree") {
    val ls = List(-10.0, -2.0, -1.0, -0.8, -0.5, 0.0, 0.5, 0.8, 1.0, 2.0, 10.0) // a
    val rs = List(0.5, -1.0, 1.0, -0.8, 0.5, 1.0, 0.5, 0.0, 1.0, 2.0, 10.0) // b
    val combinator_1 = List(1.0, 10.0, 20.0, -11.0, 21.0, -22.0, 12.0, 0.1, 0.0, -1.0, 3.0) // c
    val combinator_2 = List(12.0, 1.0, -2.0, -11.0, -1.0, -22.0, 4.0, 0.1, 0.0, 0.0, 2.0) // d
    val combinator_3 = List(-3.0, 4.0, 2.0, -11.0, 10.0, 1.0, 3.0, -0.3, -1.0, 0.0, 3.0) // e

    val a = ls
    val b = rs
    val c = combinator_1
    val d = combinator_2
    val e = combinator_3

    // sigmoid({{c, d, e}} * {{sigmoid(a)}, {sigmoid(b)}, {1.0}}) =
    // sigmoid(c * sigmoid(a) + d * sigmoid(b) + e * 1.0)

    // der(sigmoid(c * sigmoid(a) + d * sigmoid(b) + e * 1.0)) =
    // (der(c * sigmoid(a)) + der(d * sigmoid(b)) + der(e * 1.0)) * der(sigmoid)(c * sigmoid(a) + d * sigmoid(b) + e * 1.0) =
    // ( der(c) * sigmoid(a) + c * der(sigmoid(a))
    // + der(d) * sigmoid(b) + d * der(sigmoid(b))
    // + der(e) * 1.0 ) * der(sigmoid)(c * sigmoid(a) + d * sigmoid(b) + e * 1.0)

    // der a = (c * der(sigmoid(a))) * der(sigmoid)(c * sigmoid(a) + d * sigmoid(b) + e * 1.0)
    // der b = (d * der(sigmoid(b))) * der(sigmoid)(c * sigmoid(a) + d * sigmoid(b) + e * 1.0)
    // der c = sigmoid(a) * der(sigmoid)(c * sigmoid(a) + d * sigmoid(b) + e * 1.0)
    // der d = sigmoid(b) * der(sigmoid)(c * sigmoid(a) + d * sigmoid(b) + e * 1.0)
    // der e = der(sigmoid)(c * sigmoid(a) + d * sigmoid(b) + e * 1.0)

    val a_der = for (i <- a.indices)
      yield c(i) * RNN.sigmoidDerivative(a(i)) * RNN.sigmoidDerivative(c(i) * RNN.sigmoid(a(i)) + d(i) * RNN.sigmoid(b(i)) + e(i) * 1.0)
    val b_der = for (i <- a.indices)
      yield d(i) * RNN.sigmoidDerivative(b(i)) * RNN.sigmoidDerivative(c(i) * RNN.sigmoid(a(i)) + d(i) * RNN.sigmoid(b(i)) + e(i) * 1.0)
    val c_der = for (i <- a.indices)
      yield RNN.sigmoid(a(i)) * RNN.sigmoidDerivative(c(i) * RNN.sigmoid(a(i)) + d(i) * RNN.sigmoid(b(i)) + e(i) * 1.0)
    val d_der = for (i <- a.indices)
      yield RNN.sigmoid(b(i)) * RNN.sigmoidDerivative(c(i) * RNN.sigmoid(a(i)) + d(i) * RNN.sigmoid(b(i)) + e(i) * 1.0)
    val e_der = for (i <- a.indices)
      yield RNN.sigmoidDerivative(c(i) * RNN.sigmoid(a(i)) + d(i) * RNN.sigmoid(b(i)) + e(i) * 1.0)

    for (i <- a.indices) {
      println(s"test fold $i")
      val rnn = RNN(1, 0, 0, 0)
      rnn.clearCache()
      rnn.combinator = DenseMatrix((combinator_1(i), combinator_2(i), combinator_3(i)))
      val l = Leaf("l")
      val r = Leaf("r")
      val t = Node(l, r)
      rnn.wordToVecMap.put("l", DenseVector(ls(i)))
      rnn.wordToVecMap.put("r", DenseVector(rs(i)))
      val fpt = rnn.forwardPropagateTree(t)
      rnn.backwardPropagateTree(fpt, DenseVector.ones(1))
      //println(s"${rnn.wordToVecDerivativeMap.get("l").get(0)} ${a_der(i)} ${rnn.wordToVecDerivativeMap.get("l").get(0) / a_der(i)}")
      //println(s"${rnn.wordToVecDerivativeMap.get("r").get(0)} ${b_der(i)} ${rnn.wordToVecDerivativeMap.get("r").get(0) / b_der(i)}")
      //println(s"${rnn.combinatorDerivative(0, 0)} ${c_der(i)} ${rnn.combinatorDerivative(0, 0) / c_der(i)}")
      //println(s"${rnn.combinatorDerivative(0, 1)} ${d_der(i)} ${rnn.combinatorDerivative(0, 1) / d_der(i)}")
      //println(s"${rnn.combinatorDerivative(0, 2)} ${e_der(i)} ${rnn.combinatorDerivative(0, 2) / e_der(i)}")
      assert(equal(rnn.wordToVecDerivativeMap.get("l").get(0), a_der(i)))
      assert(equal(rnn.wordToVecDerivativeMap.get("r").get(0), b_der(i)))
      assert(equal(rnn.combinatorDerivative(0, 0), c_der(i)))
      assert(equal(rnn.combinatorDerivative(0, 1), d_der(i)))
      assert(equal(rnn.combinatorDerivative(0, 2), e_der(i)))
    }
  }
}
