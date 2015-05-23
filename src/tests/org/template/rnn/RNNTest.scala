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
      //println(s"test fold $i")
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

  test("test judgement") {
    // -g * log(sigmoid(b * sigmoid(a) + c)) - (1 - g) * log(1 - sigmoid(b * sigmoid(a) + c))
    val ls = List(-10.0, -2.0, -1.0, -0.8, -0.5, 0.0, 0.5, 0.8, 1.0, 2.0, 10.0) // a
    val judge_1 = List(1.0, 10.0, 20.0, -11.0, 21.0, -22.0, 12.0, 0.1, 0.0, -1.0, 3.0) // b
    val judge_2 = List(12.0, 1.0, -2.0, -11.0, -1.0, -22.0, 4.0, 0.1, 0.0, 0.0, 2.0) // c
    val judge_3 = List(-3.0, 4.0, 2.0, -11.0, 10.0, 1.0, 3.0, -0.3, -1.0, 0.0, 3.0) // d
    val judge_4 = List(-0.0, 4.5, -3.0, -1.0, 2.0, 1.0, 0.5, -0.3, -7.0, -3.0, 2.0) // e
    val expected_1 = List(1, 1, 1, 0, 0, 0, 0.7, 0.3, 0.2, 0.5, 0.6)
    val expected_2 = List(0, 0, 0, 1, 1, 1, 0.3, 0.7, 0.8, 0.5, 0.6)

    val a = ls
    val b = judge_1
    val c = judge_2
    val d = judge_3
    val e = judge_4
    val g = expected_1
    val h = expected_2

    // derivative of -g * log(f(b * f(a) + c)) - (1 - g) * log(1 - f(b * f(a) + c)) -h * log(f(d * f(a) + e)) - (1 - h) * log(1 - f(d * f(a) + e)) wrt a
    // (b (1 - g) f'[a] f'[c + b f[a]])/(1 - f[c + b f[a]]) - (b g f'[a] f'[c + b f[a]])/f[c + b f[a]] + (d (1 - h) f'[a] f'[e + d f[a]])/(1 - f[e + d f[a]]) - (d h f'[a] f'[e + d f[a]])/f[e + d f[a]]

    // derivative of log(f(b * f(a) + c)) wrt a
    // (b f'[a] f'[c + b f[a]])/f[c + b f[a]]
    // derivative of log(1 - f(b * f(a) + c)) wrt a
    // (d)/(da)(log(1-f(b f(a)+c))) = (b f'(a) f'(b f(a)+c))/(f(b f(a)+c)-1)

    val der_a = for (i <- a.indices)
      yield (-g(i) * (b(i) * RNN.sigmoidDerivative(a(i)) * RNN.sigmoidDerivative(c(i) + b(i) * RNN.sigmoid(a(i)))) / RNN.sigmoid(c(i) + b(i) * RNN.sigmoid(a(i)))
             - (1.0 - g(i)) * (b(i) * RNN.sigmoidDerivative(a(i)) * RNN.sigmoidDerivative(b(i) * RNN.sigmoid(a(i)) + c(i))) / (RNN.sigmoid(b(i) * RNN.sigmoid(a(i)) + c(i)) - 1.0)
             - h(i) * (d(i) * RNN.sigmoidDerivative(a(i)) * RNN.sigmoidDerivative(e(i) + d(i) * RNN.sigmoid(a(i)))) / RNN.sigmoid(e(i) + d(i) * RNN.sigmoid(a(i)))
             - (1.0 - h(i)) * (d(i) * RNN.sigmoidDerivative(a(i)) * RNN.sigmoidDerivative(d(i) * RNN.sigmoid(a(i)) + e(i))) / (RNN.sigmoid(d(i) * RNN.sigmoid(a(i)) + e(i)) - 1.0))
    val der_b = for (i <- a.indices)
      yield (-g(i) * (RNN.sigmoid(a(i)) * RNN.sigmoidDerivative(b(i) * RNN.sigmoid(a(i)) + c(i))) / (RNN.sigmoid(b(i) * RNN.sigmoid(a(i)) + c(i)))
             - (1.0 - g(i)) * (RNN.sigmoid(a(i)) * RNN.sigmoidDerivative(b(i) * RNN.sigmoid(a(i)) + c(i))) / (RNN.sigmoid(b(i) * RNN.sigmoid(a(i)) + c(i)) - 1.0))
    val der_c = for (i <- a.indices)
      yield (-g(i) * (RNN.sigmoidDerivative(b(i) * RNN.sigmoid(a(i)) + c(i))) / (RNN.sigmoid(b(i) * RNN.sigmoid(a(i)) + c(i)))
             - (1.0 - g(i)) * (RNN.sigmoidDerivative(b(i) * RNN.sigmoid(a(i)) + c(i))) / (RNN.sigmoid(b(i) * RNN.sigmoid(a(i)) + c(i)) - 1))

    for(i <- a.indices) {
      val rnn = RNN(1, 2, 0, 0)
      rnn.clearCache()
      rnn.judge = DenseMatrix((judge_1(i), judge_2(i)), (judge_3(i), judge_4(i)))
      val l = Leaf("l")
      rnn.wordToVecMap.put("l", DenseVector(ls(i)))
      val fpt = rnn.forwardPropagateTree(l)
      rnn.backwardPropagateError(fpt, DenseVector(expected_1(i), expected_2(i)))
      // println(s"${rnn.wordToVecDerivativeMap.get("l").get(0)} ${der_a(i)} ${rnn.wordToVecDerivativeMap.get("l").get(0) / der_a(i)}")
      // println(s"${rnn.judgeDerivative(0, 0)} ${der_b(i)} ${rnn.judgeDerivative(0, 0) / der_b(i)}")
      // println(s"${rnn.judgeDerivative(0, 0)} ${der_c(i)} ${rnn.judgeDerivative(0, 0) / der_c(i)}")
      assert(equal(rnn.wordToVecDerivativeMap.get("l").get(0), der_a(i)))
      assert(equal(rnn.judgeDerivative(0, 0), der_b(i)))
      assert(equal(rnn.judgeDerivative(0, 1), der_c(i)))
    }
  }

  test("test fit A") {
    println("test fit A")

    val rnn = RNN(10, 3, 1, 0.001)
    val l = Leaf("word")
    val ps = Vector((l, 0))
    for(i <- 0 to 10)
    {
      rnn.fit(ps)
      println(rnn.forwardPropagateJudgment(rnn.forwardPropagateTree(l)))
      println(rnn.forwardPropagateError(ps))
    }
  }

  test("test fit B") {
    println("test fit B")

    val rnn = RNN(10, 3, 1, 0.001)
    val l = Leaf("word")
    val r = Leaf("other")
    val t2 = Node(Leaf("a"), Leaf("b"))
    val t1 = Node(t2, Leaf("a"))
    val ps = Vector((l, 0), (r, 1), (t1, 2))
    for(i <- 0 to 20)
    {
      rnn.fit(ps)
      println(s"l ${rnn.forwardPropagateJudgment(rnn.forwardPropagateTree(l))}")
      println(s"r ${rnn.forwardPropagateJudgment(rnn.forwardPropagateTree(r))}")
      println(s"t1 ${rnn.forwardPropagateJudgment(rnn.forwardPropagateTree(t1))}")
      println(rnn.forwardPropagateError(ps))
    }
  }

}
