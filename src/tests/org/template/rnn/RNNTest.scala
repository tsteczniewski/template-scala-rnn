package org.template.rnn

import breeze.linalg.{DenseVector, DenseMatrix}

class RNNTest extends org.scalatest.FunSuite {

  val infinity = 100000000.0

  def equal(x: Double, y: Double): Boolean = {
    val eps = 0.000001
    (x - eps <= y && y <= x + eps) || (1.0 - eps <= x / y && x / y <= 1.0 + eps)
  }

  test("test sigmoid") {
    val xs = List(-1.0, 0.0, 1.0, 2.0, 3.0)

    // 1 / (1 + e^(-x)) for x in -1, 0, 1, 2, 3
    val sigmoidsOfXs = List(0.268941, 0.5, 0.731059, 0.880797, 0.952574)
    for((x, sigmoidOfX) <- xs.zip(sigmoidsOfXs)) {
      assert(equal(RNTN.sigmoid(x), sigmoidOfX))
    }
  }

  test("test sigmoid derivative") {
    val xs = List(-1.0, 0.0, 1.0, 2.0, 3.0)

    // differentiate 1 / (1 + e^(-x)) wrt x for x in -1, 0, 1, 2, 3
    val sigmoidDerivativesOfXs = List(0.196612, 0.25, 0.196612, 0.104994, 0.0451767)
    for((x, sigmoidDerivativeOfX) <- xs.zip(sigmoidDerivativesOfXs)) {
      assert(equal(RNTN.sigmoidDerivative(x), sigmoidDerivativeOfX))
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
      yield c(i) * RNTN.sigmoidDerivative(a(i)) * RNTN.sigmoidDerivative(c(i) * RNTN.sigmoid(a(i)) + d(i) * RNTN.sigmoid(b(i)) + e(i) * 1.0)
    val b_der = for (i <- a.indices)
      yield d(i) * RNTN.sigmoidDerivative(b(i)) * RNTN.sigmoidDerivative(c(i) * RNTN.sigmoid(a(i)) + d(i) * RNTN.sigmoid(b(i)) + e(i) * 1.0)
    val c_der = for (i <- a.indices)
      yield RNTN.sigmoid(a(i)) * RNTN.sigmoidDerivative(c(i) * RNTN.sigmoid(a(i)) + d(i) * RNTN.sigmoid(b(i)) + e(i) * 1.0)
    val d_der = for (i <- a.indices)
      yield RNTN.sigmoid(b(i)) * RNTN.sigmoidDerivative(c(i) * RNTN.sigmoid(a(i)) + d(i) * RNTN.sigmoid(b(i)) + e(i) * 1.0)
    val e_der = for (i <- a.indices)
      yield RNTN.sigmoidDerivative(c(i) * RNTN.sigmoid(a(i)) + d(i) * RNTN.sigmoid(b(i)) + e(i) * 1.0)

    for (i <- a.indices) {
      //println(s"test fold $i")
      val rnn = RNTN(1, 0, 0, 0)
      rnn.clearCache()
      rnn.labelToCombinatorMap.put(("LABEL", 2), DenseMatrix((combinator_1(i), combinator_2(i), combinator_3(i))))
      val l = Leaf("l", "LABEL")
      val r = Leaf("r", "LABEL")
      val t = Node(List(l, r), "LABEL")
      rnn.wordToVecMap.put(("l", "LABEL"), DenseVector(ls(i)))
      rnn.wordToVecMap.put(("r", "LABEL"), DenseVector(rs(i)))
      val fpt = rnn.forwardPropagateTree(t)
      rnn.backwardPropagateTree(fpt, DenseVector.ones(1))
      //println(s"${rnn.wordToVecDerivativeMap.get("l").get(0)} ${a_der(i)} ${rnn.wordToVecDerivativeMap.get("l").get(0) / a_der(i)}")
      //println(s"${rnn.wordToVecDerivativeMap.get("r").get(0)} ${b_der(i)} ${rnn.wordToVecDerivativeMap.get("r").get(0) / b_der(i)}")
      //println(s"${rnn.combinatorDerivative(0, 0)} ${c_der(i)} ${rnn.combinatorDerivative(0, 0) / c_der(i)}")
      //println(s"${rnn.combinatorDerivative(0, 1)} ${d_der(i)} ${rnn.combinatorDerivative(0, 1) / d_der(i)}")
      //println(s"${rnn.combinatorDerivative(0, 2)} ${e_der(i)} ${rnn.combinatorDerivative(0, 2) / e_der(i)}")
      assert(equal(rnn.wordToVecDerivativeMap.get(("l", "LABEL")).get(0), a_der(i)))
      assert(equal(rnn.wordToVecDerivativeMap.get(("r", "LABEL")).get(0), b_der(i)))
      assert(equal(rnn.labelToCombinatorDerivativeMap.get(("LABEL", 2)).get(0, 0), c_der(i)))
      assert(equal(rnn.labelToCombinatorDerivativeMap.get(("LABEL", 2)).get(0, 1), d_der(i)))
      assert(equal(rnn.labelToCombinatorDerivativeMap.get(("LABEL", 2)).get(0, 2), e_der(i)))
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
      yield (-g(i) * (b(i) * RNTN.sigmoidDerivative(a(i)) * RNTN.sigmoidDerivative(c(i) + b(i) * RNTN.sigmoid(a(i)))) / RNTN.sigmoid(c(i) + b(i) * RNTN.sigmoid(a(i)))
             - (1.0 - g(i)) * (b(i) * RNTN.sigmoidDerivative(a(i)) * RNTN.sigmoidDerivative(b(i) * RNTN.sigmoid(a(i)) + c(i))) / (RNTN.sigmoid(b(i) * RNTN.sigmoid(a(i)) + c(i)) - 1.0)
             - h(i) * (d(i) * RNTN.sigmoidDerivative(a(i)) * RNTN.sigmoidDerivative(e(i) + d(i) * RNTN.sigmoid(a(i)))) / RNTN.sigmoid(e(i) + d(i) * RNTN.sigmoid(a(i)))
             - (1.0 - h(i)) * (d(i) * RNTN.sigmoidDerivative(a(i)) * RNTN.sigmoidDerivative(d(i) * RNTN.sigmoid(a(i)) + e(i))) / (RNTN.sigmoid(d(i) * RNTN.sigmoid(a(i)) + e(i)) - 1.0))
    val der_b = for (i <- a.indices)
      yield (-g(i) * (RNTN.sigmoid(a(i)) * RNTN.sigmoidDerivative(b(i) * RNTN.sigmoid(a(i)) + c(i))) / (RNTN.sigmoid(b(i) * RNTN.sigmoid(a(i)) + c(i)))
             - (1.0 - g(i)) * (RNTN.sigmoid(a(i)) * RNTN.sigmoidDerivative(b(i) * RNTN.sigmoid(a(i)) + c(i))) / (RNTN.sigmoid(b(i) * RNTN.sigmoid(a(i)) + c(i)) - 1.0))
    val der_c = for (i <- a.indices)
      yield (-g(i) * (RNTN.sigmoidDerivative(b(i) * RNTN.sigmoid(a(i)) + c(i))) / (RNTN.sigmoid(b(i) * RNTN.sigmoid(a(i)) + c(i)))
             - (1.0 - g(i)) * (RNTN.sigmoidDerivative(b(i) * RNTN.sigmoid(a(i)) + c(i))) / (RNTN.sigmoid(b(i) * RNTN.sigmoid(a(i)) + c(i)) - 1))

    for(i <- a.indices) {
      val rnn = RNTN(1, 2, 0, 0)
      rnn.clearCache()
      rnn.judge = DenseMatrix((judge_1(i), judge_2(i)), (judge_3(i), judge_4(i)))
      val l = Leaf("l", "LABEL")
      rnn.wordToVecMap.put(("l", "LABEL"), DenseVector(ls(i)))
      val fpt = rnn.forwardPropagateTree(l)
      rnn.backwardPropagateError(fpt, DenseVector(expected_1(i), expected_2(i)))
      // println(s"${rnn.wordToVecDerivativeMap.get("l").get(0)} ${der_a(i)} ${rnn.wordToVecDerivativeMap.get("l").get(0) / der_a(i)}")
      // println(s"${rnn.judgeDerivative(0, 0)} ${der_b(i)} ${rnn.judgeDerivative(0, 0) / der_b(i)}")
      // println(s"${rnn.judgeDerivative(0, 0)} ${der_c(i)} ${rnn.judgeDerivative(0, 0) / der_c(i)}")
      assert(equal(rnn.wordToVecDerivativeMap.get(("l", "LABEL")).get(0), der_a(i)))
      assert(equal(rnn.judgeDerivative(0, 0), der_b(i)))
      assert(equal(rnn.judgeDerivative(0, 1), der_c(i)))
    }
  }

  test("test fit A") {
    //println("test fit A")

    val rnn = RNTN(10, 3, 1, 0.001)
    val l = Leaf("word", "LABEL")
    val ps = Vector((l, 0))
    var previousError = infinity
    for(i <- 0 to 10)
    {
      rnn.fit(ps)
      //println(rnn.forwardPropagateJudgment(rnn.forwardPropagateTree(l)))
      //println(rnn.forwardPropagateError(ps))
      val currentError = rnn.forwardPropagateError(ps)
      assert(currentError < previousError)
      previousError = currentError
    }
    assert(rnn.labelToCombinatorMap.size == 0)
    assert(rnn.labelToCombinatorDerivativeMap.size == 0)
    assert(rnn.wordToVecMap.size == 1)
    assert(rnn.wordToVecDerivativeMap.size == 1)
  }

  test("test fit B") {
    //println("test fit B")

    val rnn = RNTN(10, 3, 1, 0.001)
    val l = Leaf("word", "LABEL")
    val r = Leaf("other", "BABEL")
    val t2 = Node(List(Leaf("a", "GABEL"), Leaf("b", "REBEL")), "LABEL")
    val t1 = Node(List(t2, Leaf("a", "GABEL")), "LABEL")
    val ps = Vector((l, 0), (r, 1), (t1, 2))
    var previousError = infinity
    for(i <- 0 to 20)
    {
      rnn.fit(ps)
      //println(s"l ${rnn.forwardPropagateJudgment(rnn.forwardPropagateTree(l))}")
      //println(s"r ${rnn.forwardPropagateJudgment(rnn.forwardPropagateTree(r))}")
      //println(s"t1 ${rnn.forwardPropagateJudgment(rnn.forwardPropagateTree(t1))}")
      //println(rnn.forwardPropagateError(ps))
      val currentError = rnn.forwardPropagateError(ps)
      assert(currentError < previousError)
      previousError = currentError
    }
    assert(rnn.labelToCombinatorMap.contains(("LABEL", 2)))
    assert(rnn.labelToCombinatorDerivativeMap.contains(("LABEL", 2)))
    assert(rnn.labelToCombinatorMap.size == 1)
    assert(rnn.labelToCombinatorDerivativeMap.size == 1)
  }

  test("test fit C") {
    //println("test fit C")

    val rnn = RNTN(10, 3, 1, 0.001)
    val l = Leaf("word", "LABEL")
    val r = Leaf("other", "BABEL")
    val t2 = Node(List(Leaf("a", "GABEL"), Leaf("b", "REBEL")), "LABEL")
    val t1 = Node(List(t2, Leaf("a", "GABEL")), "BABEL")
    val t3 = Node(List(Leaf("a", "STH"), Leaf("b", "STH"), Leaf("c", "STH")), "LABEL")
    val ps = Vector((l, 0), (r, 1), (t1, 2), (t3, 0))
    var previousError = infinity
    for(i <- 0 to 20)
    {
      rnn.fit(ps)
      //println(s"l ${rnn.forwardPropagateJudgment(rnn.forwardPropagateTree(l))}")
      //println(s"r ${rnn.forwardPropagateJudgment(rnn.forwardPropagateTree(r))}")
      //println(s"t1 ${rnn.forwardPropagateJudgment(rnn.forwardPropagateTree(t1))}")
      //println(rnn.forwardPropagateError(ps))
      val currentError = rnn.forwardPropagateError(ps)
      assert(currentError < previousError)
      previousError = currentError
    }
    assert(rnn.labelToCombinatorMap.contains(("LABEL", 2)))
    assert(rnn.labelToCombinatorMap.contains(("LABEL", 3)))
    assert(rnn.labelToCombinatorMap.contains(("BABEL", 2)))
    assert(rnn.labelToCombinatorMap.size == 3)
    assert(rnn.labelToCombinatorDerivativeMap.size == 3)
  }
}
