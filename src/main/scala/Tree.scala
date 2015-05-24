package org.template.rnn

abstract class Tree(val label: String)

object Tree {
  def unapply(t: Tree) = Some((t.label))
}

case class Node(children: List[Tree], override val label: String) extends Tree(label)
case class Leaf(word: String, override val label: String) extends Tree(label)