package org.template.rnn

abstract class Tree
case class Node(left: Tree, right: Tree) extends Tree
case class Leaf(word: String) extends Tree