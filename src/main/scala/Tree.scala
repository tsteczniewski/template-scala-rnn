package org.template.rnn

import scala.collection.mutable.Stack

abstract class Tree(val label: String)

object Tree {
  def unapply(t: Tree) = Some((t.label))

  def fromPennTreeBankFormat(string: String): Tree = {
    abstract class State
    case class OpenedUnlabeled(label: String) extends State
    case class OpenedLabeled(label: String) extends State
    case class OpenedLabeledNode(label: String) extends State
    case class OpenedLabeledLeaf(label: String, word: String) extends State
    case class Closed(tree: Tree) extends State

    val stack = Stack[State]()

    for(char <- string) {
      char match {
        case '(' =>
          if(stack.nonEmpty && stack.top.isInstanceOf[OpenedLabeled]) {
            val OpenedLabeled(label) = stack.top
            stack.pop
            stack.push(OpenedLabeledNode(label))
          }
          stack.push(OpenedUnlabeled(""))
        case ')' =>
          if(stack.top.isInstanceOf[OpenedLabeledLeaf]) {
            val OpenedLabeledLeaf(label, word) = stack.top
            stack.pop
            stack.push(Closed(Leaf(word, label)))
          } else {
            var children = List[Tree]()
            while (stack.top.isInstanceOf[Closed]) {
              val Closed(tree) = stack.top
              stack.pop
              children = tree +: children
            }
            val OpenedLabeledNode(label) = stack.top
            stack.pop
            stack.push(Closed(Node(children, label)))
          }
        case ' ' =>
          if (stack.top.isInstanceOf[OpenedUnlabeled]) {
            val OpenedUnlabeled(label) = stack.top
            stack.pop
            stack.push(OpenedLabeled(label))
          }
        case x =>
          stack.top match {
            case OpenedUnlabeled(label) =>
              stack.pop
              stack.push(OpenedUnlabeled(label + x))
            case OpenedLabeled(label) =>
              stack.pop
              stack.push(OpenedLabeledLeaf(label, x.toString))
            case OpenedLabeledLeaf(label, word) =>
              stack.pop
              stack.push(OpenedLabeledLeaf(label, word + x))
          }
      }
    }

    val Closed(tree) = stack.top
    tree
  }
}


case class Node(children: List[Tree], override val label: String) extends Tree(label)
case class Leaf(word: String, override val label: String) extends Tree(label)