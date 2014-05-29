package org.apache.spark.sql.catalyst.expressions

import scala.reflect.macros.Context
import scala.language.experimental.macros

import org.apache.spark.sql.catalyst.types._

object AnonymousRow {

  def row(values: Any*) = macro rowImpl


  // TODO: What about nested rows?
  def rowImpl(c: Context)
             (values: c.Expr[Any]*): c.Expr[AnonymousRow] = {
    import c.universe._

    // Maybe a little brittle?
    val attributeNames: Seq[String] = values.map { value =>
      value
        .tree
        .children
        .head
        .children
        .head
        .children
        .head
        .children
        .last
        .children
        .last
        .asInstanceOf[scala.reflect.internal.Trees$Literal]
        .value
        .value.asInstanceOf[String]
    }

    val rawValues = values.map { value =>
      value
        .tree
        .children
        .last
    }

    // TODO: Don't hard code IntType, would be nice to get from the Typer.
    val schema = attributeNames.map(a => q"org.apache.spark.sql.catalyst.expressions.AttributeReference($a, org.apache.spark.sql.catalyst.types.IntegerType, nullable = false)()")

    // TODO: Avoid creating schema per row
    // TODO: Avoid creating Array per row?
    val e = q"new org.apache.spark.sql.catalyst.expressions.AnonymousRow(Seq(..$schema), Array[Any](..$rawValues))"

    c.Expr[AnonymousRow](e)
  }
}

class AnonymousRow(val schema: Seq[Attribute], values: Array[Any]) extends GenericRow(values)
