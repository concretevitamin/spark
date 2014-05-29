package org.apache.spark.sql.catalyst.expressions

import org.scalatest.FunSuite

import org.apache.spark.sql.catalyst.expressions.AnonymousRow._

case class Data(a: Int, b: Int)

class AnonymousRowSuite extends FunSuite {
  import org.apache.spark.sql.test.TestSQLContext._

  test("correct names") {
    assert(row('a -> 1).schema.map(_.name) === Seq("a"))
    assert(row('a -> 1, 'b -> 2).schema.map(_.name) === Seq("a", "b"))
  }

  test("old version") {
    // Had to declare Data above?
    sparkContext.parallelize(1 to 100).map(i => Data(i, i * 2)).registerAsTable("data")
    val result = sql("SELECT b, a FROM data").collect()
    result.head.getInt(0)
    Data(result.head.getInt(1), result.head.getInt(0))
  }

  test("with AnonymousRows") {
    //sparkContext.parallelize(1 to 100).map(i => row(a = i)).registerAsTable("data")
    sparkContext.parallelize(1 to 100).map(i => row('a -> i, 'b -> i * 2)).registerAsTable("data")
    val result = sql("SELECT b, a FROM data").collect()
    // result.head.a
    // result.head.as[Data]
  }
}