/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.catalyst.optimizer

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.plans.logical._

/**
 * Overrides our expression evaluation tests and reruns them after optimization has occurred.  This
 * is to ensure that constant folding and other optimizations do not break anything.
 */
class GeneratedEvaluationSuite extends ExpressionEvaluationSuite {
  override def checkEvaluation(
                                expression: Expression,
                                expected: Any,
                                inputRow: Row = EmptyRow): Unit = {
    val plan = try {
      GenerateMutableProjection(Alias(expression, s"Optimized($expression)")() :: Nil)()
    } catch {
      case e: Throwable =>
        val evaluated = new CodeGenerator().expressionEvaluator(expression)
        fail(
          s"""
            |Code generation of $expression failed:
            |${evaluated.code.mkString("\n")}
            |$e
          """.stripMargin)
    }

    val actual  = plan(inputRow).apply(0)
    if(actual != expected) {
      val input = if(inputRow == EmptyRow) "" else s", input: $inputRow"
      fail(s"Incorrect Evaluation: $expression, actual: $actual, expected: $expected$input")
    }
  }
}

class GeneratedMutableEvaluationSuite extends ExpressionEvaluationSuite {
  override def checkEvaluation(
                                expression: Expression,
                                expected: Any,
                                inputRow: Row = EmptyRow): Unit = {
    lazy val evaluated = new CodeGenerator().expressionEvaluator(expression)

    val plan = try {
      GenerateProjection(Alias(expression, s"Optimized($expression)")() :: Nil)
    } catch {
      case e: Throwable =>
        fail(
          s"""
            |Code generation of $expression failed:
            |${evaluated.code.mkString("\n")}
            |$e
          """.stripMargin)
    }

    val actual = plan(inputRow)
    val expectedRow = new GenericRow(Array[Any](expected))
    if (actual.hashCode() != expectedRow.hashCode()) {
      fail(
        s"""
          |Mismatched hashCodes for values: $actual, $expectedRow
          |Hash Codes: ${actual.hashCode()} != ${expectedRow.hashCode()}
          |${evaluated.code.mkString("\n")}
        """.stripMargin)
    }
    if (actual != expectedRow) {
      val input = if(inputRow == EmptyRow) "" else s", input: $inputRow"
      fail(s"Incorrect Evaluation: $expression, actual: $actual, expected: $expected$input")
    }
  }
}