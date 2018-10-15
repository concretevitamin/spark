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

// scalastyle:off
package org.apache.spark.sql.catalyst.optimizer

import java.io.IOException
import java.nio.FloatBuffer
import java.nio.file.{Files, Path, Paths}

import org.apache.spark.internal.Logging
import org.tensorflow.{Graph, Session, Tensor, Tensors}

class TensorFlowModel(modelDir: String) extends Logging {

  // TODO: it'd be good to move the transform in-graph.
  // val CARDINALITY_COLUMNS = Seq(42, 85)  // JOB
  val CARDINALITY_COLUMNS = Seq(72, 145) // TPC-DS

  logWarning(s"TensorFlowModel, loading from $modelDir, card columns $CARDINALITY_COLUMNS")

  // Load the model.
  private val graphDef = readAllBytesOrExit(Paths.get(modelDir, "frozen_graph.pb"))

  private val g = new Graph
  private val sess = new Session(g)
  g.importGraphDef(graphDef)
  private val EPSILON: Float = 1e-8f

  logWarning(s"TensorFlowModel: loaded")

  def run(featureBatch: Array[Array[Float]]): Array[Float] = {
    // NOTE: https://github.com/tensorflow/tensorflow/issues/8244 indicates this might be
    // a very slow call; may want to look into alternatives.
    val floatTensor = Tensors.create(transform(featureBatch))
    val output = sess.runner
      .feed("IteratorGetNext", 0, floatTensor)
      .fetch("out_denormalized/Exp", 0)
      .run
      .get(0)

    val predicted = FloatBuffer.allocate(featureBatch.length)
    output.writeTo(predicted)
//    val predicted = Array.ofDim[Float](featureBatch.length, 1)
//    output.copyTo(predicted)
    // Close the Tensor to avoid resource leaks.
    output.close()
    floatTensor.close()

    // For
    //   adam_128x3_lr1e-4_bs512_l2_disam_allData_noRootCost
    //   adam_128x3_lr1e-3_bs512_disam_allData_noRootCost
    // Forgot to subtract epsilon in graph.
    var arr = predicted.array()
    for (i <- 0 until arr.length) {
      arr(i) -= EPSILON
    }
    arr
  }

  /** Assumes normalization done in training is taking ln() on the two cardinality columns. */
  def transform(xs: Array[Array[Float]]): Array[Array[Float]] = {
    xs.foreach { arr =>
      CARDINALITY_COLUMNS.foreach { col => arr(col) = Math.log(arr(col)).toFloat }
    }
    xs
  }

  private def readAllBytesOrExit(path: Path): Array[Byte] = {
    try return Files.readAllBytes(path)
    catch {
      case e: IOException =>
        e.printStackTrace()
        System.exit(1)
    }
    null
  }
}

// scalastyle:on
