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
import java.nio.file.{Files, Path, Paths}
import java.util

import org.apache.spark.internal.Logging
import org.tensorflow.{Graph, Session, Tensors}

class TensorFlowModel(modelDir: String) extends Logging {

  logInfo(s"TensorFlowModel, loading from $modelDir")

  // Load the model.
  private val graphPath = modelDir + "frozen_graph.pb"
  private val graphDef = readAllBytesOrExit(Paths.get(graphPath))
  private val g = new Graph
  g.importGraphDef(graphDef)
  private val sess = new Session(g)

  /** Assumes normalization done in training is taking ln() on the two cardinality columns. */
  def transformFeatures(xs: Array[Array[Double]]): Array[Array[Float]] = {
//    logInfo(s"xs ${xs.map(_.mkString(",")).mkString("\n")}")
    val ret = xs.map { arr =>
      arr(21) = Math.log(arr(21)).toFloat
      arr(43) = Math.log(arr(43)).toFloat
      arr.map(_.toFloat)
    }
//    logInfo(s"ret ${ret.map(_.mkString(",")).mkString("\n")}")
    ret
  }

  def transform(xs: Array[Array[Float]]): Array[Array[Float]] = {
    xs.foreach { arr =>
      arr(21) = Math.log(arr(21)).toFloat
      arr(43) = Math.log(arr(43)).toFloat
    }
    xs
  }

  def run(featVec: Seq[Float]): Float = {
//    logInfo("In TensorFlowModel: run()")

    //    val featVec = Array(
    //      Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1380035.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1337140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    //      Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1467823.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1337140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    //      Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1467823.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1337140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    //      Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1380035.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2528312.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    //    )
    val feats = Array(featVec.toArray)
    val floatTensor = Tensors.create(transform(feats))
    val predicted = Array.ofDim[Float](feats.length, 1)
//    println("input shape: " + util.Arrays.toString(floatTensor.shape()))

    val output = sess.runner
      .feed("IteratorGetNext", 0, floatTensor)
      .fetch("out_denormalized/Exp", 0)
      .run
      .get(0)

//    System.out.println("output shape: " + util.Arrays.toString(output.shape))
    output.copyTo(predicted)

//    println("input: ")
//    feats.foreach(v => println(util.Arrays.toString(v)))
//    println("predicted: ")

    // Close the Tensor to avoid resource leaks.
    output.close()
    // Output is of shape [1,1] which prevents us from calling output.floatValue().
    predicted(0)(0)
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
