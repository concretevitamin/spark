package org.apache.spark.sql.execution.prober

import java.io.File
import org.apache.spark.SparkContext
import scala.collection.SortedMap
import scala.collection.mutable

trait SparkProber {

  val listener = new ProbingListener
  private val recordAtTaskLevelForStage: mutable.HashSet[Int] = mutable.HashSet()

  def beforeProberJob(sc: SparkContext) = {
    listener.stageCnt = 0
    listener.stageCommunicationTime = 0
    listener.stageRuntime = 0
    sc.addSparkListener(listener)
  }

  def proberResults() = listener.proberResults

  def recordAtTaskLevelForStage(stage: Int): Unit = {
    recordAtTaskLevelForStage.add(stage)
    listener.recordAtTaskLevelForStage = recordAtTaskLevelForStage.toSet
  }

}

class ProberResults(var res: Map[String, String]) {

  def print(headerMsg: String = "", sortByKey: Boolean = true) = {
    if (headerMsg.size > 0) println(headerMsg)
    var toPrint = res
    if (sortByKey)
      toPrint = SortedMap(res.toSeq: _*).toMap
    toPrint.foreach {
      case (k, v) => println(k + ": " + v)
    }
    this
  }

  def printToFile(filePath: String, headerMsg: String = "", sortByKey: Boolean = true) = {

    def helper(f: java.io.File)(op: java.io.PrintWriter => Unit) {
      val p = new java.io.PrintWriter(f)
      try {
        op(p)
      } finally {
        p.flush()
        p.close()
      }
    }

    helper(new File(filePath)) { p =>
      p.println(headerMsg)
      var toPrint = res
      if (sortByKey)
        toPrint = SortedMap(res.toSeq: _*).toMap
      toPrint.foreach {
        case (k, v) => p.println(k + ": " + v)
      }
    }
    this
  }

  def record(k: String, v: String): ProberResults = {
    res += (k -> v)
    this
  }

  def timeStats(): ProberResults = {
    ProberResults(res.filterKeys(_.toLowerCase.matches(".*time")))
  }

}

object ProberResults {

  def apply(res: Map[String, String]): ProberResults = new ProberResults(res)

}