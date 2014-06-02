package org.apache.spark.sql.execution.prober

import org.apache.spark.scheduler._
import org.apache.spark.Logging

import scala.collection.mutable


/** A high-level wrapper class of SparkListener that reports various statistics. */
class ProbingListener extends SparkListener with Logging {

  val proberResults: ProberResults = ProberResults(Map())

  // Chronological stage count (`stageCnt`)
  var recordAtTaskLevelForStage: Set[Int] = _

  var stageCnt: Int = 0
  var stageRuntime: Long = 0
  var stageCommunicationTime: Long = 0

  var taskTimes = new mutable.ArrayBuffer[Long]

  // TODO: look into TaskMetrics#executorRunTime?
  override def onTaskEnd(taskEnd: SparkListenerTaskEnd) {
    val taskCommunicationTime =
      taskEnd
        .taskMetrics
        .shuffleReadMetrics.map(_.fetchWaitTime).getOrElse(0L)
    val taskComputationTime = taskEnd.taskInfo.duration - taskCommunicationTime
    val taskRunTime = taskCommunicationTime + taskComputationTime

    // total duration
    stageRuntime += taskComputationTime
    // fetch wait time
    stageCommunicationTime += taskCommunicationTime

    // Record at task level, for convenience onStageCompleted() will record
    // aggregated stats at stage level again.
    if (recordAtTaskLevelForStage(stageCnt + 1)) {
      val stageId = "stage-" + (stageCnt + 1)
      val taskId = "task-" + taskEnd.taskInfo.taskId
      val id = stageId + "-" + taskId

      proberResults.record(id + "-computationTime",
        ProbingListener.millisToString(taskComputationTime))
      proberResults.record(id + "-communicationTime",
        ProbingListener.millisToString(taskCommunicationTime))
      proberResults.record(id + "-totalTime", ProbingListener.millisToString(taskRunTime))
      proberResults.record(id + "-computationTimeRaw", taskComputationTime.toString)
      proberResults.record(id + "-communicationTimeRaw", taskCommunicationTime.toString)
      proberResults.record(id + "-totalTimeRaw", taskRunTime.toString)
    }
  }

  override def onStageCompleted(stageCompleted: SparkListenerStageCompleted) {
    val stageInfo = stageCompleted.stageInfo

    logInfo("Finished stage: " + stageInfo.stageId)
    stageCnt += 1

    // Executor (non-fetch) time plus other time
    val totalComputationTime = stageRuntime - stageCommunicationTime

    // use chronological stage id to avoid inversion
    val stageId = "stage-" + stageCnt

    // Stage runtime breakdown
    logInfo("Total stage runtime: " + ProbingListener.millisToString(stageRuntime))
    logInfo("Total communication runtime: " + ProbingListener.millisToString(stageCommunicationTime))
    logInfo("Total computation runtime: " + ProbingListener.millisToString(totalComputationTime))

    // Other info
    logInfo("Number of partitions: " + stageInfo.rddInfos.head.numPartitions)
    logInfo("Number of tasks: " + stageInfo.numTasks)

    proberResults.record(stageId + "-computationTime", ProbingListener.millisToString(totalComputationTime))
    proberResults.record(stageId + "-communicationTime", ProbingListener.millisToString(stageCommunicationTime))
    proberResults.record(stageId + "-totalTime", ProbingListener.millisToString(stageRuntime))
    proberResults.record(stageId + "-computationTimeRaw", totalComputationTime.toString)
    proberResults.record(stageId + "-communicationTimeRaw", stageCommunicationTime.toString)
    proberResults.record(stageId + "-totalTimeRaw", stageRuntime.toString)
    proberResults.record(stageId + "-numPartitions", stageInfo.rddInfos.head.numPartitions.toString)
    proberResults.record(stageId + "-numTasks", stageInfo.numTasks.toString)

    stageRuntime = 0
    stageCommunicationTime = 0
  }

}

object ProbingListener {

  // The below vals and method are cargo-culted from SparkListener.scala

  val seconds = 1000L
  val minutes = seconds * 60
  val hours = minutes * 60

  /**
   * reformat a time interval in milliseconds to a prettier format for output
   */
  def millisToString(ms: Long) = {
    val (size, units) =
      if (ms > hours) {
        (ms.toDouble / hours, "hours")
      } else if (ms > minutes) {
        (ms.toDouble / minutes, "min")
      } else if (ms > seconds) {
        (ms.toDouble / seconds, "s")
      } else {
        (ms.toDouble, "ms")
      }
    "%.1f %s".format(size, units)
  }

}
