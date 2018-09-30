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

import java.io.{BufferedWriter, File, FileWriter}

import scala.collection.mutable
import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.catalyst.catalog.SessionCatalog
import org.apache.spark.sql.catalyst.expressions.{And, Attribute, AttributeSet, Expression, PredicateHelper}
import org.apache.spark.sql.catalyst.optimizer.JoinReorderDP.JoinPlan
import org.apache.spark.sql.catalyst.plans.{Inner, InnerLike, JoinType}
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.internal.SQLConf


object LearningOptimizer extends Logging {

  lazy val tfModel = new TensorFlowModel(SQLConf.get.joinReorderNeuralNetPath)

  def infer(featVec: Seq[Float]): Float = {
    tfModel.run(featVec)
  }

  // For >= 2-relation sets, root can either be Project or a Join.
  // Match on logical Join operators so we have correct left/right sides to work with.
  private def findJoinSides(p: LogicalPlan): Option[(LogicalPlan, LogicalPlan)] = {
    p match {
      case p: Project => findJoinSides(p.child)
      case j: Join =>
        assert(j.children.size == 2)
        Some((j.children.head, j.children(1)))
      case _ => None
    }
  }

  def relNamesToOneHot(relNames: Seq[String],
                       allTableNamesSorted: Seq[String]): Seq[Float] = {
    allTableNamesSorted.map { name => if (relNames.contains(name)) 1f else 0f }
  }

  /** Feature vector of a LogicalPlan (the label is calculated elsewhere):
    * Left-side relations (1-hot)
    * Left-side est. card.
    * Right-side relations (1-hot)
    * Right-side est. card.
    * Trajectory's final plan's all relations (1-hot)
    * */
  def featurize(plan: LogicalPlan,
                rootRelsOneHot: Seq[Float],
                conf: SQLConf,
                allTableNamesSorted: Seq[String]): Seq[Float] = {
    val result = findJoinSides(plan)

    if (result.isDefined) {
      val (left, right) = result.get
      val leftVisibleRels = left.collectLeaves().flatMap(_.baseTableName)
      val rightVisibleRels = right.collectLeaves().flatMap(_.baseTableName)

      // Which relations are present?
      val leftOneHot = relNamesToOneHot(leftVisibleRels, allTableNamesSorted)
      val rightOneHot = relNamesToOneHot(rightVisibleRels, allTableNamesSorted)

      // Each side's estimated cardinality?
      val leftEstCard = left.stats.rowCount.get.toFloat
      val rightEstCard = right.stats.rowCount.get.toFloat

      val featureVector = (leftOneHot :+ leftEstCard) ++
        (rightOneHot :+ rightEstCard) ++
        rootRelsOneHot

      // For debugging.
      logInfo(s"features $featureVector")
      logInfo(s"left Rels $leftVisibleRels OneHot $leftOneHot card $leftEstCard")
      logInfo(s"right Rels $rightVisibleRels OneHot $rightOneHot card $rightEstCard")
      logInfo(s"root rels $rootRelsOneHot")

      featureVector
    } else {
      // This can be reached for, say, a leaf Project-Filter-Relation block -- a singleton relation.
      // The NN does not need to worry about singleton base relations.
      Seq.empty
    }
  }

  /** Assumes "plan" is an optimal plan from some DP table.  Recursively collect training data. */
  def trainingData(plan: JoinPlan,
                   conf: SQLConf,
                   allTableNamesSorted: Seq[String]): Seq[Seq[Float]] = {
    // "plan" represents a terminal state, so use the same Q-val for all subplans.
    val qVal = (plan.rootCost(conf) + plan.planCost).combine()

    // The optimality of subplans is defined w.r.t. to joining these "root" rels.
    // In other words, some subplan here may no longer be optimal if the goal is to join some other
    // set of relations as the final goal.
    val rootRels = plan.plan.collectLeaves().flatMap(_.baseTableName)
    val rootRelsOneHot = relNamesToOneHot(rootRels, allTableNamesSorted)

    val trainingData = mutable.Buffer.empty[Seq[Float]]
    trainingDataHelper(plan.plan, trainingData, qVal, rootRelsOneHot, conf, allTableNamesSorted)
//    logInfo(s"filled trainingData ${trainingData.mkString("\n")}")
    trainingData
  }

  def trainingDataHelper(plan: LogicalPlan,
                         trainingData: mutable.Buffer[Seq[Float]],
                         qVal: Float,
                         rootRelsOneHot: Seq[Float],
                         conf: SQLConf,
                         allTableNamesSorted: Seq[String]): Unit = {
    val featVec = featurize(plan, rootRelsOneHot, conf, allTableNamesSorted)
    if (featVec.nonEmpty) {
      trainingData.append(featVec :+ qVal)
      logInfo(s"Emitting training data for plan $plan")
      logInfo(s"   ${trainingData.last}")
    }
    // All subplans underneath ME get the same Q-val.
    plan.children.foreach(
      trainingDataHelper(_, trainingData, qVal, rootRelsOneHot, conf, allTableNamesSorted))
  }

}


/**
 * Cost-based join reorder.
 * We may have several join reorder algorithms in the future. This class is the entry of these
 * algorithms, and chooses which one to use.
 */
case class CostBasedJoinReorder(sessionCatalog: SessionCatalog)
  extends Rule[LogicalPlan] with PredicateHelper {

  JoinReorderDP.sessionCatalog = sessionCatalog

  private def conf = SQLConf.get

  def apply(plan: LogicalPlan): LogicalPlan = {
    if (!conf.cboEnabled || !conf.joinReorderEnabled) {
      plan
    } else {
      val result = plan transformDown {
        // Start reordering with a joinable item, which is an InnerLike join with conditions.
        case j @ Join(_, _, _: InnerLike, Some(cond)) =>
          reorder(j, j.output)
        case p @ Project(projectList, Join(_, _, _: InnerLike, Some(cond)))
          if projectList.forall(_.isInstanceOf[Attribute]) =>
          reorder(p, p.output)
      }
      // After reordering is finished, convert OrderedJoin back to Join
      result transformDown {
        case OrderedJoin(left, right, jt, cond) => Join(left, right, jt, cond)
      }
    }
  }

  private def reorder(plan: LogicalPlan, output: Seq[Attribute]): LogicalPlan = {
    val (items, conditions) = extractInnerJoins(plan)

    logInfo(s"To search plan $plan")
    logInfo(s"Items $items Conditions $conditions")
    // Items is a list of Project-Filter-Relation blocks -- the base relations (LogicalPlan).
    // Conditions looks like: Set((id#32 = movie_id#11), (id#32 = movie_id#26), ...)

    val result =
      // Do reordering if the number of items is appropriate and join conditions exist.
      // We also need to check if costs of all items can be evaluated.
      if (items.size > 2 && items.size <= conf.joinReorderDPThreshold && conditions.nonEmpty &&
          items.forall(_.stats.rowCount.isDefined)) {
        JoinReorderDP.search(conf, items, conditions, output)
      } else {
        plan
      }
    // Set consecutive join nodes ordered.
    replaceWithOrderedJoin(result)
  }

  /**
   * Extracts items of consecutive inner joins and join conditions.
   * This method works for bushy trees and left/right deep trees.
   */
  private def extractInnerJoins(plan: LogicalPlan): (Seq[LogicalPlan], Set[Expression]) = {
    plan match {
      case Join(left, right, _: InnerLike, Some(cond)) =>
        val (leftPlans, leftConditions) = extractInnerJoins(left)
        val (rightPlans, rightConditions) = extractInnerJoins(right)
        (leftPlans ++ rightPlans, splitConjunctivePredicates(cond).toSet ++
          leftConditions ++ rightConditions)
      case Project(projectList, j @ Join(_, _, _: InnerLike, Some(cond)))
        if projectList.forall(_.isInstanceOf[Attribute]) =>
        extractInnerJoins(j)
      case _ =>
        (Seq(plan), Set())
    }
  }

  private def replaceWithOrderedJoin(plan: LogicalPlan): LogicalPlan = plan match {
    case j @ Join(left, right, jt: InnerLike, Some(cond)) =>
      val replacedLeft = replaceWithOrderedJoin(left)
      val replacedRight = replaceWithOrderedJoin(right)
      OrderedJoin(replacedLeft, replacedRight, jt, Some(cond))
    case p @ Project(projectList, j @ Join(_, _, _: InnerLike, Some(cond))) =>
      p.copy(child = replaceWithOrderedJoin(j))
    case _ =>
      plan
  }
}

/** This is a mimic class for a join node that has been ordered. */
case class OrderedJoin(
    left: LogicalPlan,
    right: LogicalPlan,
    joinType: JoinType,
    condition: Option[Expression]) extends BinaryNode {
  override def output: Seq[Attribute] = left.output ++ right.output
}

/**
 * Reorder the joins using a dynamic programming algorithm. This implementation is based on the
 * paper: Access Path Selection in a Relational Database Management System.
 * http://www.inf.ed.ac.uk/teaching/courses/adbs/AccessPath.pdf
 *
 * First we put all items (basic joined nodes) into level 0, then we build all two-way joins
 * at level 1 from plans at level 0 (single items), then build all 3-way joins from plans
 * at previous levels (two-way joins and single items), then 4-way joins ... etc, until we
 * build all n-way joins and pick the best plan among them.
 *
 * When building m-way joins, we only keep the best plan (with the lowest cost) for the same set
 * of m items. E.g., for 3-way joins, we keep only the best plan for items {A, B, C} among
 * plans (A J B) J C, (A J C) J B and (B J C) J A.
 * We also prune cartesian product candidates when building a new plan if there exists no join
 * condition involving references from both left and right. This pruning strategy significantly
 * reduces the search space.
 * E.g., given A J B J C J D with join conditions A.k1 = B.k1 and B.k2 = C.k2 and C.k3 = D.k3,
 * plans maintained for each level are as follows:
 * level 0: p({A}), p({B}), p({C}), p({D})
 * level 1: p({A, B}), p({B, C}), p({C, D})
 * level 2: p({A, B, C}), p({B, C, D})
 * level 3: p({A, B, C, D})
 * where p({A, B, C, D}) is the final output plan.
 *
 * For cost evaluation, since physical costs for operators are not available currently, we use
 * cardinalities and sizes to compute costs.
 */
object JoinReorderDP extends PredicateHelper with Logging {

  var sessionCatalog: SessionCatalog = _

  def search(
      conf: SQLConf,
      items: Seq[LogicalPlan],
      conditions: Set[Expression],
      output: Seq[Attribute]): LogicalPlan = {

    val startTime = System.nanoTime()
    // Level i maintains all found plans for i + 1 items.
    // Create the initial plans: each plan is a single item with zero cost.
    val itemIndex = items.zipWithIndex
    // Size == # relations.  Element at index i is the DP table for (i+1)-way join.
    val foundPlans = mutable.Buffer[JoinPlanMap](itemIndex.map {
      case (item, id) => Set(id) -> JoinPlan(Set(id), item, Set.empty, Cost(0, 0))
    }.toMap)

    // Build filters from the join graph to be used by the search algorithm.
    val filters = JoinReorderDPFilters.buildJoinGraphInfo(conf, items, conditions, itemIndex)

    // Build plans for next levels until the last level has only one plan. This plan contains
    // all items that can be joined, so there's no need to continue.
    val topOutputSet = AttributeSet(output)
    var foundPlan: LogicalPlan = null

    if (conf.joinReorderNeuralNetPath.isEmpty) {
      while (foundPlans.size < items.length) {
        // Build plans for the next level.
        foundPlans += searchLevel(foundPlans, conf, conditions, topOutputSet, filters)
      }
      logInfo(s"number of plans in memo: ${foundPlans.map(_.size).sum}")
      assert(foundPlans.size == items.length && foundPlans.last.size == 1)
      foundPlan = foundPlans.last.head._2.plan
    } else {
      def askNeuralNetForPlan(items: mutable.Buffer[JoinPlan]): LogicalPlan = {
        val rootRels = items.flatMap(_.plan.collectLeaves().map(_.baseTableName.get))
        val rootRelsOneHot =
          LearningOptimizer.relNamesToOneHot(rootRels, allTableNamesSorted)
        logInfo(s"Using NN for planning - rootRels $rootRels")

        while (items.length > 1) {
          var bestScore = Float.MaxValue
          var bestLeft: JoinPlan = null
          var bestRight: JoinPlan = null
          var newJoin: Option[JoinPlan] = None

          for (i <- items.indices) {
            for (j <- items.indices) {
              buildJoin(items(i), items(j), conf, conditions, topOutputSet, filters) match {
                case join@Some(newJoinPlan) =>
                  val featVec = LearningOptimizer.featurize(
                    newJoinPlan.plan, rootRelsOneHot, conf, allTableNamesSorted)
                  val predictedCost = LearningOptimizer.infer(featVec)

                  if (predictedCost < bestScore) {
                    newJoin = join
                    bestScore = predictedCost
                    bestLeft = items(i)
                    bestRight = items(j)
                  }

                case None =>
              }

            }
          }

          assert(bestLeft != null && bestRight != null)
          items -= bestLeft
          items -= bestRight
          items.append(newJoin.get)
        }
        items.head.plan
      }

      foundPlan = askNeuralNetForPlan(mutable.Buffer(itemIndex.map { case (item, id) =>
        JoinPlan(Set(id), item, Set.empty, Cost(0, 0))
      }: _*))
    }

    val durationInMs = (System.nanoTime() - startTime) / (1000 * 1000)
    logInfo(s"Join reordering finished. Duration: $durationInMs ms, number of items: " +
      s"${items.length}")

    // The last level must have one and only one plan, because all items are joinable.
    val retval = foundPlan match {
      case p @ Project(projectList, j: Join) if projectList != output =>
        assert(topOutputSet == p.outputSet)
        // Keep the same order of final output attributes.
        p.copy(projectList = output)
      case finalPlan =>
        finalPlan
    }

//    dumpLearningData(foundPlans)

    logInfo(s"input rels to join:\n${items.mkString("\n")}")
    logInfo(s"conditions:\n${conditions.mkString("\n")}")
    logInfo(s"output attrs:\n${output.mkString("\n")}")
    logInfo(s"final plan:\n${retval}")
    logInfo(s"final plan stats: ${retval.stats}")
//    logInfo(s"all found plans: $foundPlans")
    retval
  }

  /** Dumps appropriately calculated/featurized training data _from one particular query_. */
  private def dumpLearningData(maps: mutable.Buffer[JoinReorderDP.JoinPlanMap]): Unit = {
    logInfo(s"maps.size ${maps.size}")

//    logInfo(s"AnalysisContext map ${AnalysisContext.getAttributeMap}")
    val queryGraphRels = maps.last.head._2.plan.collectLeaves().flatMap(_.baseTableName)

    val trainingDataFile = new File("job-sparksql-training.csv")
    val bw = new BufferedWriter(new FileWriter(trainingDataFile, true))
    val indexFile = new File("job-sparksql-index.csv")  // How many points from each query?
    val bw2 = new BufferedWriter(new FileWriter(indexFile, true))
    var numDataPoints = 0

    maps.foreach { dpTable =>

      dpTable.foreach { item =>
        val plan = item._2

        // TODO: maybe look at extractInnerJoins()
        // For >= 2-relation sets, root can either be Project or a Join.
        // Match on logical Join operators so we have correct left/right sides to work with.
        def findJoinSides(p: LogicalPlan): Option[(LogicalPlan, LogicalPlan)] = {
          p match {
            case p: Project => findJoinSides(p.child)
            case j: Join =>
              assert(j.children.size == 2)
              Some((j.children.head, j.children(1)))
            case _ => None
          }
        }

        val result = findJoinSides(plan.plan)
        if (result.isDefined) {
          val (left, right) = result.get
          val leftVisibleRels = left.collectLeaves().flatMap(_.baseTableName)
          val rightVisibleRels = right.collectLeaves().flatMap(_.baseTableName)

          // TODO(zongheng): we can incorporate sizeInBytes as well.
          val estimatedCard = plan.planCost.card.floatValue()
          val myCost = plan.planCost.combine()
          val leftEstCard = left.stats.rowCount.get.toFloat
          val rightEstCard = right.stats.rowCount.get.toFloat

          val feats =
            s"""$leftVisibleRels $leftEstCard $rightVisibleRels $rightEstCard $queryGraphRels $estimatedCard $myCost (my planCost=${plan.planCost}; my rootCost=${plan.rootCost(SQLConf.get)})"""

          val data = LearningOptimizer.trainingData(plan, SQLConf.get, allTableNamesSorted)
          data.foreach(point => bw.write(point.mkString("", ",", "\n")))
          numDataPoints += data.length

          logInfo(s"features $feats")

//          val leftVisibleAttrs = denormalizeAttributes(getVisibleAttributes(left))
//          val rightVisibleAttrs = denormalizeAttributes(getVisibleAttributes(right))
//          logInfo(s"left attrSet ${getVisibleAttributes(left)}")
//          logInfo(s"right attrSet ${getVisibleAttributes(right)}")
//          logInfo(s"left attrs $leftVisibleAttrs right attrs $rightVisibleAttrs")
//          logInfo(s"  all attrs ${AnalysisContext.allAttributesSorted}")
        }


        logInfo(s"plan:\n${plan.plan}\n  attrs:${getVisibleAttributes(plan.plan)}\n")
      }

    }

    bw.close()
    logInfo(s"num data points $numDataPoints")
    bw2.write(numDataPoints.toString)
    bw2.write("\n")
    bw2.close()

    logInfo(s"queryGraphRels $queryGraphRels")
    logInfo(s"allTables ${allTableNamesSorted}")
//    maps.foreach { ()}
//    ()
  }

  //  val file = new File(canonicalFilename)
  //  val bw = new BufferedWriter(new FileWriter(file))
  //  bw.write(text)
  //  bw.close()



  /** Find all possible plans at the next level, based on existing levels. */
  private def searchLevel(
      existingLevels: Seq[JoinPlanMap],
      conf: SQLConf,
      conditions: Set[Expression],
      topOutput: AttributeSet,
      filters: Option[JoinGraphInfo]): JoinPlanMap = {

    val nextLevel = mutable.Map.empty[Set[Int], JoinPlan]
    var k = 0
    val lev = existingLevels.length - 1
    // Build plans for the next level from plans at level k (one side of the join) and level
    // lev - k (the other side of the join).
    // For the lower level k, we only need to search from 0 to lev - k, because when building
    // a join from A and B, both A J B and B J A are handled.
    while (k <= lev - k) {
      val oneSideCandidates = existingLevels(k).values.toSeq
      for (i <- oneSideCandidates.indices) {
        val oneSidePlan = oneSideCandidates(i)
        val otherSideCandidates = if (k == lev - k) {
          // Both sides of a join are at the same level, no need to repeat for previous ones.
          // I.e., no self-joins.
          oneSideCandidates.drop(i)
        } else {
          existingLevels(lev - k).values.toSeq
        }

        otherSideCandidates.foreach { otherSidePlan =>
          buildJoin(oneSidePlan, otherSidePlan, conf, conditions, topOutput, filters) match {
            case Some(newJoinPlan) =>
              // Check if it's the first plan for the item set, or it's a better plan than
              // the existing one due to lower cost.
              val existingPlan = nextLevel.get(newJoinPlan.itemIds)
              if (existingPlan.isEmpty || newJoinPlan.betterThan(existingPlan.get, conf)) {
//                if (existingPlan.isDefined && newJoinPlan.betterThan(existingPlan.get, conf)) {
//                  // A better plan.  Let's log.
//                  logInfo(s"existingPlan ${existingPlan.get.plan} planCost ${existingPlan.get.planCost} rootCost ${existingPlan.get.rootCost(conf)}")
//                  logInfo(s"new ${newJoinPlan.plan} planCost ${newJoinPlan.planCost} rootCost ${newJoinPlan.rootCost(conf)}")
//                }
                nextLevel.update(newJoinPlan.itemIds, newJoinPlan)
              }
            case None =>
          }
        }
      }
      k += 1
    }
    nextLevel.toMap
  }

  /**
   * Builds a new JoinPlan if the following conditions hold:
   * - the sets of items contained in left and right sides do not overlap.
   * - there exists at least one join condition involving references from both sides.
   * - if star-join filter is enabled, allow the following combinations:
   *         1) (oneJoinPlan U otherJoinPlan) is a subset of star-join
   *         2) star-join is a subset of (oneJoinPlan U otherJoinPlan)
   *         3) (oneJoinPlan U otherJoinPlan) is a subset of non star-join
   *
   * @param oneJoinPlan One side JoinPlan for building a new JoinPlan.
   * @param otherJoinPlan The other side JoinPlan for building a new join node.
   * @param conf SQLConf for statistics computation.
   * @param conditions The overall set of join conditions.
   * @param topOutput The output attributes of the final plan.
   * @param filters Join graph info to be used as filters by the search algorithm.
   * @return Builds and returns a new JoinPlan if both conditions hold. Otherwise, returns None.
   */
  private def buildJoin(
      oneJoinPlan: JoinPlan,
      otherJoinPlan: JoinPlan,
      conf: SQLConf,
      conditions: Set[Expression],
      topOutput: AttributeSet,
      filters: Option[JoinGraphInfo]): Option[JoinPlan] = {

    if (oneJoinPlan.itemIds.intersect(otherJoinPlan.itemIds).nonEmpty) {
      // Should not join two overlapping item sets.
      return None
    }

    if (filters.isDefined) {
      // Apply star-join filter, which ensures that tables in a star schema relationship
      // are planned together. The star-filter will eliminate joins among star and non-star
      // tables until the star joins are built. The following combinations are allowed:
      // 1. (oneJoinPlan U otherJoinPlan) is a subset of star-join
      // 2. star-join is a subset of (oneJoinPlan U otherJoinPlan)
      // 3. (oneJoinPlan U otherJoinPlan) is a subset of non star-join
      val isValidJoinCombination =
        JoinReorderDPFilters.starJoinFilter(oneJoinPlan.itemIds, otherJoinPlan.itemIds,
          filters.get)
      if (!isValidJoinCombination) return None
    }

    val onePlan = oneJoinPlan.plan
    val otherPlan = otherJoinPlan.plan
    val joinConds = conditions
      .filterNot(l => canEvaluate(l, onePlan))
      .filterNot(r => canEvaluate(r, otherPlan))
      .filter(e => e.references.subsetOf(onePlan.outputSet ++ otherPlan.outputSet))
    if (joinConds.isEmpty) {
      // Cartesian product is very expensive, so we exclude them from candidate plans.
      // This also significantly reduces the search space.
      return None
    }

    // Put the deeper side on the left, tend to build a left-deep tree.
    val (left, right) = if (oneJoinPlan.itemIds.size >= otherJoinPlan.itemIds.size) {
      (onePlan, otherPlan)
    } else {
      (otherPlan, onePlan)
    }
    val newJoin = Join(left, right, Inner, joinConds.reduceOption(And))
    val collectedJoinConds = joinConds ++ oneJoinPlan.joinConds ++ otherJoinPlan.joinConds
    val remainingConds = conditions -- collectedJoinConds
    val neededAttr = AttributeSet(remainingConds.flatMap(_.references)) ++ topOutput
    val neededFromNewJoin = newJoin.output.filter(neededAttr.contains)
    val newPlan =
      if ((newJoin.outputSet -- neededFromNewJoin).nonEmpty) {
        Project(neededFromNewJoin, newJoin)
      } else {
        newJoin
      }

    val itemIds = oneJoinPlan.itemIds.union(otherJoinPlan.itemIds)
    // Now the root node of onePlan/otherPlan becomes an intermediate join (if it's a non-leaf
    // item), so the cost of the new join should also include its own cost.
    val newPlanCost = oneJoinPlan.planCost + oneJoinPlan.rootCost(conf) +
      otherJoinPlan.planCost + otherJoinPlan.rootCost(conf)

    logInfo(s"** ${oneJoinPlan.planCost} ${oneJoinPlan.rootCost(conf)}")
    logInfo(s"${otherJoinPlan.planCost} ${otherJoinPlan.rootCost(conf)}")
    logInfo(s"$oneJoinPlan")
    logInfo(s"$otherJoinPlan")

    Some(JoinPlan(itemIds, newPlan, collectedJoinConds, newPlanCost))
  }

  /** Map[set of item ids, join plan for these items] */
  type JoinPlanMap = Map[Set[Int], JoinPlan]

  def findBaseRel(plan: LogicalPlan): Option[LogicalPlan] = {
    plan match {
      case p: LeafNode => Some(p)
      case _ => findBaseRel(plan.children.head)
    }
  }

  lazy val allTables: Seq[TableIdentifier] =
    sessionCatalog.listTables(SessionCatalog.DEFAULT_DATABASE)

  lazy val allTableNamesSorted: Seq[String] = allTables.map(_.table).sorted

  lazy val allTableColumns: Seq[Seq[String]] =
    allTables.map(sessionCatalog.getTableMetadata(_)).map(_.schema.map(_.name))

  /** Static schema, not per-query data.  "tableName" -> Seq(col1, col2, ...). */
  lazy val tableToColumns: Map[String, Seq[String]] = allTableNamesSorted.zip(allTableColumns).toMap

  /** Static schema, not per-query data.  Seq(col1, col2, ...) -> tableName. */
  lazy val columnsToTable: Map[Seq[String], String] =
    allTableColumns.map(_.sorted).zip(allTableNamesSorted).toMap

  def getVisibleAttributes(plan: LogicalPlan): AttributeSet = {
    plan match {
      case p: LeafNode =>
//        val colNames = p.references.toSeq.map(_.name).sorted
//        val tableName = columnsToTable.get(colNames)
//
//        logInfo(s"colNames ${colNames} tableName ${tableName} p.references ${p.references}")
//        logInfo(s"allTables ${allTableNames}")
//        logInfo(s"p.baseTableName ${p.baseTableName}")

        p.references  // Assume filters/projects not pushed down.
      case o => o.children.map(getVisibleAttributes).reduce(_ ++ _)
    }
  }

//  /** Maps (id#1, anotherCol#111) into (20, 1), the ordinals of the attrs in DB. */
//  def denormalizeAttributes(attrs: AttributeSet): Seq[Int] = {
//    val translation = AnalysisContext.getAttributeMap
//    val attrStrings = attrs.map(attr => translation.getOrElse(attr.toString, null)).toSeq
//    val allAttrs = AnalysisContext.allAttributesSorted
//    attrStrings.map(allAttrs.indexOf(_))
//  }

  /**
   * Partial join order in a specific level.
   *
   * @param itemIds Set of item ids participating in this partial plan.
   * @param plan The plan tree with the lowest cost for these items found so far.
   * @param joinConds Join conditions included in the plan.
   * @param planCost The cost of this plan tree is the sum of costs of all intermediate joins.
   */
  case class JoinPlan(
      itemIds: Set[Int],
      plan: LogicalPlan,
      joinConds: Set[Expression],
      planCost: Cost) {

    /** Get the cost of the root node of this plan tree. */
    def rootCost(conf: SQLConf): Cost = {
      if (itemIds.size > 1) {
        val rootStats = plan.stats
        Cost(rootStats.rowCount.get, rootStats.sizeInBytes)
      } else {
        // If the plan is a leaf item, it has zero cost.
        Cost(0, 0)
      }
    }

    def betterThan(other: JoinPlan, conf: SQLConf): Boolean = {
      if (other.planCost.card == 0 || other.planCost.size == 0) {
        false
      } else {
//        LearningOptimizer.featurize()
        val relativeRows = BigDecimal(this.planCost.card) / BigDecimal(other.planCost.card)
        val relativeSize = BigDecimal(this.planCost.size) / BigDecimal(other.planCost.size)
        relativeRows * conf.joinReorderCardWeight +
          relativeSize * (1 - conf.joinReorderCardWeight) < 1
      }
    }
  }
}

/**
 * This class defines the cost model for a plan.
 * @param card Cardinality (number of rows).
 * @param size Size in bytes.
 */
case class Cost(card: BigInt, size: BigInt) {
  def +(other: Cost): Cost = Cost(this.card + other.card, this.size + other.size)

  /** NOTE(zongheng): should be kept in sync with JoinPlan.betterThan() above. */
  def combine(): Float = {
    val weight = SQLConf.get.joinReorderCardWeight
    // This represents the cost model of Spark SQL's cost-based optimizer.
    (BigDecimal(card) * weight + BigDecimal(size) * (1.0 - weight)).toFloat
  }
}

/**
 * Implements optional filters to reduce the search space for join enumeration.
 *
 * 1) Star-join filters: Plan star-joins together since they are assumed
 *    to have an optimal execution based on their RI relationship.
 * 2) Cartesian products: Defer their planning later in the graph to avoid
 *    large intermediate results (expanding joins, in general).
 * 3) Composite inners: Don't generate "bushy tree" plans to avoid materializing
 *   intermediate results.
 *
 * Filters (2) and (3) are not implemented.
 */
object JoinReorderDPFilters extends PredicateHelper {
  /**
   * Builds join graph information to be used by the filtering strategies.
   * Currently, it builds the sets of star/non-star joins.
   * It can be extended with the sets of connected/unconnected joins, which
   * can be used to filter Cartesian products.
   */
  def buildJoinGraphInfo(
      conf: SQLConf,
      items: Seq[LogicalPlan],
      conditions: Set[Expression],
      itemIndex: Seq[(LogicalPlan, Int)]): Option[JoinGraphInfo] = {

    if (conf.joinReorderDPStarFilter) {
      // Compute the tables in a star-schema relationship.
      val starJoin = StarSchemaDetection.findStarJoins(items, conditions.toSeq)
      val nonStarJoin = items.filterNot(starJoin.contains(_))

      if (starJoin.nonEmpty && nonStarJoin.nonEmpty) {
        val itemMap = itemIndex.toMap
        Some(JoinGraphInfo(starJoin.map(itemMap).toSet, nonStarJoin.map(itemMap).toSet))
      } else {
        // Nothing interesting to return.
        None
      }
    } else {
      // Star schema filter is not enabled.
      None
    }
  }

  /**
   * Applies the star-join filter that eliminates join combinations among star
   * and non-star tables until the star join is built.
   *
   * Given the oneSideJoinPlan/otherSideJoinPlan, which represent all the plan
   * permutations generated by the DP join enumeration, and the star/non-star plans,
   * the following plan combinations are allowed:
   * 1. (oneSideJoinPlan U otherSideJoinPlan) is a subset of star-join
   * 2. star-join is a subset of (oneSideJoinPlan U otherSideJoinPlan)
   * 3. (oneSideJoinPlan U otherSideJoinPlan) is a subset of non star-join
   *
   * It assumes the sets are disjoint.
   *
   * Example query graph:
   *
   * t1   d1 - t2 - t3
   *  \  /
   *   f1
   *   |
   *   d2
   *
   * star: {d1, f1, d2}
   * non-star: {t2, t1, t3}
   *
   * level 0: (f1 ), (d2 ), (t3 ), (d1 ), (t1 ), (t2 )
   * level 1: {t3 t2 }, {f1 d2 }, {f1 d1 }
   * level 2: {d2 f1 d1 }
   * level 3: {t1 d1 f1 d2 }, {t2 d1 f1 d2 }
   * level 4: {d1 t2 f1 t1 d2 }, {d1 t3 t2 f1 d2 }
   * level 5: {d1 t3 t2 f1 t1 d2 }
   *
   * @param oneSideJoinPlan One side of the join represented as a set of plan ids.
   * @param otherSideJoinPlan The other side of the join represented as a set of plan ids.
   * @param filters Star and non-star plans represented as sets of plan ids
   */
  def starJoinFilter(
      oneSideJoinPlan: Set[Int],
      otherSideJoinPlan: Set[Int],
      filters: JoinGraphInfo) : Boolean = {
    val starJoins = filters.starJoins
    val nonStarJoins = filters.nonStarJoins
    val join = oneSideJoinPlan.union(otherSideJoinPlan)

    // Disjoint sets
    oneSideJoinPlan.intersect(otherSideJoinPlan).isEmpty &&
      // Either star or non-star is empty
      (starJoins.isEmpty || nonStarJoins.isEmpty ||
        // Join is a subset of the star-join
        join.subsetOf(starJoins) ||
        // Star-join is a subset of join
        starJoins.subsetOf(join) ||
        // Join is a subset of non-star
        join.subsetOf(nonStarJoins))
  }
}

/**
 * Helper class that keeps information about the join graph as sets of item/plan ids.
 * It currently stores the star/non-star plans. It can be
 * extended with the set of connected/unconnected plans.
 */
case class JoinGraphInfo (starJoins: Set[Int], nonStarJoins: Set[Int])
// scalastyle:on
