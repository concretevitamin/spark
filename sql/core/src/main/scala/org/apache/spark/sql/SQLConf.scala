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

package org.apache.spark.sql

import java.util.Properties

import scala.collection.JavaConverters._

/**
 * SQLConf holds mutable config parameters and hints.  These can be set and
 * queried either by passing SET commands into Spark SQL's DSL
 * functions (sql(), hql(), etc.), or by programmatically using setters and
 * getters of this class.  This class is thread-safe.
 */
class SQLConf {

  /** Number of partitions to use for shuffle operators. */
  def numShufflePartitions(default: Int = 150): Int = getOption("spark.sql.shuffle.partitions")
    .map(_.toInt).getOrElse(default)

  private val settings = java.util.Collections.synchronizedMap(
    new java.util.HashMap[String, String]())

  def this(props: Properties) = {
    this()
    props.asScala.foreach { case (k, v) => this.settings.put(k, v) }
  }

  def set(key: String, value: String): SQLConf = {
    require(key != null, "key cannot be null")
    require(value != null, s"value cannot be null for ${key}")
    settings.put(key, value)
    this
  }

  def get(key: String): String = {
    if (!settings.containsKey(key)) {
      throw new NoSuchElementException(key)
    }
    settings.get(key)
  }

  def get(key: String, defaultValue: String): String = {
    if (!settings.containsKey(key)) defaultValue else settings.get(key)
  }

  def getAll: Array[(String, String)] = settings.asScala.toArray

  def getOption(key: String): Option[String] = {
    if (!settings.containsKey(key)) None else Some(settings.get(key))
  }

  def contains(key: String): Boolean = settings.containsKey(key)

  def toDebugString: String = {
    settings.synchronized {
      settings.asScala.toArray.sorted.map{ case (k, v) => s"$k=$v" }.mkString("\n")
    }
  }

  private[spark] def clear() {
    settings.clear()
  }

}
