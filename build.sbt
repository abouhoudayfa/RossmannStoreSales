name := "RossmannStoreSales"

version := "0.1"

scalaVersion := "2.11.8"

lazy val sparkVersion = "2.2.0"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.11" % sparkVersion % "provided",
  "org.apache.spark" % "spark-sql_2.11" % sparkVersion % "provided",
  "org.apache.spark" % "spark-mllib_2.11" % sparkVersion % "provided"
)