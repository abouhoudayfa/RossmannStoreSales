package com.rossmann.store.sales

import org.apache.log4j.Logger
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

object RossmannStoreSales extends App {
  //TODO you use spark until Windows, if Unix you mast delete this line
  System.setProperty("hadoop.home.dir", "C:\\hadoop")
  private lazy val logger = Logger.getLogger("")

  // Create SparkContext Instance
  val conf = new SparkConf()
    .setMaster("local[*]")
    .setAppName("RossmannSoreSales")

  val sc = new SparkContext(conf)
  val sqlContext = new SQLContext(sc)

    if (args.length < 3) {
      System.err.println("Usage: Titanic <train file> <test file> <output file>")
      System.exit(1)
    }

  private val trainFile = args(0)
  private val testFile = args(1)
  private val result = args(2)


  def linearRegressionPipeline(): TrainValidationSplit = {
    val lr = new LinearRegression()

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 0.25, 0.5, 0.75, 1.0))
      .build()

    // preprocessing & preparing Pipelines
    // Indexers & Encoders

    val stateHolidayIndexer = new StringIndexer()
      .setInputCol("StateHoliday")
      .setOutputCol("StateHolidayIndex")
    val schoolHolidayIndexer = new StringIndexer()
      .setInputCol("SchoolHoliday")
      .setOutputCol("SchoolHolidayIndex")
    val stateHolidayEncoder = new OneHotEncoder()
      .setInputCol("StateHolidayIndex")
      .setOutputCol("StateHolidayVec")
    val schoolHolidayEncoder = new OneHotEncoder()
      .setInputCol("SchoolHolidayIndex")
      .setOutputCol("SchoolHolidayVec")
    val dayOfMonthEncoder = new OneHotEncoder()
      .setInputCol("DayOfMonth")
      .setOutputCol("DayOfMonthVec")
    val dayOfWeekEncoder = new OneHotEncoder()
      .setInputCol("DayOfWeek")
      .setOutputCol("DayOfWeekVec")
    val storeEncoder = new OneHotEncoder()
      .setInputCol("Store")
      .setOutputCol("StoreVec")

    // assemble all vectors in to one vector to input to Model

    val assembler = new VectorAssembler()
      .setInputCols(Array("StoreVec", "DayOfWeekVec", "Open", "DayOfMonthVec", "StateHolidayVec", "SchoolHolidayVec"))
      .setOutputCol("features")

    // Pipeline

    val pipeline = new Pipeline()
      .setStages(Array(stateHolidayIndexer, schoolHolidayIndexer,
        stateHolidayEncoder, schoolHolidayEncoder, storeEncoder,
        dayOfWeekEncoder, dayOfMonthEncoder,
        assembler, lr))

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.75)
    trainValidationSplit
  }

  def loadTrainData(sqlContext: SQLContext): DataFrame = {
    val trainRaw = sqlContext
      .read.format("com.databricks.spark.csv")
      .option("header", "true")
      .load(trainFile)
      .repartition(6)
    trainRaw.registerTempTable("raw_training_data")

    sqlContext.sql(
      """SELECT
        double(Sales) label, double(Store) Store, int(Open) Open, double(DayOfWeek) DayOfWeek,
        StateHoliday, SchoolHoliday, (double(regexp_extract(Date, '\\d+-\\d+-(\\d+)', 1))) DayOfMonth
        FROM raw_training_data
      """).na.drop()
  }

  def loadTestData(sqlContext: SQLContext) = {
    val testRaw = sqlContext
      .read.format("com.databricks.spark.csv")
      .option("header", "true")
      .load(testFile)
      .repartition(6)
    testRaw.registerTempTable("raw_test_data")

    val testData = sqlContext.sql(
      """SELECT
        Id, double(Store) Store, int(Open) Open, double(DayOfWeek) DayOfWeek, StateHoliday,
        SchoolHoliday, (double(regexp_extract(Date, '\\d+-\\d+-(\\d+)', 1))) DayOfMonth
        FROM raw_test_data
        WHERE !(ISNULL(Id) OR ISNULL(Store) OR ISNULL(Open) OR ISNULL(DayOfWeek)
          OR ISNULL(StateHoliday) OR ISNULL(SchoolHoliday))
      """).na.drop() // weird things happen if you don't filter out the null values manually

    Array(testRaw, testData) // got to hold onto testRaw so we can make sure
    // to have all the prediction IDs to submit to kaggle if you want
  }

  def savePredictionsToCsv(predictions: DataFrame, testRaw: DataFrame) = {
    val tdOut = testRaw
      .select("Id")
      .distinct()
      .join(predictions, testRaw("Id") === predictions("PredId"), "outer")
      .select("Id", "Sales")
      .na.fill(0: Double) // some of our inputs were null so we have to
    // fill these with something
    tdOut
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .mode("overwrite")
      .save(result)
  }

  def fitModel(tvs: TrainValidationSplit, data: DataFrame) = {
    val Array(training, test) = data.randomSplit(Array(0.8, 0.2), seed = 12345)
    logger.info("Fitting data")
    val model = tvs.fit(training)
    logger.info("Now performing test on hold out set")
    val holdout = model.transform(test).select("prediction", "label")

    // have to do a type conversion for RegressionMetrics
    val rm = new RegressionMetrics(holdout.rdd.map(x =>
      (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

    logger.info("======================= Test Metrics ======================")
    logger.info("======================= Test Explained Variance: ==========")
    logger.info(rm.explainedVariance)
    logger.info("======================= Test R^2 Coef: ====================")
    logger.info(rm.r2)
    logger.info("======================= Test MSE: =========================")
    logger.info(rm.meanSquaredError)
    logger.info(s"======================= Test RMSE: ======================== ${1-(rm.r2)}")

    model
  }

  val data = loadTrainData(sqlContext)
  val Array(testRaw, testData) = loadTestData(sqlContext)

  testRaw.show(5)
  testRaw.printSchema()

  testData.show(5)
  testData.printSchema()

  //   The linear Regression Pipeline
  val linearTvs = linearRegressionPipeline()
  logger.info("evaluating linear regression")
  val lrModel = fitModel(linearTvs, data)
  logger.info("Generating kaggle predictions")
  val lrOut = lrModel.transform(testData)
    .withColumnRenamed("prediction", "Sales")
    .withColumnRenamed("Id", "PredId")
    .select("PredId", "Sales")
  savePredictionsToCsv(lrOut, testRaw)
}
