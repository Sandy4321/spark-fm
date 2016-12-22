package org.apache.spark.ml.fm

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.optimization.ParallelGradientDescent
import org.apache.spark.ml.param.shared.{HasMaxIter, HasStepSize, HasThreshold, HasTol}
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{PredictionModel, Predictor, PredictorParams}
import org.apache.spark.mllib.linalg.{DenseVector, Vector => MLlibVector, Vectors => MLlibVectors}
import org.apache.spark.mllib.optimization.{Gradient, GradientDescent, LBFGS, Updater}
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.sql.Dataset
import org.apache.spark.storage.StorageLevel

import scala.util.Random

/** Params for Factorization Machines. */
private[ml] trait FactorizationMachinesParams extends PredictorParams
  with HasMaxIter with HasTol with HasStepSize with HasThreshold {

  /**
    * The task that Factorization Machines are used to carry out.
    * Supported options: "regression" or "classification"(binary classification).
    * Default: "classification"
    *
    * @group expertParam
    */
  final val task: Param[String] = new Param[String](this, "task",
    "The task that Factorization Machines are used to carry out. Supported options: " +
      s"${FactorizationMachines.supportedTasks.mkString(",")}. (Default classification)",
    ParamValidators.inArray[String](FactorizationMachines.supportedTasks))

  /** @group expertGetParam */
  final def getTask: String = $(task)

  /**
    * The solver algorithm for optimization.
    * Supported options: "gd" (mini-batch stochastic gradient descent),
    * "pgd"(parallel stochastic gradient descent) or "l-bfgs".
    * Default: "gd"
    *
    * @group expertParam
    */
  final val solver: Param[String] = new Param[String](this, "solver",
    "The solver algorithm for optimization. Supported options: " +
      s"${FactorizationMachines.supportedSolvers.mkString(", ")}. (Default gd)",
    ParamValidators.inArray[String](FactorizationMachines.supportedSolvers))

  /** @group expertGetParam */
  final def getSolver: String = $(solver)

  /**
    * The initial weights of the model.
    *
    * @group expertParam
    */
  final val initialWeights: Param[Vector] = new Param[Vector](this, "initialWeights",
    "The initial weights of the model")

  /** @group expertGetParam */
  final def getInitialWeights: Vector = $(initialWeights)

  /**
    * The standard deviation for initializing weights.
    *
    * @group expertParam
    */
  final val initialStd: Param[Double] = new Param[Double](this,
    "initialStd", "The standard deviation for initializing weights")

  /** @group expertGetParam */
  final def getInitialStd: Double = $(initialStd)

  /**
    * Whether or not to use bias term.
    *
    * @group expertGetParam
    */
  final val useBiasTerm: Param[Boolean] = new Param[Boolean](this, "useBiasTerm",
    "Whether or not to use global bias term to train the model")

  /** @group expertGetParam */
  final def getUseBiasTerm: Boolean = $(useBiasTerm)

  /**
    * Whether or not to use linear terms.
    *
    * @group expertGetParam
    */
  final val useLinearTerms: Param[Boolean] = new Param[Boolean](this, "useLinearTerms",
    "Whether or not to use linear terms to train the model")

  /** @group expertGetParam */
  final def getUseLinearTerms: Boolean = $(useLinearTerms)

  /**
    * The number of factors that are used for pairwise interactions.
    *
    * @group expertGetParam
    */
  final val numFactors: Param[Int] = new Param[Int](this, "numFactors",
    "The number of factors that are used for pairwise interactions")

  /** @group expertGetParam */
  final def getNumFactors: Int = $(numFactors)

  /**
    * The regularization parameters of bias term, linear terms and pairwise interactions, respectively.
    *
    * @group expertGetParam
    */
  final val regParams: Param[(Double, Double, Double)] =
    new Param[(Double, Double, Double)](this, "regularization",
    "The regularization parameters of bias term, linear terms and pairwise interactions, respectively.")

  /** @group expertGetParam */
  final def getRegParams: (Double, Double, Double) = $(regParams)

  /**
    * Fraction of data to be used per iteration.
    *
    * @group expertGetParam
    */
  final val miniBatchFraction: Param[Double] = new Param[Double](this,
    "mini-batch fraction", "Fraction of data to be used per iteration.")

  /** @group expertGetParam */
  final def getMiniBatchFraction: Double = $(miniBatchFraction)

  setDefault(maxIter -> 5, tol -> 1e-4, stepSize -> 0.01, initialStd -> 0.1,
    useBiasTerm -> true, useLinearTerms -> true, numFactors -> 8,
    regParams -> (0, 1e-3, 1e-4), solver -> FactorizationMachines.GD,
    task -> FactorizationMachines.Classification,
    threshold -> 0.5, miniBatchFraction -> 1)
}

/**
  * Factorization Machines
  *
  * @param uid
  */
class FactorizationMachines(override val uid: String)
  extends Predictor[Vector, FactorizationMachines, FactorizationMachinesModel]
  with FactorizationMachinesParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("fm"))

  /**
    * Sets the value of param [[task]].
    * Default is "classification".
    *
    * @group expertSetParam
    */
  def setTask(value: String): this.type  = set(task, value)

  /**
    * Sets the value of param [[solver]].
    * Default is "gd".
    *
    * @group expertSetParam
    */
  def setSolver(value: String): this.type = set(solver, value)

  /**
    * Set the maximum number of iterations.
    * Default is 100.
    *
    * @group setParam
    */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /**
    * Sets the value of param [[initialWeights]].
    *
    * @group expertSetParam
    */
  def setInitialWeights(value: Vector): this.type = set(initialWeights, value)

  /**
    * Sets the value of param [[stepSize]].
    * Default is 0.03.
    *
    * @group setParam
    */
  def setStepSize(value: Double): this.type = {
    require($(solver) != FactorizationMachines.LBFGS,
      s"stepSize can not be applied in ${FactorizationMachines.LBFGS} solver.")
    set(stepSize, value)
  }

  /**
    * Sets the value of param [[initialStd]].
    * Default is 0.01.
    *
    * @group setParam
    */
  def setInitialStd(value: Double): this.type = set(initialStd, value)

  /**
    * Sets the value of param [[useBiasTerm]].
    * Default is true.
    *
    * @group setParam
    */
  def setUseBiasTerm(value: Boolean): this.type = set(useBiasTerm, value)

  /**
    * Sets the value of param [[useLinearTerms]].
    * Default is true.
    *
    * @group setParam
    */
  def setUseLinearTerms(value: Boolean): this.type = set(useLinearTerms, value)

  /**
    * Sets the value of param [[numFactors]].
    * Default is 8.
    *
    * @group setParam
    */
  def setNumFactors(value: Int): this.type = set(numFactors, value)

  /**
    * Sets the value of param [[regParams]].
    * Default is (0, 1e-3, 1e-4).
    *
    * @group setParam
    */
  def setRegParams(value: (Double, Double, Double)): this.type = set(regParams, value)

  /**
    * Sets the value of param [[threshold]].
    * Default is 0.5.
    *
    * @group setParam
    */
  def setThreshold(value: Double): this.type = set(threshold, value)

  /**
    * Sets the value of param [[tol]].
    * Default is 1e-4.
    *
    * @group setParam
    */
  def setTol(value: Double): this.type = set(tol, value)

  /**
    * Sets the value of param [[miniBatchFraction]].
    * Default is 1.
    *
    * @group setParam
    */
  def setMiniBatchFraction(value: Double): this.type = {
    require($(solver) == FactorizationMachines.GD,
      s"miniBatchFraction only works in ${FactorizationMachines.GD} solver.")
    set(miniBatchFraction, value)
  }

  override def copy(extra: ParamMap): FactorizationMachines = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): FactorizationMachinesModel = {
    val lpData = extractLabeledPoints(dataset)
    val numFeatures = lpData.first().features.size
    val weights = if (isDefined(initialWeights)) {
      $(initialWeights)
    } else {
      initializeWeights(numFeatures)
    }
    val trainData = lpData.map { lp =>
      (lp.label, MLlibVectors.fromML(lp.features))
    }
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) {
      trainData.persist(StorageLevel.MEMORY_AND_DISK)
    }
    val gradient = new FactorizationMachinesGradient(
      $(task), $(useBiasTerm), $(useLinearTerms), $(numFactors), numFeatures)
    val updater = new FactorizationMachinesUpdater(
      $(useBiasTerm), $(useLinearTerms), $(numFactors), $(regParams), numFeatures)
    val optimizer = if ($(solver) == FactorizationMachines.GD) {
      new GradientDescent(gradient, updater)
        .setStepSize($(stepSize))
        .setConvergenceTol($(tol))
        .setNumIterations($(maxIter))
        .setMiniBatchFraction($(miniBatchFraction))
    } else if ($(solver) == FactorizationMachines.PGD){
      new ParallelGradientDescent(gradient, updater)
        .setStepSize($(stepSize))
        .setConvergenceTol($(tol))
        .setNumIterations($(maxIter))
    } else if ($(solver) == FactorizationMachines.LBFGS){
      new LBFGS(gradient, updater)
        .setConvergenceTol($(tol))
        .setNumIterations($(maxIter))
    } else {
      throw new IllegalArgumentException(
        s"The solver $solver is not supported by FactorizationMachines.")
    }
    val newWeights = optimizer.optimize(trainData, weights)
    if (handlePersistence) {
      trainData.unpersist()
    }
    FactorizationMachinesModel(uid, $(task), newWeights, $(useBiasTerm), $(useLinearTerms),
      $(numFactors), $(threshold), numFeatures)
  }

  protected def initializeWeights(numFeatures: Int): Vector = {
    val initMean = 0
    ($(useBiasTerm), $(useLinearTerms)) match {
      case (true, true) =>
        Vectors.dense(Array.fill(numFeatures * $(numFactors))(Random.nextGaussian() * $(initialStd) + initMean) ++
          Array.fill(numFeatures + 1)(0.0))

      case (true, false) =>
        Vectors.dense(Array.fill(numFeatures * $(numFactors))(Random.nextGaussian() * $(initialStd) + initMean) ++
          Array(0.0))

      case (false, true) =>
        Vectors.dense(Array.fill(numFeatures * $(numFactors))(Random.nextGaussian() * $(initialStd) + initMean) ++
          Array.fill(numFeatures)(0.0))

      case (false, false) =>
        Vectors.dense(Array.fill(numFeatures * $(numFactors))(Random.nextGaussian() * $(initialStd) + initMean))
    }
  }
}

object FactorizationMachines {
  /** String name for "gd" (minibatch gradient descent) solver. */
  private[ml] val GD = "gd"

  /** String name for "pgd" (parallel stochastic gradient descent) solver. */
  private[ml] val PGD = "pgd"

  /** String name for "l-bfgs" solver. */
  private[ml] val LBFGS = "l-bfgs"

  /** Set of solvers that Factorization Machines support. */
  private[ml] val supportedSolvers = Array(GD, PGD, LBFGS)

  /** String name for "regression" task. */
  private[ml] val Regression = "regression"

  /** String name for "classification"(binary classification) task. */
  private[ml] val Classification = "classification"

  /** Set of tasks that Factorization Machines support. */
  private[ml] val supportedTasks = Array(Regression, Classification)
}

class FactorizationMachinesModel private[ml](
    override val uid: String,
    task: String,
    weights: Vector,
    useBiasTerm: Boolean,
    useLinearTerms: Boolean,
    numFactors: Int,
    threshold: Double,
    numFeatures: Int)
  extends PredictionModel[Vector, FactorizationMachinesModel]
  with Serializable {

  override protected def predict(features: Vector): Double = {
    val (prediction, _) = FactorizationMachinesModel.predictAndSum(
      features, weights, useBiasTerm, useLinearTerms, numFactors, numFeatures)
    if (task == FactorizationMachines.Regression) {
      prediction
    } else if (task == FactorizationMachines.Classification) {
      val output = 1 / (1 + Math.exp(-prediction))
      if (output > threshold) {
        1.0d
      } else {
        -1.0d
      }
    } else {
      throw new IllegalArgumentException(
        s"The task $task is not supported by FactorizationMachines.")
    }
  }

  override def copy(extra: ParamMap): FactorizationMachinesModel = {
    copyValues(new FactorizationMachinesModel(
      uid, task, weights, useBiasTerm, useLinearTerms, numFactors, threshold, numFeatures), extra)
  }
}

object FactorizationMachinesModel {
  /**
    * Creates a model from weights
    *
    * @param weights weights
    * @return model
    */
  def apply(
      uid: String,
      task: String,
      weights: Vector,
      useBiasTerm: Boolean,
      useLinearTerms: Boolean,
      numFactors: Int,
      threshold: Double,
      numFeatures: Int): FactorizationMachinesModel = {
    new FactorizationMachinesModel(
      uid, task, weights, useBiasTerm, useLinearTerms, numFactors, threshold, numFeatures)
  }

  def predictAndSum(
      features: Vector,
      weights: Vector,
      useBiasTerm: Boolean,
      useLinearTerms: Boolean,
      numFactors: Int,
      numFeatures: Int): (Double, Array[Double]) = {

    var prediction = if (useBiasTerm) {
      weights(weights.size - 1)
    }  else {
      0.0d
    }

    if (useLinearTerms) {
      val base = numFeatures * numFactors
      features.foreachActive {
        case (k, v) =>
          prediction += weights(base + k) * v
      }
    }

    val sum = Array.fill(numFactors)(0.0)
    for (i <- 0 until numFactors) {
      var sumSqr = 0.0
      features.foreachActive {
        case (k, v) =>
          val t = weights(k * numFactors + i) * v
          sum(i) += t
          sumSqr += t * t
      }
      prediction += (sum(i) * sum(i) - sumSqr) / 2
    }

    (prediction, sum)
  }
}

/**
  * :: DeveloperApi ::
  * Compute gradient and loss for a Least-squared loss function, as used in factorization machines.
  * For the detailed mathematical derivation, see the reference at
  * http://doi.acm.org/10.1145/2168752.2168771
  */
class FactorizationMachinesGradient(
    val task: String,
    val useBiasTerm: Boolean,
    val useLinearTerms: Boolean,
    val numFactors: Int,
    val numFeatures: Int) extends Gradient {

  override def compute(
      data: MLlibVector,
      label: Double,
      weights: MLlibVector): (MLlibVector, Double) = {
    val cumGradient = Vectors.dense(Array.fill(weights.size)(0.0))
    val loss = compute(data, label, weights, cumGradient)
    (cumGradient, loss)
  }

  override def compute(
      data: MLlibVector,
      label: Double,
      weights: MLlibVector,
      cumGradient: MLlibVector): Double = {
    require(data.size == numFeatures)
    val (prediction, sum) = FactorizationMachinesModel.predictAndSum(
      data, weights, useBiasTerm, useLinearTerms, numFactors, numFeatures)
    val multiplier = task match {
      case FactorizationMachines.Regression =>
        prediction - label
      case FactorizationMachines.Classification =>
        label * (1.0 / (1.0 + Math.exp(-prediction * label)) - 1.0)
    }

    cumGradient match {
      case vec: DenseVector =>
        val cumValues = vec.values

        if (useBiasTerm) {
          cumValues(cumValues.length - 1) += multiplier
        }

        if (useLinearTerms) {
          val pos = numFeatures * numFactors
          data.foreachActive {
            case (k, v) =>
              cumValues(pos + k) += v * multiplier
          }
        }

        data.foreachActive {
          case (k, v) =>
            val pos = k * numFactors
            for (f <- 0 until numFactors) {
              cumValues(pos + f) += (sum(f) * v - weights(pos + f) * v * v) * multiplier
            }
        }

      case _ =>
        throw new IllegalArgumentException(
          s"cumulateGradient only supports adding to a dense vector but got type ${cumGradient.getClass}.")
    }

    task match {
      case FactorizationMachines.Regression =>
        (prediction - label) * (prediction - label)
      case FactorizationMachines.Classification =>
        -Math.log(1 + 1 / (1 + Math.exp(-prediction * label)))
    }
  }

}


class FactorizationMachinesUpdater(
    useBiasTerm: Boolean,
    useLinearTerms: Boolean,
    numFactors: Int,
    regParams: (Double, Double, Double),
    val numFeatures: Int) extends Updater {

  override def compute(
      weightsOld: MLlibVector,
      gradient: MLlibVector,
      stepSize: Double,
      iter: Int,
      regParam: Double): (MLlibVector, Double) = {
    val r0 = regParams._1
    val r1 = regParams._2
    val r2 = regParams._3

    val thisIterStepSize = stepSize / math.sqrt(iter)
    val size = weightsOld.size

    val weightsNew = Array.fill(size)(0.0)
    var regVal = 0.0

    if (useBiasTerm) {
      weightsNew(size - 1) = weightsOld(size - 1) - thisIterStepSize * (gradient(size - 1) + r0 * weightsOld(size - 1))
      regVal += regParams._1 * weightsNew(size - 1) * weightsNew(size - 1)
    }

    if (useLinearTerms) {
      for (i <- numFeatures * numFactors until numFeatures * numFactors + numFeatures) {
        weightsNew(i) = weightsOld(i) - thisIterStepSize * (gradient(i) + r1 * weightsOld(i))
        regVal += r1 * weightsNew(i) * weightsNew(i)
      }
    }

    for (i <- 0 until numFeatures * numFactors) {
      weightsNew(i) = weightsOld(i) - thisIterStepSize * (gradient(i) + r2 * weightsOld(i))
      regVal += r2 * weightsNew(i) * weightsNew(i)
    }

    (Vectors.dense(weightsNew), regVal / 2)
  }
}