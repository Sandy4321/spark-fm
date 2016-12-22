# spark-fm
Factorization Machines is a general predictor like SVMs but is also able to estimate reliable parameters under very high sparsity. However, they are costly to scale to large amounts of data and large numbers of features. spark-fm is a parallel implementation of factorization machines based on Spark. It aims to utilize Spark's in-memory computing to address above problems.

# Highlight
In order to meet users' demands, spark-fm supports various of optimization methods to train the model as follows.
 + Mini-batch Stochastic Gradient Descent ("gd")
 + Parallel Stochastic Gradient Descent ("pgd") [[reference]](http://www.research.rutgers.edu/~lihong/pub/Zinkevich11Parallelized.pdf)
 + L-BFGS ("l-bfgs")
 + ALS // todo
 + MCMC // todo

# Examples
## Scala API
```scala
    val spark = SparkSession
        .builder()
        .appName("FactorizationMachinesExample")
        .master("local[4]")
        .getOrCreate()

    val train = spark.read.format("libsvm").load("data/a9a")
    val test = spark.read.format("libsvm").load("data/a9a.t")
    val trainer = new FactorizationMachines()
        .setTask("classification")
        .setSolver("gd")
        .setInitialStd(0.01)
        .setStepSize(0.01)
        .setUseBiasTerm(true)
        .setUseLinearTerms(true)
        .setNumFactors(8)
        .setRegParams((0, 1e-3, 1e-4))
        .setTol(1e-3)
        .setMaxIter(5)
        .setMiniBatchFraction(1)

    val model = trainer.fit(train)
    val result = model.transform(test)
    val predictionAndLabel = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    println("Accuracy: " + evaluator.evaluate(predictionAndLabel))
    spark.stop()
```

# Requirements
spark-fm is built against Spark 2.0.1.

# Build From Source
```scala
sbt package
```

# Licenses
spark-fm is available under Apache Licenses 2.0.

# Contact & Feedback
If you encounter bugs, feel free to submit an issue or pull request. Also you can mail to:
+ Chen Lin (m2linchen@gmail.com).

# Acknowledgement
Special thanks to [Qian Huang](https://github.com/hqzizania) for his contributions to this project.
