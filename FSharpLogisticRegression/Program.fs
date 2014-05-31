// Learn more about F# at http://fsharp.net
// See the 'F# Tutorial' project for more help.
namespace FunctionalServer 

module Program = 

    open System
    open System.Windows.Forms

    open FSharp.Charting
    open OnlineLogisticRegression
    open MathNet.Numerics
    open MathNet.Numerics.Distributions
    open MathNet.Numerics.LinearAlgebra.Double

    /// Feature Type, either positive or negative, tuple of the feature and its outcome
    type Feature =
        | Positive of DenseVector * float
        | Negative of DenseVector * float

    /// Gets the feature class (real value, 0.0 or 1.0) of a feature type
    let FeatureClass feature =
        match feature with
        | Positive(_, x) -> x
        | Negative(_, x) -> x

    /// Gets the feature vector of a feature type
    let FeatureVector feature =
        match feature with
        | Positive(x, _) -> x
        | Negative(x, _) -> x

    /// Samples a new feature with the given dimension and random number generator. 
    /// Class 0 generated from a normal distribution of mean = 25 and standard deviation = 10.
    /// Class 1 generated from a normal distribution of mean = 75 and standard deviation = 10.
    /// This yields to two distinct clusters that are easily separable. 
    /// Both classes are generated with equal chance by flipping a coin (rand() > 0.5).
    let SampleNewFeature(dimension : int, rnd : System.Random) : Feature =
        if rnd.NextDouble() > 0.5 
        // pass the random source to prevent the default rand be initialized with the same tick, thus giving the same random values
        then Positive(DenseVector.CreateRandom(dimension, Normal(25.0, 10.0, RandomSource = rnd)), 1.0)
        else Negative(DenseVector.CreateRandom(dimension, Normal(75.0, 10.0, RandomSource = rnd)), 0.0)
 
    /// Returns a sequence that only contains the plottable dimensions of a vector (0 and 1).
    let GetPlottableDimension(featureSequence : seq<DenseVector>) =
        Seq.map(fun (vec : DenseVector) -> vec.At(0), vec.At(1)) featureSequence

    /// Generates coordinates for the decision boundary, given a feature and the learned weights.
    let GetCoordinates(feat : DenseVector, weights : DenseVector) = 
        // this boundary is defined by: 
        // 0 = feature dot theta
        // feature = [1, x, y]
        // thus solving for y yields:
        // 0 = theta(0) + x * theta(1) + y * theta(2)
        // -y * theta(2) = theta(0) + x * theta(1)
        // y = -(theta(0) + x * theta(1)) / theta(2)
        feat.At(0), -1.0 * (weights.At(1) * feat.At(0) + weights.At(0)) / weights.At(2)
    
    /// Plots positive and negative features including the decision boundary.
    let PlotFeatures(positiveFeatures : seq<DenseVector>, negativeFeatures : seq<DenseVector>, weights : DenseVector) =
        
        let all = Seq.append positiveFeatures negativeFeatures
        let minFeature = Seq.minBy (fun (x : DenseVector) -> x.ToArray()) all
        let maxFeature = Seq.maxBy (fun (x : DenseVector) -> x.ToArray()) all
        
        let chartSeq = Seq.ofList([ 
                                    Chart.Point(GetPlottableDimension positiveFeatures, Color = Drawing.Color.Green); 
                                    Chart.Point(GetPlottableDimension negativeFeatures, Color = Drawing.Color.Red); 
                                    Chart.Line(Seq.ofList [ GetCoordinates(minFeature, weights); GetCoordinates(maxFeature, weights)])
                                  ])

        Chart.Combine(chartSeq).ShowChart()

    /// Stochastic trainining by sampling a new feature vector every iteration and doing a gradient descent step.
    /// Returns the learned weights.
    // TODO rewrite this by monitoring the difference in theta to detect convergence, break on divergence
    let Train(numIterations, dim, learningRate, rnd) = 
        // new vector including the bias
        let mutable start = DenseVector.CreateRandom(dim + 1, Normal())
        for i = 0 to numIterations do
            let feature = SampleNewFeature(dim, rnd)
            start <- OnlineLogisticRegression.GradientDescentStep(learningRate, start, FeatureVector(feature), FeatureClass(feature))

            if i % (numIterations/1000) = 0 then
                Console.WriteLine("{0} -> {1}", i, System.String.Join(", ", start.ToArray()))

        Console.WriteLine("Learned Weights: {0}", System.String.Join(", ", start.ToArray()))
        start

    /// Tests the learned weights, by sampling new features and predicting the new outcome.
    /// The resulting classification is then evaluated using the Accuracy metric and plotted.
    let Test(testSetSize, dim, rnd, weights) =
        let testSetSource = List.init testSetSize (fun (x) -> SampleNewFeature(dim, rnd))
        let testSet = 
            Seq.ofList testSetSource
            |> Seq.map (fun(feat) -> feat, OnlineLogisticRegression.Predict(FeatureVector(feat), weights))
            |> Seq.map (fun(feat, prediction) -> feat, prediction, abs(FeatureClass(feat) - prediction) < 0.5)
        let filterFeatureByClass classValue = Seq.ofList(List.map(fun(feat) -> FeatureVector(feat)) (List.filter(fun(feat) -> FeatureClass(feat) = classValue) testSetSource))
        let countCorrectPredictions = Seq.sumBy (fun(feat, prediction, correct) -> if correct then 1 else 0)
        let numCorrect = countCorrectPredictions testSet
      
        Console.WriteLine("Num correct: {0} of {1} = {2:P2} accuracy", numCorrect, testSetSize, float(numCorrect) / float(testSetSize))

        // plot the result
        PlotFeatures(filterFeatureByClass 1.0, filterFeatureByClass 0.0, weights)

    [<EntryPoint>]
    let main argv = 
        
        let dim = 2
        let numSamplesPerClass = 1000;
        let rnd = System.Random()
        let numIterations = 100000
        let learningRate = 0.1
        let testSetSize = 1000

        let weights = Train(numIterations, dim, learningRate, rnd)
    
        Test(testSetSize, dim, rnd, weights)
        
        ignore(Console.ReadKey())
        0 // return an integer exit code


