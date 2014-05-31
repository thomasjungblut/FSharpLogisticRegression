namespace FunctionalServer 

module OnlineLogisticRegression = 

    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra.Double

    /// Prepends a 1 at the beginning of a newly created vector, copies the rest of the dimension.
    let CreateBiasVector(vector: DenseVector) = 
        DenseVector.Create(vector.Count + 1, new System.Func<int, float>(function i -> match i with | i when i = 0 -> 1.0 | _ -> vector.At(i - 1)))

    /// Applies the logistic function on the input vector
    let Sigmoid input = 
        SpecialFunctions.Logistic(input)

    /// Applies the logistic function on every element in the input vector '''inplace''' = doesn't return a new vector.
    let SigmoidVector (vec : DenseVector) = 
        vec.MapInplace(fun (x) -> Sigmoid x)
        vec
 
    /// Predicts the given vector using the given parameters.
    /// This automatically creates a bias if there aren't enough dimensions.
    let Predict (currentFeatureVector : DenseVector, currentParameters : DenseVector) = 
        let biased = if currentFeatureVector.Count < currentParameters.Count then CreateBiasVector currentFeatureVector else currentFeatureVector
        Sigmoid(biased.DotProduct currentParameters)

    /// Does a stochastic gradient descent step. Returns a new vector with the updated weights.
    let GradientDescentStep(learningRate, currentParameters, currentFeatureVector, outcome) =
        let biasedFeature = CreateBiasVector currentFeatureVector
        let prediction = Predict(biasedFeature, currentParameters)
        let loss = prediction - outcome
        // do a gradient descent step into the gradient direction
        DenseVector.OfVector(currentParameters.Subtract(biasedFeature.Multiply(loss).Multiply(learningRate)))
