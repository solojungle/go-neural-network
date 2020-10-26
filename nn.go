package main

import (
	"gonum.org/v1/gonum/mat"
)

// Network is the object that holds the model state
type Network struct {
	Layers          []Layer
	NeuronsPerLayer int
	NumberOfLayers  int
}

// Layer holds computational state to prevent reevaluating a needed variable more than once
type Layer struct {
	Bias                *mat.Dense
	Weights             *mat.Dense
	Delta               *mat.Dense                        // cumulative layer error delta (𝚫)
	Gradient            *mat.Dense                        // gradient dZ (derivative of weighted outputs)
	Activations         *mat.Dense                        // layer activations
	ActivationFunc      func(i, j int, v float64) float64 // activation function f(x)
	ActivationPrimeFunc func(i, j int, v float64) float64 // derivative of activation f’(x)
	IsLastLayer         bool                              // boolean to show if current layer is output layer
}

// NewNetwork creates a new neural network object
func NewNetwork(neuronsPerLayer, numberOfHiddenLayers, numberOfInputs int) (*Network, error) {
	network := new(Network)
	network.Layers = make([]Layer, numberOfHiddenLayers+2)

	// Create input layer
	inputWeights := KaimingInitialization(numberOfInputs, neuronsPerLayer)
	inputLayer := layerConstructor(inputWeights, false)
	network.Layers[0] = *inputLayer

	// Populate hidden layers array
	// Start after input, end right before output
	for i := 1; i < numberOfHiddenLayers+1; i++ {
		weights := KaimingInitialization(neuronsPerLayer, neuronsPerLayer)
		temp := layerConstructor(weights, false)
		network.Layers[i] = *temp
	}

	// Create the output layer (single neuron)
	weights := KaimingInitialization(neuronsPerLayer, 1)
	temp := layerConstructor(weights, true)
	network.Layers[numberOfHiddenLayers+1] = *temp

	// Set network information
	network.NeuronsPerLayer = neuronsPerLayer
	network.NumberOfLayers = numberOfHiddenLayers + 2

	return network, nil
}

// layerConstructor is a private function that creates layers for the neural network object
func layerConstructor(weights *mat.Dense, isLastLayer bool) *Layer {
	layer := new(Layer)

	// Set user inputs
	layer.Weights = weights
	layer.IsLastLayer = isLastLayer

	// Set functions
	layer.ActivationFunc = Sigmoid
	layer.ActivationPrimeFunc = SigmoidDerivative

	// Create bias
	_, cols := layer.Weights.Dims()
	layer.Bias = mat.NewDense(1, cols, nil)

	// Must create empty structs to avoid a nil pointer dereference when hard copying
	layer.Delta = mat.NewDense(1, 1, nil)
	layer.Gradient = mat.NewDense(1, 1, nil)
	layer.Activations = mat.NewDense(1, 1, nil)

	return layer
}

// Predict will run inputs through the network and return a probability
func (network *Network) Predict(input *mat.Dense) *mat.Dense {
	currentInput := input
	for _, layer := range network.Layers {
		layer.ForwardPass(currentInput)
		currentInput = layer.Activations
	}

	// strconv.ParseFloat(fmt.Sprintf("%.2f", v), 64)
	return currentInput
}

// ForwardPass is a single step function that sets a layer's state for Activations, and Gradient
func (layer *Layer) ForwardPass(input *mat.Dense) {
	z := Multiply(input, layer.Weights) // Z = Input ⋅ Weight
	zB := Add(z, layer.Bias)            // Adding in the bias

	// Hard copy struct pointers *a = *b
	*layer.Activations = *Update(Map(layer.ActivationFunc, zB))   // Set activations
	*layer.Gradient = *Update(Map(layer.ActivationPrimeFunc, zB)) // Set deriv. of the weighted outputs (dZ)
}

// Train will use Predict, BackwarPass and UpdateWeights to do back propagation on the network
func (network *Network) Train(batchInput, batchTarget *mat.Dense, learningRate float64, epochs int) {
	// Training cycles (how many times to repeat the batch)
	lastLayer := network.Layers[network.NumberOfLayers-1]
	for ; epochs > 0; epochs-- {

		_, targetCols := batchTarget.Dims()
		inputRows, inputCols := batchInput.Dims()

		for i := 0; i < inputRows; i++ {
			// Extract input from batch
			inputRowFloat := batchInput.RawRowView(i)
			tempInput := mat.NewDense(1, inputCols, inputRowFloat)

			// Predict on given row to set state
			network.Predict(tempInput)

			// Backwards Propogation
			// Go backwards from the last layer
			for j := network.NumberOfLayers - 1; j >= 0; j-- {
				// Extract expected value from batch
				targetRowFloat := batchTarget.RawRowView(i)
				tempTarget := mat.NewDense(1, targetCols, targetRowFloat)

				// Calculate gradient for current layer
				network.Layers[j].BackwardPass(tempTarget, lastLayer)
				lastLayer = network.Layers[j]
			}

			// Get initial input and start calculating the new weights
			currInput := tempInput
			for j := 0; j < len(network.Layers); j++ {
				network.Layers[j].UpdateWeights(learningRate, currInput)
				currInput = network.Layers[j].Activations
			}
		}
	}
}

// BackwardPass is a single step function that sets the layer's state for Delta
func (layer *Layer) BackwardPass(target *mat.Dense, rightLayer Layer) {
	if layer.IsLastLayer {
		err := Subtract(layer.Activations, target)
		layer.Delta = Update(MultiplyElems(err, layer.Gradient))
		return
	}

	layer.Delta = Update(MultiplyElems(Multiply(rightLayer.Delta, rightLayer.Weights.T()), layer.Gradient))
}

// UpdateWeights will calculate the correct new weights and apply them to a layer
func (layer *Layer) UpdateWeights(learningRate float64, leftA *mat.Dense) {
	// Calculate and update weights
	ad := Multiply(leftA.T(), layer.Delta)
	change := Scale(learningRate, ad)
	layer.Weights = Update(Subtract(layer.Weights, change))

	// Calculate and update bias
	dB := Scale(learningRate, layer.Delta)
	layer.Bias = Update(Subtract(layer.Bias, dB))
}

// Print will display a formatted matrix
func (network *Network) Print() {
	var matrix *mat.Dense
	for i := 0; i < len(network.Layers); i++ {
		matrix = Concat(matrix, network.Layers[i].Weights)
	}

	Print(matrix)
}
