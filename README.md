# ml-go
A simple machine learning library for Go.

## Setup

Use `go get` to retreive the package.
```
go get github.com/jpmendel/ml-go
```

## Usage

### Simple Neural Networks

```go
import "github.com/jpmendel/ml-go/nn"

// Start building a neural network.
neuralNetwork := nn.NewNeuralNetwork()

// Add layers to construct the neural network. Outputs of each layer must match the inputs of the next one.
neuralNetwork.Add(
    nn.NewDenseLayer(2, 3, nn.ActivationSigmoid),
    nn.NewDenseLayer(3, 1, nn.ActivationSidmoid),
)

myTrainingData := [][][][]float32{ ... }
myTargets := [][][][]float32 { ... }

// Train neural network.
for i := 0; i < len(myData); i++ {
    // Use input data, target answer, learning rate, and momentum.
    neuralNetwork.Train(myTrainingData[i], myTargets[i], 0.2, 0.3)
}

myTestData := [][][]float32{ ... }

// Make prediction.
prediction, _ := neuralNetwork.Predict(myTestData)

/* ... use prediction ... */
```

### Convolutional Neural Networks (work in progress)

```go
import (
    "github.com/jpmendel/ml-go/tensor"
    "github.com/jpmendel/ml-go/nn"
)

// Start building a neural network.
neuralNetwork := nn.NewNeuralNetwork()

// ConvolutionLayer applies various operations to the input data.
convolutionLayer := nn.NewConvolutionLayer(16, 16, 1, []*tensor.Tensor{nn.FilterVerticalEdges, nn.FilterHorizontalEdges}, nn.ActivationRELU)

// PoolingLayer subsamples data down to a smaller size.
poolingLayer := nn.NewPoolingLayer(16, 16, 2, 2, nn.PoolingMax)

// FlattenLayer converts multi-dimensional data into one-dimensional data.
flattenLayer := nn.NewFlattenLayer(4, 4, 2)

// DenseLayer is a fully connected layer with weights and bias.
denseLayer1 := nn.NewDenseLayer(32, 16, nn.ActivationSigmoid)
denseLayer2 := nn.NewDenseLayer(16, 8, nn.ActivationSoftmax)

// Add layers to network.
neuralNetwork.Add(
    convolutionLayer,
    poolingLayer,
    flattenLayer,
    denseLayer1,
    denseLayer2,
)

/* ... train and predict ... */
```

### Store and Load Neural Networks with JSON
```go
import "github.com/jpmendel/ml-go/nn"

// Start building a neural network.
neuralNetwork := nn.NewNeuralNetwork()

// Construct neural network.
neuralNetwork.Add(
    nn.NewDenseLayer(2, 3, nn.ActivationSigmoid),
    nn.NewDenseLayer(3, 1, nn.ActivationSidmoid),
)

// Save the neural network configuration.
neuralNetwork.SaveToFile("nn.json")

// Load the neural network configuration.
loadedNeuralNetwork := NewNeuralNetwork()
loadedNeuralNetwork.LoadFromFile("nn.json")
```