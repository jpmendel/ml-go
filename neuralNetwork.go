package ml

import (
	"encoding/json"
	"fmt"
	"os"
)

// NeuralNetwork is a basic fully connected neural network.
type NeuralNetwork struct {
	layers []Layer
}

// NewNeuralNetwork Creates a new instance of a NeuralNetwork.
func NewNeuralNetwork() *NeuralNetwork {
	return &NeuralNetwork{layers: []Layer{}}
}

// Copy creates a deep copy of the neural network.
func (neuralNetwork *NeuralNetwork) Copy() *NeuralNetwork {
	newNeuralNetwork := NewNeuralNetwork()
	for _, layer := range neuralNetwork.layers {
		newNeuralNetwork.AddLayer(layer.Copy())
	}
	return newNeuralNetwork
}

// LayerCount returns the number of layers in the neural network.
func (neuralNetwork *NeuralNetwork) LayerCount() int {
	return len(neuralNetwork.layers)
}

// LayerAt gets a fully connected layer at a certain index.
func (neuralNetwork *NeuralNetwork) LayerAt(index int) Layer {
	if index < 0 || index >= len(neuralNetwork.layers) {
		return nil
	}
	return neuralNetwork.layers[index]
}

// AddLayer adds a new fully connected layer to the end of the neural network.
func (neuralNetwork *NeuralNetwork) AddLayer(layer Layer) error {
	if len(neuralNetwork.layers) > 0 {
		lastLayer := neuralNetwork.layers[len(neuralNetwork.layers)-1]
		if lastLayer.OutputShape() != layer.InputShape() {
			return fmt.Errorf(
				"Output shape of last layer does not match input shape of new layer: (%d, %d, %d) != (%d, %d, %d)",
				lastLayer.OutputShape().Rows, lastLayer.OutputShape().Cols, lastLayer.OutputShape().Length,
				layer.InputShape().Rows, layer.InputShape().Cols, layer.InputShape().Length,
			)
		}
	}
	neuralNetwork.layers = append(neuralNetwork.layers, layer)
	return nil
}

// Predict generates a prediction for a certain set of inputs.
func (neuralNetwork *NeuralNetwork) Predict(inputs [][]float32) ([][]float32, error) {
	outputs, err := neuralNetwork.feedForward(inputs)
	if err != nil {
		return nil, err
	}
	outputArray := make([][]float32, outputs.Rows)
	for row := 0; row < outputs.Rows; row++ {
		outputArray[row] = make([]float32, outputs.Cols)
		for col := 0; col < outputs.Cols; col++ {
			outputArray[row][col] = outputs.Get(row, col)
		}
	}
	return outputArray, nil
}

// Train takes a set of inputs and their respective targets, and adjusts the layers to produce the
// given outputs through supervised learning.
func (neuralNetwork *NeuralNetwork) Train(inputs [][]float32, targets [][]float32, learningRate float32, momentum float32) error {
	outputs, err := neuralNetwork.feedForward(inputs)
	if err != nil {
		return err
	}
	deltas := NewMatrixWithValues(targets)
	err = deltas.SubtractMatrix(outputs)
	if err != nil {
		return err
	}
	return neuralNetwork.backPropagate(deltas, learningRate, momentum)
}

func (neuralNetwork *NeuralNetwork) feedForward(inputs [][]float32) (*Matrix, error) {
	nextInputs := []*Matrix{NewMatrixWithValues(inputs)}
	var err error
	for _, layer := range neuralNetwork.layers {
		nextInputs, err = layer.FeedForward(nextInputs)
		if err != nil {
			return nil, err
		}
	}
	return nextInputs[0], nil
}

func (neuralNetwork *NeuralNetwork) backPropagate(deltas *Matrix, learningRate float32, momentum float32) error {
	nextDeltas := []*Matrix{deltas}
	var err error
	for i := len(neuralNetwork.layers) - 1; i >= 0; i-- {
		layer := neuralNetwork.layers[i]
		nextDeltas, err = layer.BackPropagate(nextDeltas, learningRate, momentum)
		if err != nil {
			return err
		}
	}
	return nil
}

// SaveToFile saves a neural network to a file.
func (neuralNetwork *NeuralNetwork) SaveToFile(fileName string) error {
	file, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer file.Close()
	neuralNetworkData := struct {
		Layers []Layer `json:"layers"`
	}{
		Layers: neuralNetwork.layers,
	}
	return json.NewEncoder(file).Encode(neuralNetworkData)
}

// LoadFromFile loads a neural network from a file.
func (neuralNetwork *NeuralNetwork) LoadFromFile(fileName string) error {
	file, err := os.Open(fileName)
	if err != nil {
		return err
	}
	defer file.Close()
	neuralNetworkData := struct {
		Layers []map[string]interface{} `json:"layers"`
	}{}
	err = json.NewDecoder(file).Decode(&neuralNetworkData)
	if err != nil {
		return err
	}
	for _, layerData := range neuralNetworkData.Layers {
		layerType, _ := layerData["type"].(string)
		layer, err := layerForType(LayerType(layerType))
		if err != nil {
			return err
		}
		layerBytes, _ := json.Marshal(layerData)
		err = json.Unmarshal(layerBytes, layer)
		if err != nil {
			return err
		}
		err = neuralNetwork.AddLayer(layer)
		if err != nil {
			return err
		}
	}
	return nil
}
