package ml

import (
	"encoding/json"
	"fmt"
	"os"
)

// NeuralNetwork is a basic neural network that can handle multiple layer types.
type NeuralNetwork struct {
	layers []Layer
}

// NewNeuralNetwork Creates a new instance of a NeuralNetwork.
func NewNeuralNetwork() *NeuralNetwork {
	return &NeuralNetwork{layers: []Layer{}}
}

// Copy creates a deep copy of the neural network.
func (nn *NeuralNetwork) Copy() *NeuralNetwork {
	new := NewNeuralNetwork()
	for _, layer := range nn.layers {
		new.Add(layer.Copy())
	}
	return new
}

// LayerCount returns the number of layers in the neural network.
func (nn *NeuralNetwork) LayerCount() int {
	return len(nn.layers)
}

// LayerAt gets a layer at a certain index.
func (nn *NeuralNetwork) LayerAt(index int) Layer {
	if index < 0 || index >= len(nn.layers) {
		return nil
	}
	return nn.layers[index]
}

// Add adds a number of new layers to the neural network.
func (nn *NeuralNetwork) Add(layers ...Layer) error {
	for _, layer := range layers {
		if len(nn.layers) > 0 {
			lastLayer := nn.layers[len(nn.layers)-1]
			if lastLayer.OutputShape() != layer.InputShape() {
				return fmt.Errorf(
					"output shape of last layer does not match input shape of new layer: (%d, %d, %d) != (%d, %d, %d)",
					lastLayer.OutputShape().Rows, lastLayer.OutputShape().Cols, lastLayer.OutputShape().Frames,
					layer.InputShape().Rows, layer.InputShape().Cols, layer.InputShape().Frames,
				)
			}
		}
		nn.layers = append(nn.layers, layer)
	}
	return nil
}

// Predict generates a prediction for a certain set of inputs.
func (nn *NeuralNetwork) Predict(inputs [][][]float32) ([][][]float32, error) {
	outputs, err := nn.feedForward(inputs)
	if err != nil {
		return nil, err
	}
	return outputs.Copy().GetAll(), nil
}

// Train takes a set of inputs and their respective targets, and adjusts the layers to produce the
// given outputs through supervised learning.
func (nn *NeuralNetwork) Train(inputs [][][]float32, targets [][][]float32, learningRate float32, momentum float32) error {
	outputs, err := nn.feedForward(inputs)
	if err != nil {
		return err
	}
	deltas := NewValueTensor3D(targets)
	err = deltas.SubtractTensor(outputs)
	if err != nil {
		return err
	}
	return nn.backPropagate(deltas, learningRate, momentum)
}

func (nn *NeuralNetwork) feedForward(inputs [][][]float32) (*Tensor, error) {
	nextInputs := NewValueTensor3D(inputs)
	var err error
	for _, layer := range nn.layers {
		nextInputs, err = layer.FeedForward(nextInputs)
		if err != nil {
			return nil, err
		}
	}
	return nextInputs, nil
}

func (nn *NeuralNetwork) backPropagate(deltas *Tensor, learningRate float32, momentum float32) error {
	nextDeltas := deltas
	var err error
	for i := len(nn.layers) - 1; i >= 0; i-- {
		layer := nn.layers[i]
		nextDeltas, err = layer.BackPropagate(nextDeltas, learningRate, momentum)
		if err != nil {
			return err
		}
	}
	return nil
}

// SaveToFile saves a neural network to a file.
func (nn *NeuralNetwork) SaveToFile(fileName string) error {
	file, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer file.Close()
	nnData := struct {
		Layers []Layer `json:"layers"`
	}{
		Layers: nn.layers,
	}
	return json.NewEncoder(file).Encode(nnData)
}

// LoadFromFile loads a neural network from a file.
func (nn *NeuralNetwork) LoadFromFile(fileName string) error {
	file, err := os.Open(fileName)
	if err != nil {
		return err
	}
	defer file.Close()
	nnData := struct {
		Layers []map[string]interface{} `json:"layers"`
	}{}
	err = json.NewDecoder(file).Decode(&nnData)
	if err != nil {
		return err
	}
	for _, layerData := range nnData.Layers {
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
		err = nn.Add(layer)
		if err != nil {
			return err
		}
	}
	return nil
}
