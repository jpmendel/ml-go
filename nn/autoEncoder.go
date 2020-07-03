package nn

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"

	tsr "../tensor"
)

// AutoEncoder is a neural network that trains against its own inputs to encode features.
type AutoEncoder struct {
	inputSize int
	layers    []*DenseLayer
	isClosed  bool
}

// NewAutoEncoder Creates a new instance of an AutoEncoder.
func NewAutoEncoder(inputSize int) *AutoEncoder {
	return &AutoEncoder{
		inputSize: inputSize,
		layers:    []*DenseLayer{},
		isClosed:  false,
	}
}

// Copy creates a deep copy of the auto encoder.
func (autoEncoder *AutoEncoder) Copy() *AutoEncoder {
	newAutoEncoder := NewAutoEncoder(autoEncoder.inputSize)
	for i, layer := range autoEncoder.layers {
		if i == len(autoEncoder.layers)-1 {
			newAutoEncoder.AddDecodingLayer(layer.Activation)
		} else {
			newAutoEncoder.AddCodingLayer(layer.OutputShape().Cols, layer.Activation)
		}
	}
	return newAutoEncoder
}

// LayerCount returns the number of layers in the neural network.
func (autoEncoder *AutoEncoder) LayerCount() int {
	return len(autoEncoder.layers)
}

// LayerAt gets a layer at a certain index.
func (autoEncoder *AutoEncoder) LayerAt(index int) Layer {
	if index < 0 || index >= len(autoEncoder.layers) {
		return nil
	}
	return autoEncoder.layers[index]
}

// AddCodingLayer adds an intermediate layer of features to the auto encoder.
func (autoEncoder *AutoEncoder) AddCodingLayer(coded int, activation ActivationFunction) error {
	if autoEncoder.isClosed {
		return fmt.Errorf("The auto encoder has already been closed with a decoding layer")
	}
	var inputSize int
	if len(autoEncoder.layers) > 0 {
		inputSize = autoEncoder.layers[len(autoEncoder.layers)-1].OutputShape().Cols
	} else {
		inputSize = autoEncoder.inputSize
	}
	layer := NewDenseLayer(inputSize, coded, activation)
	autoEncoder.layers = append(autoEncoder.layers, layer)
	return nil
}

// AddDecodingLayer closes the auto encoder, allowing it to train itself. This must be the last layer.
func (autoEncoder *AutoEncoder) AddDecodingLayer(activation ActivationFunction) error {
	if autoEncoder.isClosed {
		return fmt.Errorf("The auto encoder has already been closed with a decoding layer")
	}
	var inputSize int
	if len(autoEncoder.layers) > 0 {
		inputSize = autoEncoder.layers[len(autoEncoder.layers)-1].OutputShape().Cols
	} else {
		inputSize = autoEncoder.inputSize
	}
	layer := NewDenseLayer(inputSize, autoEncoder.inputSize, activation)
	autoEncoder.layers = append(autoEncoder.layers, layer)
	autoEncoder.isClosed = true
	return nil
}

func (autoEncoder *AutoEncoder) isSparse() bool {
	for _, layer := range autoEncoder.layers {
		if layer.OutputShape().Cols >= autoEncoder.inputSize {
			return true
		}
	}
	return false
}

// Predict generates a prediction for a certain set of inputs.
func (autoEncoder *AutoEncoder) Predict(inputs []float32) ([][]float32, error) {
	if !autoEncoder.isClosed {
		return nil, fmt.Errorf("The auto encoder has not been closed with a decoding layer")
	}
	_, err := autoEncoder.feedForward(inputs)
	if err != nil {
		return nil, err
	}
	outputs := make([][]float32, len(autoEncoder.layers)-1)
	for i := 0; i < len(autoEncoder.layers)-1; i++ {
		layerData := autoEncoder.layers[i].outputs
		outputs[i] = make([]float32, layerData.Cols)
		for col := 0; col < layerData.Cols; col++ {
			outputs[i][col] = layerData.Get(0, 0, col)
		}
	}
	return outputs, nil
}

// Train takes a set of inputs and their respective targets, and adjusts the layers to produce the
// given outputs through supervised learning.
func (autoEncoder *AutoEncoder) Train(inputs []float32, learningRate float32, momentum float32) error {
	if !autoEncoder.isClosed {
		return fmt.Errorf("The auto encoder has not been closed with a decoding layer")
	}
	outputs, err := autoEncoder.feedForward(inputs)
	if err != nil {
		return err
	}
	deltas := tsr.NewValueTensor1D(inputs)
	err = deltas.SubtractTensor(outputs)
	if err != nil {
		return err
	}
	return autoEncoder.backPropagate(deltas, learningRate, momentum)
}

func (autoEncoder *AutoEncoder) feedForward(inputs []float32) (*tsr.Tensor, error) {
	nextInputs := tsr.NewValueTensor1D(inputs)
	var err error
	for _, layer := range autoEncoder.layers {
		nextInputs, err = layer.FeedForward(nextInputs)
		if err != nil {
			return nil, err
		}
		if autoEncoder.isSparse() {
			for i := 0; i < nextInputs.Cols/2; i++ {
				col := rand.Intn(nextInputs.Cols)
				nextInputs.Set(0, 0, col, 0.0)
			}
		}
	}
	return nextInputs, nil
}

func (autoEncoder *AutoEncoder) backPropagate(deltas *tsr.Tensor, learningRate float32, momentum float32) error {
	nextDeltas := deltas
	var err error
	for i := len(autoEncoder.layers) - 1; i >= 0; i-- {
		layer := autoEncoder.layers[i]
		nextDeltas, err = layer.BackPropagate(nextDeltas, learningRate, momentum)
		if err != nil {
			return err
		}
	}
	return nil
}

// SaveToFile saves an auto encoder to a file.
func (autoEncoder *AutoEncoder) SaveToFile(fileName string) error {
	file, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer file.Close()
	autoEncoderData := struct {
		Layers []*DenseLayer `json:"layers"`
	}{
		Layers: autoEncoder.layers,
	}
	return json.NewEncoder(file).Encode(autoEncoderData)
}

// LoadFromFile loads an auto encoder from a file.
func (autoEncoder *AutoEncoder) LoadFromFile(fileName string) error {
	file, err := os.Open(fileName)
	if err != nil {
		return err
	}
	defer file.Close()
	autoEncoderData := struct {
		Layers []*DenseLayer `json:"layers"`
	}{}
	err = json.NewDecoder(file).Decode(&autoEncoderData)
	if err != nil {
		return err
	}
	for i, layer := range autoEncoderData.Layers {
		var err error
		if i == len(autoEncoder.layers)-1 {
			err = autoEncoder.AddDecodingLayer(layer.Activation)
		} else {
			err = autoEncoder.AddCodingLayer(layer.OutputShape().Cols, layer.Activation)
		}
		if err != nil {
			return err
		}
	}
	return nil
}
