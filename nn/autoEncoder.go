package nn

import (
	"encoding/json"
	"math/rand"
	"os"

	tsr "../tensor"
)

// AutoEncoder is a neural network that trains against its own inputs to encode features.
type AutoEncoder struct {
	inputSize      int
	encodingLayers []*DenseLayer
	decodingLayers []*DenseLayer
}

// NewAutoEncoder Creates a new instance of an AutoEncoder.
func NewAutoEncoder(inputSize int) *AutoEncoder {
	return &AutoEncoder{
		inputSize:      inputSize,
		encodingLayers: []*DenseLayer{},
		decodingLayers: []*DenseLayer{},
	}
}

// Copy creates a deep copy of the auto encoder.
func (autoEncoder *AutoEncoder) Copy() *AutoEncoder {
	newAutoEncoder := NewAutoEncoder(autoEncoder.inputSize)
	for _, layer := range autoEncoder.encodingLayers {
		newAutoEncoder.encodingLayers = append(newAutoEncoder.encodingLayers, layer)
	}
	for _, layer := range autoEncoder.decodingLayers {
		newAutoEncoder.decodingLayers = append(newAutoEncoder.decodingLayers, layer)
	}
	return newAutoEncoder
}

// LayerCount returns the number of layers in the neural network.
func (autoEncoder *AutoEncoder) LayerCount() int {
	return len(autoEncoder.encodingLayers) + len(autoEncoder.decodingLayers)
}

// LayerAt gets a layer at a certain index.
func (autoEncoder *AutoEncoder) LayerAt(index int) Layer {
	if index < 0 || index >= len(autoEncoder.encodingLayers) {
		return nil
	}
	return autoEncoder.encodingLayers[index]
}

// AddCodingLayer adds an intermediate layer of features to the auto encoder.
func (autoEncoder *AutoEncoder) AddCodingLayer(coded int, activation ActivationFunction) error {
	var inputSize int
	if len(autoEncoder.encodingLayers) > 0 {
		inputSize = autoEncoder.encodingLayers[len(autoEncoder.encodingLayers)-1].OutputShape().Cols
	} else {
		inputSize = autoEncoder.inputSize
	}
	encodingLayer := NewDenseLayer(inputSize, coded, activation)
	decodingLayer := NewDenseLayer(coded, inputSize, activation)
	autoEncoder.encodingLayers = append(autoEncoder.encodingLayers, encodingLayer)
	autoEncoder.decodingLayers = append(autoEncoder.decodingLayers, nil)
	copy(autoEncoder.decodingLayers[1:], autoEncoder.decodingLayers)
	autoEncoder.decodingLayers[0] = decodingLayer
	return nil
}

func (autoEncoder *AutoEncoder) isSparse() bool {
	for _, layer := range autoEncoder.encodingLayers {
		if layer.OutputShape().Cols >= autoEncoder.inputSize {
			return true
		}
	}
	for _, layer := range autoEncoder.decodingLayers {
		if layer.OutputShape().Cols >= autoEncoder.inputSize {
			return true
		}
	}
	return false
}

// Encode generates an encoded representation for a certain set of inputs.
func (autoEncoder *AutoEncoder) Encode(inputs []float32) ([]float32, error) {
	inputsTensor := tsr.NewValueTensor1D(inputs)
	encoded, err := autoEncoder.feedForward(inputsTensor, autoEncoder.encodingLayers, false)
	if err != nil {
		return nil, err
	}
	outputs := make([]float32, encoded.Cols)
	for i := 0; i < encoded.Cols; i++ {
		outputs[i] = encoded.Get(0, 0, i)
	}
	return outputs, nil
}

// Decode decodes a coded representation to a set of outputs.
func (autoEncoder *AutoEncoder) Decode(coded []float32) ([]float32, error) {
	codedTensor := tsr.NewValueTensor1D(coded)
	decoded, err := autoEncoder.feedForward(codedTensor, autoEncoder.decodingLayers, false)
	if err != nil {
		return nil, err
	}
	outputs := make([]float32, decoded.Cols)
	for i := 0; i < decoded.Cols; i++ {
		outputs[i] = decoded.Get(0, 0, i)
	}
	return outputs, nil
}

// Train takes a set of inputs and their respective targets, and adjusts the layers to produce the
// given outputs through supervised learning.
func (autoEncoder *AutoEncoder) Train(inputs []float32, learningRate float32, momentum float32) error {
	inputsTensor := tsr.NewValueTensor1D(inputs)
	coded, err := autoEncoder.feedForward(inputsTensor, autoEncoder.encodingLayers, true)
	if err != nil {
		return err
	}
	outputs, err := autoEncoder.feedForward(coded, autoEncoder.decodingLayers, true)
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

func (autoEncoder *AutoEncoder) feedForward(inputs *tsr.Tensor, layers []*DenseLayer, train bool) (*tsr.Tensor, error) {
	nextInputs := inputs
	var err error
	for _, layer := range layers {
		nextInputs, err = layer.FeedForward(nextInputs)
		if err != nil {
			return nil, err
		}
		if train && autoEncoder.isSparse() {
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
	for i := autoEncoder.LayerCount() - 1; i >= 0; i-- {
		var layer *DenseLayer
		if i < len(autoEncoder.encodingLayers) {
			layer = autoEncoder.encodingLayers[i]
		} else {
			layer = autoEncoder.decodingLayers[i-len(autoEncoder.encodingLayers)]
		}
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
		EncodingLayers []*DenseLayer `json:"encodingLayers"`
		DecodingLayers []*DenseLayer `json:"decodingLayers"`
	}{
		EncodingLayers: autoEncoder.encodingLayers,
		DecodingLayers: autoEncoder.decodingLayers,
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
		EncodingLayers []*DenseLayer `json:"encodingLayers"`
		DecodingLayers []*DenseLayer `json:"decodingLayers"`
	}{}
	err = json.NewDecoder(file).Decode(&autoEncoderData)
	if err != nil {
		return err
	}
	for _, layer := range autoEncoderData.EncodingLayers {
		autoEncoder.encodingLayers = append(autoEncoder.encodingLayers, layer)
	}
	for _, layer := range autoEncoderData.DecodingLayers {
		autoEncoder.decodingLayers = append(autoEncoder.decodingLayers, layer)
	}
	return nil
}
