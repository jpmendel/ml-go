package ml

import (
	"encoding/json"
	"math/rand"
	"os"
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
func (ae *AutoEncoder) Copy() *AutoEncoder {
	new := NewAutoEncoder(ae.inputSize)
	new.encodingLayers = append(new.encodingLayers, ae.encodingLayers...)
	new.decodingLayers = append(new.decodingLayers, ae.decodingLayers...)
	return new
}

// LayerCount returns the number of layers in the neural network.
func (ae *AutoEncoder) LayerCount() int {
	return len(ae.encodingLayers) + len(ae.decodingLayers)
}

// LayerAt gets a layer at a certain index.
func (ae *AutoEncoder) LayerAt(index int) Layer {
	if index < 0 || index >= len(ae.encodingLayers) {
		return nil
	}
	return ae.encodingLayers[index]
}

// AddCodingLayer adds an intermediate layer of features to the auto encoder.
func (ae *AutoEncoder) AddCodingLayer(coded int, activation ActivationFunction) error {
	var inputSize int
	if len(ae.encodingLayers) > 0 {
		inputSize = ae.encodingLayers[len(ae.encodingLayers)-1].OutputShape().Cols
	} else {
		inputSize = ae.inputSize
	}
	encodingLayer := NewDenseLayer(inputSize, coded, activation)
	decodingLayer := NewDenseLayer(coded, inputSize, activation)
	ae.encodingLayers = append(ae.encodingLayers, encodingLayer)
	ae.decodingLayers = append(ae.decodingLayers, nil)
	copy(ae.decodingLayers[1:], ae.decodingLayers)
	ae.decodingLayers[0] = decodingLayer
	return nil
}

func (ae *AutoEncoder) isSparse() bool {
	for _, layer := range ae.encodingLayers {
		if layer.OutputShape().Cols >= ae.inputSize {
			return true
		}
	}
	for _, layer := range ae.decodingLayers {
		if layer.OutputShape().Cols >= ae.inputSize {
			return true
		}
	}
	return false
}

// Encode generates an encoded representation for a certain set of inputs.
func (ae *AutoEncoder) Encode(inputs []float32) ([]float32, error) {
	inputsTensor := NewValueTensor1D(inputs)
	encoded, err := ae.feedForward(inputsTensor, ae.encodingLayers, false)
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
func (ae *AutoEncoder) Decode(coded []float32) ([]float32, error) {
	codedTensor := NewValueTensor1D(coded)
	decoded, err := ae.feedForward(codedTensor, ae.decodingLayers, false)
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
func (ae *AutoEncoder) Train(inputs []float32, learningRate float32, momentum float32) error {
	inputsTensor := NewValueTensor1D(inputs)
	coded, err := ae.feedForward(inputsTensor, ae.encodingLayers, true)
	if err != nil {
		return err
	}
	outputs, err := ae.feedForward(coded, ae.decodingLayers, true)
	if err != nil {
		return err
	}
	deltas := NewValueTensor1D(inputs)
	err = deltas.SubtractTensor(outputs)
	if err != nil {
		return err
	}
	return ae.backPropagate(deltas, learningRate, momentum)
}

func (ae *AutoEncoder) feedForward(inputs *Tensor, layers []*DenseLayer, train bool) (*Tensor, error) {
	nextInputs := inputs
	var err error
	for _, layer := range layers {
		nextInputs, err = layer.FeedForward(nextInputs)
		if err != nil {
			return nil, err
		}
		if train && ae.isSparse() {
			for i := 0; i < nextInputs.Cols/2; i++ {
				col := rand.Intn(nextInputs.Cols)
				nextInputs.Set(0, 0, col, 0.0)
			}
		}
	}
	return nextInputs, nil
}

func (ae *AutoEncoder) backPropagate(deltas *Tensor, learningRate float32, momentum float32) error {
	nextDeltas := deltas
	var err error
	for i := ae.LayerCount() - 1; i >= 0; i-- {
		var layer *DenseLayer
		if i < len(ae.encodingLayers) {
			layer = ae.encodingLayers[i]
		} else {
			layer = ae.decodingLayers[i-len(ae.encodingLayers)]
		}
		nextDeltas, err = layer.BackPropagate(nextDeltas, learningRate, momentum)
		if err != nil {
			return err
		}
	}
	return nil
}

// SaveToFile saves an auto encoder to a file.
func (ae *AutoEncoder) SaveToFile(fileName string) error {
	file, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer file.Close()
	autoEncoderData := struct {
		EncodingLayers []*DenseLayer `json:"encodingLayers"`
		DecodingLayers []*DenseLayer `json:"decodingLayers"`
	}{
		EncodingLayers: ae.encodingLayers,
		DecodingLayers: ae.decodingLayers,
	}
	return json.NewEncoder(file).Encode(autoEncoderData)
}

// LoadFromFile loads an auto encoder from a file.
func (ae *AutoEncoder) LoadFromFile(fileName string) error {
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
	ae.encodingLayers = append(ae.encodingLayers, autoEncoderData.EncodingLayers...)
	ae.decodingLayers = append(ae.decodingLayers, autoEncoderData.DecodingLayers...)
	return nil
}
