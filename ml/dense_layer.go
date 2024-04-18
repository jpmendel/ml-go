package ml

import (
	"encoding/json"
	"fmt"
)

// DenseLayer is a fully connected layer for a neural network.
type DenseLayer struct {
	inputShape  LayerShape
	outputShape LayerShape
	inputs      *Tensor
	outputs     *Tensor
	Weights     *Tensor
	Bias        *Tensor
	PrevUpdate  *Tensor
	Activation  ActivationFunction
}

// NewDenseLayer creates a new instance of a fully connected layer.
func NewDenseLayer(inputSize int, outputSize int, activation ActivationFunction) *DenseLayer {
	inputs := NewEmptyTensor1D(inputSize)
	outputs := NewEmptyTensor1D(outputSize)
	weights := NewEmptyTensor2D(inputSize, outputSize)
	weights.SetRandom(-1.0, 1.0)
	bias := NewEmptyTensor1D(outputSize)
	bias.SetRandom(-1.0, 1.0)
	prevUpdate := NewEmptyTensor2D(inputSize, outputSize)
	return &DenseLayer{
		inputShape:  LayerShape{1, inputSize, 1},
		outputShape: LayerShape{1, outputSize, 1},
		inputs:      inputs,
		outputs:     outputs,
		Weights:     weights,
		Bias:        bias,
		PrevUpdate:  prevUpdate,
		Activation:  activation,
	}
}

// Copy creates a deep copy of the layer.
func (l *DenseLayer) Copy() Layer {
	newLayer := NewDenseLayer(l.InputShape().Cols, l.OutputShape().Cols, l.Activation)
	newLayer.Weights.SetTensor(l.Weights)
	newLayer.Bias.SetTensor(l.Bias)
	return newLayer
}

// InputShape returns the rows, columns and frames of the inputs to the layer.
func (l *DenseLayer) InputShape() LayerShape {
	return l.inputShape
}

// OutputShape returns the rows, columns and frames of outputs from the layer.
func (l *DenseLayer) OutputShape() LayerShape {
	return l.outputShape
}

// FeedForward computes the outputs of the layer based on the inputs, weights and bias.
func (l *DenseLayer) FeedForward(inputs *Tensor) (*Tensor, error) {
	if inputs.Frames != 1 {
		return nil, fmt.Errorf("input shape must have frame length of 1, is: %d", inputs.Frames)
	}
	l.inputs.SetTensor(inputs)
	_, err := MatrixMultiply(l.inputs, l.Weights, l.outputs)
	if err != nil {
		return nil, err
	}
	err = l.outputs.AddTensor(l.Bias)
	if err != nil {
		return nil, err
	}
	l.Activation.Function(l.outputs)
	return l.outputs, nil
}

// BackPropagate updates the weights and bias of the layer based on a set of deltas and a learning rate.
func (l *DenseLayer) BackPropagate(outputs *Tensor, learningRate float32, momentum float32) (*Tensor, error) {
	if outputs.Frames != 1 {
		return nil, fmt.Errorf("input shape must have frame length of 1, is: %d", outputs.Frames)
	}
	gradient := l.Activation.Derivative(l.outputs.Copy())
	err := gradient.ScaleTensor(outputs)
	if err != nil {
		return nil, err
	}
	gradient.Scale(learningRate)
	transposedInputs, _ := MatrixTranspose(l.inputs, nil)
	weightChange, err := MatrixMultiply(transposedInputs, gradient, nil)
	if err != nil {
		return nil, err
	}
	err = l.Weights.AddTensor(weightChange)
	if err != nil {
		return nil, err
	}
	l.PrevUpdate.Scale(momentum)
	err = l.Weights.AddTensor(l.PrevUpdate)
	if err != nil {
		return nil, err
	}
	err = l.PrevUpdate.SetTensor(weightChange)
	if err != nil {
		return nil, err
	}
	err = l.Bias.AddTensor(gradient)
	if err != nil {
		return nil, err
	}
	transposedWeights, _ := MatrixTranspose(l.Weights, nil)
	nextDeltas, err := MatrixMultiply(outputs, transposedWeights, nil)
	if err != nil {
		return nil, err
	}
	return nextDeltas, nil
}

// DenseLayerData represents a serialized layer that can be saved to a file.
type DenseLayerData struct {
	Type       LayerType      `json:"type"`
	InputSize  int            `json:"inputSize"`
	OutputSize int            `json:"outputSize"`
	Weights    [][]float32    `json:"weights"`
	Bias       []float32      `json:"bias"`
	Activation ActivationType `json:"activation"`
}

// MarshalJSON converts the layer to JSON.
func (l *DenseLayer) MarshalJSON() ([]byte, error) {
	data := DenseLayerData{
		Type:       LayerTypeDense,
		InputSize:  l.InputShape().Cols,
		OutputSize: l.OutputShape().Cols,
		Weights:    l.Weights.GetFrame(0),
		Bias:       l.Bias.GetFrame(0)[0],
		Activation: l.Activation.Type,
	}
	return json.Marshal(data)
}

// UnmarshalJSON creates a new layer from JSON.
func (l *DenseLayer) UnmarshalJSON(b []byte) error {
	data := DenseLayerData{}
	err := json.Unmarshal(b, &data)
	if err != nil {
		return err
	}
	l.inputs = NewEmptyTensor1D(data.InputSize)
	l.outputs = NewEmptyTensor1D(data.OutputSize)
	l.Weights = NewValueTensor2D(data.Weights)
	l.Bias = NewValueTensor1D(data.Bias)
	l.Activation = activationFunctionOfType(data.Activation)
	l.inputShape = LayerShape{1, data.InputSize, 1}
	l.outputShape = LayerShape{1, data.OutputSize, 1}
	return nil
}
