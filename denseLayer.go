package ml

import (
	"encoding/json"
	"fmt"
)

// DenseLayer is a fully connected layer for a neural network.
type DenseLayer struct {
	inputs     *Matrix
	outputs    *Matrix
	Weights    *Matrix
	Bias       *Matrix
	PrevUpdate *Matrix
	Activation ActivationFunction
}

// NewDenseLayer creates a new instance of a fully connected layer.
func NewDenseLayer(inputSize int, outputSize int, activation ActivationFunction) *DenseLayer {
	inputs := NewEmptyMatrix(1, inputSize)
	outputs := NewEmptyMatrix(1, outputSize)
	weights := NewEmptyMatrix(inputSize, outputSize)
	weights.SetRandom(-1.0, 1.0)
	bias := NewEmptyMatrix(1, outputSize)
	bias.SetRandom(-1.0, 1.0)
	prevUpdate := NewEmptyMatrix(inputSize, outputSize)
	return &DenseLayer{
		inputs:     inputs,
		outputs:    outputs,
		Weights:    weights,
		Bias:       bias,
		PrevUpdate: prevUpdate,
		Activation: activation,
	}
}

// Copy creates a deep copy of the layer.
func (layer *DenseLayer) Copy() Layer {
	newLayer := NewDenseLayer(layer.InputShape().Cols, layer.OutputShape().Cols, layer.Activation)
	newLayer.Weights.SetAll(layer.Weights)
	newLayer.Bias.SetAll(layer.Bias)
	return newLayer
}

// InputShape returns the rows, columns and length of the inputs to the layer.
func (layer *DenseLayer) InputShape() LayerShape {
	return LayerShape{layer.inputs.Rows, layer.inputs.Cols, 1}
}

// OutputShape returns the rows, columns and length of outputs from the layer.
func (layer *DenseLayer) OutputShape() LayerShape {
	return LayerShape{layer.outputs.Rows, layer.outputs.Cols, 1}
}

// FeedForward computes the outputs of the layer based on the inputs, weights and bias.
func (layer *DenseLayer) FeedForward(inputs []*Matrix) ([]*Matrix, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Input shape must have length of 1, is: %d", len(inputs))
	}
	layer.inputs.SetAll(inputs[0])
	_, err := MatrixMultiply(layer.inputs, layer.Weights, layer.outputs)
	if err != nil {
		return nil, err
	}
	err = layer.outputs.AddMatrix(layer.Bias)
	if err != nil {
		return nil, err
	}
	layer.outputs = layer.Activation.Function(layer.outputs)
	inputs[0] = layer.outputs
	return inputs, nil
}

// BackPropagate updates the weights and bias of the layer based on a set of deltas and a learning rate.
func (layer *DenseLayer) BackPropagate(outputs []*Matrix, learningRate float32, momentum float32) ([]*Matrix, error) {
	if len(outputs) != 1 {
		return nil, fmt.Errorf("Input shape must have length of 1, is: %d", len(outputs))
	}
	deltasMatrix := outputs[0]
	gradient := layer.Activation.Derivative(layer.outputs.Copy())
	err := gradient.ScaleMatrix(deltasMatrix)
	if err != nil {
		return nil, err
	}
	gradient.Scale(learningRate)
	transposedInputs, _ := MatrixTranspose(layer.inputs, nil)
	weightChange, err := MatrixMultiply(transposedInputs, gradient, nil)
	if err != nil {
		return nil, err
	}
	err = layer.Weights.AddMatrix(weightChange)
	if err != nil {
		return nil, err
	}
	layer.PrevUpdate.Scale(momentum)
	err = layer.Weights.AddMatrix(layer.PrevUpdate)
	if err != nil {
		return nil, err
	}
	err = layer.PrevUpdate.SetAll(weightChange)
	if err != nil {
		return nil, err
	}
	err = layer.Bias.AddMatrix(gradient)
	if err != nil {
		return nil, err
	}
	transposedWeights, _ := MatrixTranspose(layer.Weights, nil)
	nextDeltas, err := MatrixMultiply(deltasMatrix, transposedWeights, nil)
	if err != nil {
		return nil, err
	}
	outputs[0] = nextDeltas
	return outputs, nil
}

// DenseLayerData represents a serialized layer that can be saved to a file.
type DenseLayerData struct {
	Type       LayerType      `json:"type"`
	InputSize  int            `json:"inputSize"`
	OutputSize int            `json:"outputSize"`
	Weights    [][]float32    `json:"weights"`
	Bias       [][]float32    `json:"bias"`
	Activation ActivationType `json:"activation"`
}

// MarshalJSON converts the layer to JSON.
func (layer *DenseLayer) MarshalJSON() ([]byte, error) {
	data := DenseLayerData{
		Type:       LayerTypeDense,
		InputSize:  layer.InputShape().Cols,
		OutputSize: layer.OutputShape().Cols,
		Weights:    layer.Weights.values,
		Bias:       layer.Bias.values,
		Activation: layer.Activation.Type,
	}
	return json.Marshal(data)
}

// UnmarshalJSON creates a new layer from JSON.
func (layer *DenseLayer) UnmarshalJSON(b []byte) error {
	data := DenseLayerData{}
	err := json.Unmarshal(b, &data)
	if err != nil {
		return err
	}
	layer.inputs = NewEmptyMatrix(1, data.InputSize)
	layer.outputs = NewEmptyMatrix(1, data.OutputSize)
	layer.Weights = NewMatrixWithValues(data.Weights)
	layer.Bias = NewMatrixWithValues(data.Bias)
	layer.Activation = activationFunctionOfType(data.Activation)
	return nil
}
