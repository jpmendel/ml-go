package nn

import (
	"encoding/json"
	"fmt"

	tsr "../tensor"
)

// DenseLayer is a fully connected layer for a neural network.
type DenseLayer struct {
	inputShape  LayerShape
	outputShape LayerShape
	inputs      *tsr.Tensor
	outputs     *tsr.Tensor
	Weights     *tsr.Tensor
	Bias        *tsr.Tensor
	PrevUpdate  *tsr.Tensor
	Activation  ActivationFunction
}

// NewDenseLayer creates a new instance of a fully connected layer.
func NewDenseLayer(inputSize int, outputSize int, activation ActivationFunction) *DenseLayer {
	inputs := tsr.NewEmptyTensor1D(inputSize)
	outputs := tsr.NewEmptyTensor1D(outputSize)
	weights := tsr.NewEmptyTensor2D(inputSize, outputSize)
	weights.SetRandom(-1.0, 1.0)
	bias := tsr.NewEmptyTensor1D(outputSize)
	bias.SetRandom(-1.0, 1.0)
	prevUpdate := tsr.NewEmptyTensor2D(inputSize, outputSize)
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
func (layer *DenseLayer) Copy() Layer {
	newLayer := NewDenseLayer(layer.InputShape().Cols, layer.OutputShape().Cols, layer.Activation)
	newLayer.Weights.SetAll(layer.Weights)
	newLayer.Bias.SetAll(layer.Bias)
	return newLayer
}

// InputShape returns the rows, columns and frames of the inputs to the layer.
func (layer *DenseLayer) InputShape() LayerShape {
	return layer.inputShape
}

// OutputShape returns the rows, columns and frames of outputs from the layer.
func (layer *DenseLayer) OutputShape() LayerShape {
	return layer.outputShape
}

// FeedForward computes the outputs of the layer based on the inputs, weights and bias.
func (layer *DenseLayer) FeedForward(inputs *tsr.Tensor) (*tsr.Tensor, error) {
	if inputs.Frames != 1 {
		return nil, fmt.Errorf("Input shape must have frame length of 1, is: %d", inputs.Frames)
	}
	layer.inputs.SetAll(inputs)
	_, err := tsr.MatrixMultiply(layer.inputs, layer.Weights, layer.outputs)
	if err != nil {
		return nil, err
	}
	err = layer.outputs.AddTensor(layer.Bias)
	if err != nil {
		return nil, err
	}
	layer.Activation.Function(layer.outputs)
	return layer.outputs, nil
}

// BackPropagate updates the weights and bias of the layer based on a set of deltas and a learning rate.
func (layer *DenseLayer) BackPropagate(outputs *tsr.Tensor, learningRate float32, momentum float32) (*tsr.Tensor, error) {
	if outputs.Frames != 1 {
		return nil, fmt.Errorf("Input shape must have frame length of 1, is: %d", outputs.Frames)
	}
	gradient := layer.Activation.Derivative(layer.outputs.Copy())
	err := gradient.ScaleTensor(outputs)
	if err != nil {
		return nil, err
	}
	gradient.Scale(learningRate)
	transposedInputs, _ := tsr.MatrixTranspose(layer.inputs, nil)
	weightChange, err := tsr.MatrixMultiply(transposedInputs, gradient, nil)
	if err != nil {
		return nil, err
	}
	err = layer.Weights.AddTensor(weightChange)
	if err != nil {
		return nil, err
	}
	layer.PrevUpdate.Scale(momentum)
	err = layer.Weights.AddTensor(layer.PrevUpdate)
	if err != nil {
		return nil, err
	}
	err = layer.PrevUpdate.SetAll(weightChange)
	if err != nil {
		return nil, err
	}
	err = layer.Bias.AddTensor(gradient)
	if err != nil {
		return nil, err
	}
	transposedWeights, _ := tsr.MatrixTranspose(layer.Weights, nil)
	nextDeltas, err := tsr.MatrixMultiply(outputs, transposedWeights, nil)
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
func (layer *DenseLayer) MarshalJSON() ([]byte, error) {
	data := DenseLayerData{
		Type:       LayerTypeDense,
		InputSize:  layer.InputShape().Cols,
		OutputSize: layer.OutputShape().Cols,
		Weights:    layer.Weights.GetFrame(0),
		Bias:       layer.Bias.GetFrame(0)[0],
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
	layer.inputs = tsr.NewEmptyTensor1D(data.InputSize)
	layer.outputs = tsr.NewEmptyTensor1D(data.OutputSize)
	layer.Weights = tsr.NewValueTensor2D(data.Weights)
	layer.Bias = tsr.NewValueTensor1D(data.Bias)
	layer.Activation = activationFunctionOfType(data.Activation)
	layer.inputShape = LayerShape{1, data.InputSize, 1}
	layer.outputShape = LayerShape{1, data.OutputSize, 1}
	return nil
}
