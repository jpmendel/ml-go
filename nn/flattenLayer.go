package nn

import (
	"encoding/json"

	"../mat"
)

// FlattenLayer is a layer that flattens data into length 1.
type FlattenLayer struct {
	inputs  []*mat.Matrix
	outputs *mat.Matrix
}

// NewFlattenLayer creates a new instance of a flattening layer.
func NewFlattenLayer(inputRows int, inputCols int, inputLength int) *FlattenLayer {
	inputs := make([]*mat.Matrix, inputLength)
	for i := 0; i < inputLength; i++ {
		inputs[i] = mat.NewEmptyMatrix(inputRows, inputCols)
	}
	outputSize := inputRows * inputCols * inputLength
	outputs := mat.NewEmptyMatrix(1, outputSize)
	return &FlattenLayer{
		inputs:  inputs,
		outputs: outputs,
	}
}

// Copy creates a deep copy of the layer.
func (layer *FlattenLayer) Copy() Layer {
	return NewFlattenLayer(layer.InputShape().Rows, layer.InputShape().Cols, layer.InputShape().Length)
}

// InputShape returns the rows, columns and length of the inputs to the layer.
func (layer *FlattenLayer) InputShape() LayerShape {
	return LayerShape{layer.inputs[0].Rows, layer.inputs[0].Cols, len(layer.inputs)}
}

// OutputShape returns the rows, columns and length of outputs from the layer.
func (layer *FlattenLayer) OutputShape() LayerShape {
	return LayerShape{layer.outputs.Rows, layer.outputs.Cols, 1}
}

// FeedForward flattens the data from its input length to a length of 1.
func (layer *FlattenLayer) FeedForward(inputs []*mat.Matrix) ([]*mat.Matrix, error) {
	index := 0
	for i, input := range inputs {
		layer.inputs[i].SetAll(input)
		for row := 0; row < input.Rows; row++ {
			for col := 0; col < input.Cols; col++ {
				layer.outputs.Set(0, index, input.Get(row, col))
				index++
			}
		}
	}
	return []*mat.Matrix{layer.outputs}, nil
}

// BackPropagate unflattens the data to its original length.
func (layer *FlattenLayer) BackPropagate(outputs []*mat.Matrix, learningRate float32, momentum float32) ([]*mat.Matrix, error) {
	return layer.inputs, nil
}

// FlattenLayerData represents a serialized layer that can be saved to a file.
type FlattenLayerData struct {
	Type        LayerType `json:"type"`
	InputSize   int       `json:"inputSize"`
	InputLength int       `json:"inputLength"`
}

// MarshalJSON converts the layer to JSON.
func (layer *FlattenLayer) MarshalJSON() ([]byte, error) {
	data := FlattenLayerData{
		Type:        LayerTypeFlatten,
		InputSize:   layer.InputShape().Cols,
		InputLength: layer.InputShape().Length,
	}
	return json.Marshal(data)
}

// UnmarshalJSON creates a new layer from JSON.
func (layer *FlattenLayer) UnmarshalJSON(b []byte) error {
	data := FlattenLayerData{}
	err := json.Unmarshal(b, &data)
	if err != nil {
		return err
	}
	layer.inputs = make([]*mat.Matrix, data.InputLength)
	for i := 0; i < data.InputLength; i++ {
		layer.inputs[i] = mat.NewEmptyMatrix(1, data.InputSize)
	}
	outputSize := data.InputSize * data.InputLength
	layer.outputs = mat.NewEmptyMatrix(1, outputSize)
	return nil
}
