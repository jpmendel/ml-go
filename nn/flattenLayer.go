package nn

import (
	"encoding/json"

	tsr "../tensor"
)

// FlattenLayer is a layer that flattens data into length 1.
type FlattenLayer struct {
	inputShape  LayerShape
	outputShape LayerShape
	inputs      *tsr.Tensor
	outputs     *tsr.Tensor
}

// NewFlattenLayer creates a new instance of a flattening layer.
func NewFlattenLayer(inputRows int, inputCols int, inputFrames int) *FlattenLayer {
	inputs := tsr.NewEmptyTensor3D(inputFrames, inputRows, inputCols)
	outputSize := inputRows * inputCols * inputFrames
	outputs := tsr.NewEmptyTensor1D(outputSize)
	return &FlattenLayer{
		inputShape:  LayerShape{inputRows, inputCols, inputFrames},
		outputShape: LayerShape{1, outputSize, 1},
		inputs:      inputs,
		outputs:     outputs,
	}
}

// Copy creates a deep copy of the layer.
func (layer *FlattenLayer) Copy() Layer {
	return NewFlattenLayer(layer.InputShape().Rows, layer.InputShape().Cols, layer.InputShape().Length)
}

// InputShape returns the rows, columns and length of the inputs to the layer.
func (layer *FlattenLayer) InputShape() LayerShape {
	return layer.inputShape
}

// OutputShape returns the rows, columns and length of outputs from the layer.
func (layer *FlattenLayer) OutputShape() LayerShape {
	return layer.outputShape
}

// FeedForward flattens the data from its input length to a length of 1.
func (layer *FlattenLayer) FeedForward(inputs *tsr.Tensor) (*tsr.Tensor, error) {
	flattenedIndex := 0
	layer.inputs.SetAll(inputs)
	for frame := 0; frame < inputs.Frames; frame++ {
		for row := 0; row < inputs.Rows; row++ {
			for col := 0; col < inputs.Cols; col++ {
				layer.outputs.Set(0, 0, flattenedIndex, inputs.Get(frame, row, col))
				flattenedIndex++
			}
		}
	}
	return layer.outputs, nil
}

// BackPropagate unflattens the data to its original length.
func (layer *FlattenLayer) BackPropagate(outputs *tsr.Tensor, learningRate float32, momentum float32) (*tsr.Tensor, error) {
	return layer.inputs, nil
}

// FlattenLayerData represents a serialized layer that can be saved to a file.
type FlattenLayerData struct {
	Type        LayerType `json:"type"`
	InputRows   int       `json:"inputRows"`
	InputCols   int       `json:"inputCols"`
	InputFrames int       `json:"inputFrames"`
}

// MarshalJSON converts the layer to JSON.
func (layer *FlattenLayer) MarshalJSON() ([]byte, error) {
	data := FlattenLayerData{
		Type:        LayerTypeFlatten,
		InputRows:   layer.InputShape().Rows,
		InputCols:   layer.InputShape().Cols,
		InputFrames: layer.InputShape().Length,
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
	layer.inputs = tsr.NewEmptyTensor3D(data.InputFrames, data.InputRows, data.InputCols)
	outputSize := data.InputRows * data.InputCols * data.InputFrames
	layer.outputs = tsr.NewEmptyTensor1D(outputSize)
	layer.inputShape = LayerShape{data.InputRows, data.InputCols, data.InputFrames}
	layer.outputShape = LayerShape{1, outputSize, 1}
	return nil
}
