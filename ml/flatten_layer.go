package ml

import "encoding/json"

// FlattenLayer is a layer that flattens data into a single frame with a single row.
type FlattenLayer struct {
	inputShape  LayerShape
	outputShape LayerShape
	inputs      *Tensor
	outputs     *Tensor
}

// NewFlattenLayer creates a new instance of a flattening layer.
func NewFlattenLayer(inputRows int, inputCols int, inputFrames int) *FlattenLayer {
	inputs := NewEmptyTensor3D(inputFrames, inputRows, inputCols)
	outputSize := inputRows * inputCols * inputFrames
	outputs := NewEmptyTensor1D(outputSize)
	return &FlattenLayer{
		inputShape:  LayerShape{inputRows, inputCols, inputFrames},
		outputShape: LayerShape{1, outputSize, 1},
		inputs:      inputs,
		outputs:     outputs,
	}
}

// Copy creates a deep copy of the layer.
func (l *FlattenLayer) Copy() Layer {
	return NewFlattenLayer(l.InputShape().Rows, l.InputShape().Cols, l.InputShape().Frames)
}

// InputShape returns the rows, columns and frames of the inputs to the layer.
func (l *FlattenLayer) InputShape() LayerShape {
	return l.inputShape
}

// OutputShape returns the rows, columns and frames of outputs from the layer.
func (l *FlattenLayer) OutputShape() LayerShape {
	return l.outputShape
}

// FeedForward flattens the data from its input shape to a shape of 1 row and 1 frame.
func (l *FlattenLayer) FeedForward(inputs *Tensor) (*Tensor, error) {
	flattenedIndex := 0
	l.inputs.SetTensor(inputs)
	for frame := 0; frame < inputs.Frames; frame++ {
		for row := 0; row < inputs.Rows; row++ {
			for col := 0; col < inputs.Cols; col++ {
				l.outputs.Set(0, 0, flattenedIndex, inputs.Get(frame, row, col))
				flattenedIndex++
			}
		}
	}
	return l.outputs, nil
}

// BackPropagate unflattens the data to its original shape.
func (l *FlattenLayer) BackPropagate(outputs *Tensor, learningRate float32, momentum float32) (*Tensor, error) {
	return l.inputs, nil
}

// FlattenLayerData represents a serialized layer that can be saved to a file.
type FlattenLayerData struct {
	Type        LayerType `json:"type"`
	InputRows   int       `json:"inputRows"`
	InputCols   int       `json:"inputCols"`
	InputFrames int       `json:"inputFrames"`
}

// MarshalJSON converts the layer to JSON.
func (l *FlattenLayer) MarshalJSON() ([]byte, error) {
	data := FlattenLayerData{
		Type:        LayerTypeFlatten,
		InputRows:   l.InputShape().Rows,
		InputCols:   l.InputShape().Cols,
		InputFrames: l.InputShape().Frames,
	}
	return json.Marshal(data)
}

// UnmarshalJSON creates a new layer from JSON.
func (l *FlattenLayer) UnmarshalJSON(b []byte) error {
	data := FlattenLayerData{}
	err := json.Unmarshal(b, &data)
	if err != nil {
		return err
	}
	l.inputs = NewEmptyTensor3D(data.InputFrames, data.InputRows, data.InputCols)
	outputSize := data.InputRows * data.InputCols * data.InputFrames
	l.outputs = NewEmptyTensor1D(outputSize)
	l.inputShape = LayerShape{data.InputRows, data.InputCols, data.InputFrames}
	l.outputShape = LayerShape{1, outputSize, 1}
	return nil
}
