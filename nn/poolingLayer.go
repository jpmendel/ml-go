package nn

import (
	"encoding/json"

	"../mat"
)

// PoolingLayer is a layer that pools data into a smaller form.
type PoolingLayer struct {
	inputShape  LayerShape
	outputShape LayerShape
	inputs      []*mat.Matrix
	outputs     []*mat.Matrix
	PoolSize    int
	Pooling     PoolingFunction
}

// NewPoolingLayer creates a new instance of a pooling layer.
func NewPoolingLayer(inputRows int, inputCols int, inputLength int, poolSize int, pooling PoolingFunction) *PoolingLayer {
	inputs := make([]*mat.Matrix, inputLength)
	for i := 0; i < inputLength; i++ {
		inputs[i] = mat.NewEmptyMatrix(inputRows, inputCols)
	}
	outputRows := inputRows / poolSize
	outputCols := inputCols / poolSize
	outputs := make([]*mat.Matrix, inputLength)
	for i := 0; i < inputLength; i++ {
		outputs[i] = mat.NewEmptyMatrix(outputRows, outputCols)
	}
	return &PoolingLayer{
		inputShape:  LayerShape{inputRows, inputCols, inputLength},
		outputShape: LayerShape{outputRows, outputCols, inputLength},
		inputs:      inputs,
		outputs:     outputs,
		PoolSize:    poolSize,
		Pooling:     pooling,
	}
}

// Copy creates a deep copy of the layer.
func (layer *PoolingLayer) Copy() Layer {
	return NewPoolingLayer(layer.InputShape().Rows, layer.InputShape().Cols, layer.InputShape().Length, layer.PoolSize, layer.Pooling)
}

// InputShape returns the rows, columns and length of the inputs to the layer.
func (layer *PoolingLayer) InputShape() LayerShape {
	return layer.inputShape
}

// OutputShape returns the rows, columns and length of outputs from the layer.
func (layer *PoolingLayer) OutputShape() LayerShape {
	return layer.outputShape
}

// FeedForward reduces the input data by the pool size.
func (layer *PoolingLayer) FeedForward(inputs []*mat.Matrix) ([]*mat.Matrix, error) {
	for i, input := range inputs {
		layer.inputs[i].SetAll(input)
		for row := 0; row < layer.outputs[i].Rows; row++ {
			poolRow := row * layer.PoolSize
			for col := 0; col < layer.outputs[i].Cols; col++ {
				poolCol := col * layer.PoolSize
				pooledValue := layer.Pooling.FindPooledValue(input, poolRow, poolCol, layer.PoolSize)
				layer.outputs[i].Set(row, col, pooledValue)
			}
		}
	}
	return layer.outputs, nil
}

// BackPropagate does not operate on the data in a pooling layer.
func (layer *PoolingLayer) BackPropagate(outputs []*mat.Matrix, learningRate float32, momentum float32) ([]*mat.Matrix, error) {
	return layer.inputs, nil
}

// PoolingLayerData represents a serialized layer that can be saved to a file.
type PoolingLayerData struct {
	Type        LayerType     `json:"type"`
	InputRows   int           `json:"inputRows"`
	InputCols   int           `json:"inputCols"`
	InputLength int           `json:"inputLength"`
	PoolSize    int           `json:"poolSize"`
	Pooling     PoolingMethod `json:"pooling"`
}

// MarshalJSON converts the layer to JSON.
func (layer *PoolingLayer) MarshalJSON() ([]byte, error) {
	data := PoolingLayerData{
		Type:        LayerTypePooling,
		InputRows:   layer.InputShape().Rows,
		InputCols:   layer.InputShape().Cols,
		InputLength: layer.InputShape().Length,
		PoolSize:    layer.PoolSize,
		Pooling:     layer.Pooling.Method,
	}
	return json.Marshal(data)
}

// UnmarshalJSON creates a new layer from JSON.
func (layer *PoolingLayer) UnmarshalJSON(b []byte) error {
	data := PoolingLayerData{}
	err := json.Unmarshal(b, &data)
	if err != nil {
		return err
	}
	layer.inputs = make([]*mat.Matrix, data.InputLength)
	for i := 0; i < data.InputLength; i++ {
		layer.inputs[i] = mat.NewEmptyMatrix(data.InputRows, data.InputCols)
	}
	outputRows := data.InputRows / data.PoolSize
	outputCols := data.InputCols / data.PoolSize
	layer.outputs = make([]*mat.Matrix, data.InputLength)
	for i := 0; i < data.InputLength; i++ {
		layer.outputs[i] = mat.NewEmptyMatrix(outputRows, outputCols)
	}
	layer.PoolSize = data.PoolSize
	layer.Pooling = poolingFunctionOfMethod(data.Pooling)
	layer.inputShape = LayerShape{data.InputRows, data.InputCols, data.InputLength}
	layer.outputShape = LayerShape{outputRows, outputCols, data.InputLength}
	return nil
}
