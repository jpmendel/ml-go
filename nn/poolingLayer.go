package nn

import (
	"encoding/json"

	tsr "github.com/jpmendel/ml-go/tensor"
)

// PoolingLayer is a layer that pools data into a smaller form.
type PoolingLayer struct {
	inputShape  LayerShape
	outputShape LayerShape
	inputs      *tsr.Tensor
	outputs     *tsr.Tensor
	PoolSize    int
	Pooling     PoolingFunction
}

// NewPoolingLayer creates a new instance of a pooling layer.
func NewPoolingLayer(inputRows int, inputCols int, inputFrames int, poolSize int, pooling PoolingFunction) *PoolingLayer {
	inputs := tsr.NewEmptyTensor3D(inputFrames, inputRows, inputCols)
	outputRows := inputRows / poolSize
	outputCols := inputCols / poolSize
	outputs := tsr.NewEmptyTensor3D(inputFrames, outputRows, outputCols)
	return &PoolingLayer{
		inputShape:  LayerShape{inputRows, inputCols, inputFrames},
		outputShape: LayerShape{outputRows, outputCols, inputFrames},
		inputs:      inputs,
		outputs:     outputs,
		PoolSize:    poolSize,
		Pooling:     pooling,
	}
}

// Copy creates a deep copy of the layer.
func (layer *PoolingLayer) Copy() Layer {
	return NewPoolingLayer(layer.InputShape().Rows, layer.InputShape().Cols, layer.InputShape().Frames, layer.PoolSize, layer.Pooling)
}

// InputShape returns the rows, columns and frames of the inputs to the layer.
func (layer *PoolingLayer) InputShape() LayerShape {
	return layer.inputShape
}

// OutputShape returns the rows, columns and frames of outputs from the layer.
func (layer *PoolingLayer) OutputShape() LayerShape {
	return layer.outputShape
}

// FeedForward reduces the input data by the pool size.
func (layer *PoolingLayer) FeedForward(inputs *tsr.Tensor) (*tsr.Tensor, error) {
	layer.inputs.SetTensor(inputs)
	for frame := 0; frame < layer.outputs.Frames; frame++ {
		for row := 0; row < layer.outputs.Rows; row++ {
			poolRow := row * layer.PoolSize
			for col := 0; col < layer.outputs.Cols; col++ {
				poolCol := col * layer.PoolSize
				pooledValue := layer.Pooling.FindPooledValue(inputs, frame, poolRow, poolCol, layer.PoolSize)
				layer.outputs.Set(frame, row, col, pooledValue)
			}
		}
	}
	return layer.outputs, nil
}

// BackPropagate does not operate on the data in a pooling layer.
func (layer *PoolingLayer) BackPropagate(outputs *tsr.Tensor, learningRate float32, momentum float32) (*tsr.Tensor, error) {
	return layer.inputs, nil
}

// PoolingLayerData represents a serialized layer that can be saved to a file.
type PoolingLayerData struct {
	Type        LayerType     `json:"type"`
	InputRows   int           `json:"inputRows"`
	InputCols   int           `json:"inputCols"`
	InputFrames int           `json:"inputFrames"`
	PoolSize    int           `json:"poolSize"`
	Pooling     PoolingMethod `json:"pooling"`
}

// MarshalJSON converts the layer to JSON.
func (layer *PoolingLayer) MarshalJSON() ([]byte, error) {
	data := PoolingLayerData{
		Type:        LayerTypePooling,
		InputRows:   layer.InputShape().Rows,
		InputCols:   layer.InputShape().Cols,
		InputFrames: layer.InputShape().Frames,
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
	layer.inputs = tsr.NewEmptyTensor3D(data.InputFrames, data.InputRows, data.InputCols)
	outputRows := data.InputRows / data.PoolSize
	outputCols := data.InputCols / data.PoolSize
	layer.outputs = tsr.NewEmptyTensor3D(data.InputFrames, outputRows, outputCols)
	layer.PoolSize = data.PoolSize
	layer.Pooling = poolingFunctionOfMethod(data.Pooling)
	layer.inputShape = LayerShape{data.InputRows, data.InputCols, data.InputFrames}
	layer.outputShape = LayerShape{outputRows, outputCols, data.InputFrames}
	return nil
}
