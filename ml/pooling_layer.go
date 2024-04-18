package ml

import "encoding/json"

// PoolingLayer is a layer that pools data into a smaller form.
type PoolingLayer struct {
	inputShape  LayerShape
	outputShape LayerShape
	inputs      *Tensor
	outputs     *Tensor
	PoolSize    int
	Pooling     PoolingFunction
}

// NewPoolingLayer creates a new instance of a pooling layer.
func NewPoolingLayer(inputRows int, inputCols int, inputFrames int, poolSize int, pooling PoolingFunction) *PoolingLayer {
	inputs := NewEmptyTensor3D(inputFrames, inputRows, inputCols)
	outputRows := inputRows / poolSize
	outputCols := inputCols / poolSize
	outputs := NewEmptyTensor3D(inputFrames, outputRows, outputCols)
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
func (l *PoolingLayer) Copy() Layer {
	return NewPoolingLayer(l.InputShape().Rows, l.InputShape().Cols, l.InputShape().Frames, l.PoolSize, l.Pooling)
}

// InputShape returns the rows, columns and frames of the inputs to the layer.
func (l *PoolingLayer) InputShape() LayerShape {
	return l.inputShape
}

// OutputShape returns the rows, columns and frames of outputs from the layer.
func (l *PoolingLayer) OutputShape() LayerShape {
	return l.outputShape
}

// FeedForward reduces the input data by the pool size.
func (l *PoolingLayer) FeedForward(inputs *Tensor) (*Tensor, error) {
	l.inputs.SetTensor(inputs)
	for frame := 0; frame < l.outputs.Frames; frame++ {
		for row := 0; row < l.outputs.Rows; row++ {
			poolRow := row * l.PoolSize
			for col := 0; col < l.outputs.Cols; col++ {
				poolCol := col * l.PoolSize
				pooledValue := l.Pooling.FindPooledValue(inputs, frame, poolRow, poolCol, l.PoolSize)
				l.outputs.Set(frame, row, col, pooledValue)
			}
		}
	}
	return l.outputs, nil
}

// BackPropagate does not operate on the data in a pooling layer.
func (l *PoolingLayer) BackPropagate(outputs *Tensor, learningRate float32, momentum float32) (*Tensor, error) {
	return l.inputs, nil
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
func (l *PoolingLayer) MarshalJSON() ([]byte, error) {
	data := PoolingLayerData{
		Type:        LayerTypePooling,
		InputRows:   l.InputShape().Rows,
		InputCols:   l.InputShape().Cols,
		InputFrames: l.InputShape().Frames,
		PoolSize:    l.PoolSize,
		Pooling:     l.Pooling.Method,
	}
	return json.Marshal(data)
}

// UnmarshalJSON creates a new layer from JSON.
func (l *PoolingLayer) UnmarshalJSON(b []byte) error {
	data := PoolingLayerData{}
	err := json.Unmarshal(b, &data)
	if err != nil {
		return err
	}
	l.inputs = NewEmptyTensor3D(data.InputFrames, data.InputRows, data.InputCols)
	outputRows := data.InputRows / data.PoolSize
	outputCols := data.InputCols / data.PoolSize
	l.outputs = NewEmptyTensor3D(data.InputFrames, outputRows, outputCols)
	l.PoolSize = data.PoolSize
	l.Pooling = poolingFunctionOfMethod(data.Pooling)
	l.inputShape = LayerShape{data.InputRows, data.InputCols, data.InputFrames}
	l.outputShape = LayerShape{outputRows, outputCols, data.InputFrames}
	return nil
}
