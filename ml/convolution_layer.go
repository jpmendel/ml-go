package ml

import "encoding/json"

// ConvolutionLayer is a layer that performs convolutional filters on data.
type ConvolutionLayer struct {
	inputShape  LayerShape
	outputShape LayerShape
	inputs      *Tensor
	outputs     *Tensor
	Filters     []*Tensor
	Activation  ActivationFunction
}

// NewConvolutionLayer creates a new instance of a convolutional layer.
func NewConvolutionLayer(inputRows int, inputCols int, inputFrames int, filters []*Tensor, activation ActivationFunction) *ConvolutionLayer {
	inputs := NewEmptyTensor3D(inputFrames, inputRows, inputCols)
	outputFrames := inputFrames * len(filters)
	outputs := NewEmptyTensor3D(outputFrames, inputRows, inputCols)
	return &ConvolutionLayer{
		inputShape:  LayerShape{inputRows, inputCols, inputFrames},
		outputShape: LayerShape{inputRows, inputCols, outputFrames},
		inputs:      inputs,
		outputs:     outputs,
		Filters:     filters,
		Activation:  activation,
	}
}

// Copy creates a deep copy of the layer.
func (l *ConvolutionLayer) Copy() Layer {
	return NewConvolutionLayer(
		l.InputShape().Rows,
		l.InputShape().Cols,
		l.InputShape().Frames,
		l.Filters,
		l.Activation,
	)
}

// InputShape returns the rows, columns and frames of the inputs to the layer.
func (l *ConvolutionLayer) InputShape() LayerShape {
	return l.inputShape
}

// OutputShape returns the rows, columns and frames of outputs from the layer.
func (l *ConvolutionLayer) OutputShape() LayerShape {
	return l.outputShape
}

// FeedForward applies convolutions to the input for each of the filters.
func (l *ConvolutionLayer) FeedForward(inputs *Tensor) (*Tensor, error) {
	l.inputs.SetTensor(inputs)
	for _, filter := range l.Filters {
		for frame := 0; frame < inputs.Frames; frame++ {
			for row := 0; row < inputs.Rows; row++ {
				for col := 0; col < inputs.Cols; col++ {
					value := l.convolution(inputs, frame, row, col, filter)
					l.outputs.Set(frame, row, col, value)
				}
			}
		}
	}
	l.Activation.Function(l.outputs)
	return l.outputs, nil
}

// BackPropagate does not operate on the data in a convolution layer.
func (l *ConvolutionLayer) BackPropagate(outputs *Tensor, learningRate float32, momentum float32) (*Tensor, error) {
	return l.inputs, nil
}

func (l *ConvolutionLayer) convolution(matrix *Tensor, frame int, row int, col int, filter *Tensor) float32 {
	sum := float32(0.0)
	for or := -filter.Rows / 2; or <= filter.Rows/2; or++ {
		convRow := row + or
		if convRow < 0 || convRow >= matrix.Rows {
			continue
		}
		for oc := -filter.Cols / 2; oc <= filter.Cols/2; oc++ {
			convCol := col + oc
			if convCol < 0 || convCol >= matrix.Cols {
				continue
			}
			sum += matrix.Get(frame, convRow, convCol) * filter.Get(0, or+filter.Rows/2, oc+filter.Cols/2)
		}
	}
	return sum
}

// ConvolutionLayerData represents a serialized layer that can be saved to a file.
type ConvolutionLayerData struct {
	Type        LayerType      `json:"type"`
	InputRows   int            `json:"inputRows"`
	InputCols   int            `json:"inputCols"`
	InputFrames int            `json:"inputFrames"`
	Filters     [][][]float32  `json:"filters"`
	Activation  ActivationType `json:"activation"`
}

// MarshalJSON converts the layer to JSON.
func (l *ConvolutionLayer) MarshalJSON() ([]byte, error) {
	filters := make([][][]float32, len(l.Filters))
	for i, filter := range l.Filters {
		filters[i] = filter.GetFrame(0)
	}
	data := ConvolutionLayerData{
		Type:        LayerTypeConvolution,
		InputRows:   l.InputShape().Rows,
		InputCols:   l.InputShape().Cols,
		InputFrames: l.InputShape().Frames,
		Filters:     filters,
		Activation:  l.Activation.Type,
	}
	return json.Marshal(data)
}

// UnmarshalJSON creates a new layer from JSON.
func (l *ConvolutionLayer) UnmarshalJSON(b []byte) error {
	data := ConvolutionLayerData{}
	err := json.Unmarshal(b, &data)
	if err != nil {
		return err
	}
	l.inputs = NewEmptyTensor3D(data.InputFrames, data.InputRows, data.InputCols)
	outputFrames := data.InputFrames * len(data.Filters)
	l.outputs = NewEmptyTensor3D(data.InputFrames, data.InputRows, data.InputCols)
	l.Filters = make([]*Tensor, len(data.Filters))
	for i, filter := range data.Filters {
		l.Filters[i] = NewValueTensor2D(filter)
	}
	l.Activation = activationFunctionOfType(data.Activation)
	l.inputShape = LayerShape{data.InputRows, data.InputCols, data.InputFrames}
	l.outputShape = LayerShape{data.InputRows, data.InputCols, outputFrames}
	return nil
}
