package nn

import (
	"encoding/json"

	tsr "../tensor"
)

// ConvolutionLayer is a layer that performs convolutional filters on data.
type ConvolutionLayer struct {
	inputShape  LayerShape
	outputShape LayerShape
	inputs      *tsr.Tensor
	outputs     *tsr.Tensor
	Filters     []*tsr.Tensor
	Activation  ActivationFunction
}

// NewConvolutionLayer creates a new instance of a convolutional layer.
func NewConvolutionLayer(inputRows int, inputCols int, inputFrames int, filters []*tsr.Tensor, activation ActivationFunction) *ConvolutionLayer {
	inputs := tsr.NewEmptyTensor3D(inputFrames, inputRows, inputCols)
	outputFrames := inputFrames * len(filters)
	outputs := tsr.NewEmptyTensor3D(outputFrames, inputRows, inputCols)
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
func (layer *ConvolutionLayer) Copy() Layer {
	return NewConvolutionLayer(
		layer.InputShape().Rows,
		layer.InputShape().Cols,
		layer.InputShape().Length,
		layer.Filters,
		layer.Activation,
	)
}

// InputShape returns the rows, columns and length of the inputs to the layer.
func (layer *ConvolutionLayer) InputShape() LayerShape {
	return layer.inputShape
}

// OutputShape returns the rows, columns and length of outputs from the layer.
func (layer *ConvolutionLayer) OutputShape() LayerShape {
	return layer.outputShape
}

// FeedForward applies convolutions to the input for each of the filters.
func (layer *ConvolutionLayer) FeedForward(inputs *tsr.Tensor) (*tsr.Tensor, error) {
	layer.inputs.SetAll(inputs)
	for _, filter := range layer.Filters {
		for frame := 0; frame < inputs.Frames; frame++ {
			for row := 0; row < inputs.Rows; row++ {
				for col := 0; col < inputs.Cols; col++ {
					value := layer.convolution(inputs, frame, row, col, filter)
					layer.outputs.Set(frame, row, col, value)
				}
			}
		}
	}
	layer.Activation.Function(layer.outputs)
	return layer.outputs, nil
}

// BackPropagate does not operate on the data in a convolution layer.
func (layer *ConvolutionLayer) BackPropagate(outputs *tsr.Tensor, learningRate float32, momentum float32) (*tsr.Tensor, error) {
	return layer.inputs, nil
}

func (layer *ConvolutionLayer) convolution(matrix *tsr.Tensor, frame int, row int, col int, filter *tsr.Tensor) float32 {
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
func (layer *ConvolutionLayer) MarshalJSON() ([]byte, error) {
	filters := make([][][]float32, len(layer.Filters))
	for i, filter := range layer.Filters {
		filters[i] = filter.GetFrame(0)
	}
	data := ConvolutionLayerData{
		Type:        LayerTypeConvolution,
		InputRows:   layer.InputShape().Rows,
		InputCols:   layer.InputShape().Cols,
		InputFrames: layer.InputShape().Length,
		Filters:     filters,
		Activation:  layer.Activation.Type,
	}
	return json.Marshal(data)
}

// UnmarshalJSON creates a new layer from JSON.
func (layer *ConvolutionLayer) UnmarshalJSON(b []byte) error {
	data := ConvolutionLayerData{}
	err := json.Unmarshal(b, &data)
	if err != nil {
		return err
	}
	layer.inputs = tsr.NewEmptyTensor3D(data.InputFrames, data.InputRows, data.InputCols)
	outputFrames := data.InputFrames * len(data.Filters)
	layer.outputs = tsr.NewEmptyTensor3D(data.InputFrames, data.InputRows, data.InputCols)
	layer.Filters = make([]*tsr.Tensor, len(data.Filters))
	for i, filter := range data.Filters {
		layer.Filters[i] = tsr.NewValueTensor2D(filter)
	}
	layer.Activation = activationFunctionOfType(data.Activation)
	layer.inputShape = LayerShape{data.InputRows, data.InputCols, data.InputFrames}
	layer.outputShape = LayerShape{data.InputRows, data.InputCols, outputFrames}
	return nil
}
