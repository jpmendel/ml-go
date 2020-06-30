package ml

import "encoding/json"

// ConvolutionLayer is a layer that performs convolutional filters on data.
type ConvolutionLayer struct {
	inputs     []*Matrix
	outputs    []*Matrix
	Filters    []*Matrix
	Activation ActivationFunction
}

// NewConvolutionLayer creates a new instance of a convolutional layer.
func NewConvolutionLayer(rows int, cols int, inputLength int, filters []*Matrix, activation ActivationFunction) *ConvolutionLayer {
	inputs := make([]*Matrix, inputLength)
	for i := 0; i < inputLength; i++ {
		inputs[i] = NewEmptyMatrix(rows, cols)
	}
	outputLength := inputLength * len(filters)
	outputs := make([]*Matrix, outputLength)
	for i := 0; i < outputLength; i++ {
		outputs[i] = NewEmptyMatrix(rows, cols)
	}
	return &ConvolutionLayer{
		inputs:     inputs,
		outputs:    outputs,
		Filters:    filters,
		Activation: activation,
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
	return LayerShape{layer.inputs[0].Rows, layer.inputs[0].Cols, len(layer.inputs)}
}

// OutputShape returns the rows, columns and length of outputs from the layer.
func (layer *ConvolutionLayer) OutputShape() LayerShape {
	return LayerShape{layer.outputs[0].Rows, layer.outputs[0].Cols, len(layer.outputs)}
}

// FeedForward applies convolutions to the input for each of the filters.
func (layer *ConvolutionLayer) FeedForward(inputs []*Matrix) ([]*Matrix, error) {
	for i, input := range inputs {
		layer.inputs[i].SetAll(input)
		for _, filter := range layer.Filters {
			for row := 0; row < input.Rows; row++ {
				for col := 0; col < input.Cols; col++ {
					value := layer.convolution(input, row, col, filter)
					input.Set(row, col, value)
				}
			}
		}
	}
	return layer.outputs, nil
}

// BackPropagate does not operate on the data in a convolution layer.
func (layer *ConvolutionLayer) BackPropagate(outputs []*Matrix, learningRate float32, momentum float32) ([]*Matrix, error) {
	return layer.inputs, nil
}

func (layer *ConvolutionLayer) convolution(matrix *Matrix, row int, col int, filter *Matrix) float32 {
	sum := float32(0.0)
	for or := -filter.Rows / 2; or <= filter.Rows/2; or++ {
		convRow := row + or
		for oc := -filter.Cols / 2; oc <= filter.Cols/2; oc++ {
			convCol := row + oc
			if convRow < 0 || convRow >= matrix.Rows || convCol < 0 || convCol >= matrix.Cols {
				continue
			}
			sum += matrix.Get(convRow, convCol) * filter.Get(or+filter.Rows/2, oc+filter.Cols/2)
		}
	}
	return sum
}

// ConvolutionLayerData represents a serialized layer that can be saved to a file.
type ConvolutionLayerData struct {
	Type        LayerType      `json:"type"`
	Rows        int            `json:"inputRows"`
	Cols        int            `json:"inputCols"`
	InputLength int            `json:"length"`
	Filters     [][][]float32  `json:"filters"`
	Activation  ActivationType `json:"activation"`
}

// MarshalJSON converts the layer to JSON.
func (layer *ConvolutionLayer) MarshalJSON() ([]byte, error) {
	filters := make([][][]float32, len(layer.Filters))
	for i, filter := range layer.Filters {
		filters[i] = filter.values
	}
	data := ConvolutionLayerData{
		Type:        LayerTypeConvolution,
		Rows:        layer.InputShape().Rows,
		Cols:        layer.InputShape().Cols,
		InputLength: layer.InputShape().Length,
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
	layer.inputs = make([]*Matrix, data.InputLength)
	for i := 0; i < data.InputLength; i++ {
		layer.inputs[i] = NewEmptyMatrix(data.Rows, data.Cols)
	}
	outputLength := data.InputLength * len(data.Filters)
	layer.outputs = make([]*Matrix, outputLength)
	for i := 0; i < outputLength; i++ {
		layer.outputs[i] = NewEmptyMatrix(data.Rows, data.Cols)
	}
	layer.Filters = make([]*Matrix, len(data.Filters))
	for i, filter := range data.Filters {
		layer.Filters[i] = NewMatrixWithValues(filter)
	}
	layer.Activation = activationFunctionOfType(data.Activation)
	return nil
}
