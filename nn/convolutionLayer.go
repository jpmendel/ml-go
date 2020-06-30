package nn

import (
	"encoding/json"

	"../mat"
)

// ConvolutionLayer is a layer that performs convolutional filters on data.
type ConvolutionLayer struct {
	inputShape  LayerShape
	outputShape LayerShape
	inputs      []*mat.Matrix
	outputs     []*mat.Matrix
	Filters     []*mat.Matrix
	Activation  ActivationFunction
}

// NewConvolutionLayer creates a new instance of a convolutional layer.
func NewConvolutionLayer(inputRows int, inputCols int, inputLength int, filters []*mat.Matrix, activation ActivationFunction) *ConvolutionLayer {
	inputs := make([]*mat.Matrix, inputLength)
	for i := 0; i < inputLength; i++ {
		inputs[i] = mat.NewEmptyMatrix(inputRows, inputCols)
	}
	outputLength := inputLength * len(filters)
	outputs := make([]*mat.Matrix, outputLength)
	for i := 0; i < outputLength; i++ {
		outputs[i] = mat.NewEmptyMatrix(inputRows, inputCols)
	}
	return &ConvolutionLayer{
		inputShape:  LayerShape{inputRows, inputCols, inputLength},
		outputShape: LayerShape{inputRows, inputCols, outputLength},
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
func (layer *ConvolutionLayer) FeedForward(inputs []*mat.Matrix) ([]*mat.Matrix, error) {
	for i, input := range inputs {
		layer.inputs[i].SetAll(input)
		for _, filter := range layer.Filters {
			for row := 0; row < input.Rows; row++ {
				for col := 0; col < input.Cols; col++ {
					value := layer.convolution(input, row, col, filter)
					layer.outputs[i].Set(row, col, value)
				}
			}
		}
	}
	return layer.outputs, nil
}

// BackPropagate does not operate on the data in a convolution layer.
func (layer *ConvolutionLayer) BackPropagate(outputs []*mat.Matrix, learningRate float32, momentum float32) ([]*mat.Matrix, error) {
	return layer.inputs, nil
}

func (layer *ConvolutionLayer) convolution(matrix *mat.Matrix, row int, col int, filter *mat.Matrix) float32 {
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
	InputRows   int            `json:"inputRows"`
	InputCols   int            `json:"inputCols"`
	InputLength int            `json:"inputLength"`
	Filters     [][][]float32  `json:"filters"`
	Activation  ActivationType `json:"activation"`
}

// MarshalJSON converts the layer to JSON.
func (layer *ConvolutionLayer) MarshalJSON() ([]byte, error) {
	filters := make([][][]float32, len(layer.Filters))
	for i, filter := range layer.Filters {
		filters[i] = filter.GetAll()
	}
	data := ConvolutionLayerData{
		Type:        LayerTypeConvolution,
		InputRows:   layer.InputShape().Rows,
		InputCols:   layer.InputShape().Cols,
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
	layer.inputs = make([]*mat.Matrix, data.InputLength)
	for i := 0; i < data.InputLength; i++ {
		layer.inputs[i] = mat.NewEmptyMatrix(data.InputRows, data.InputCols)
	}
	outputLength := data.InputLength * len(data.Filters)
	layer.outputs = make([]*mat.Matrix, outputLength)
	for i := 0; i < outputLength; i++ {
		layer.outputs[i] = mat.NewEmptyMatrix(data.InputRows, data.InputCols)
	}
	layer.Filters = make([]*mat.Matrix, len(data.Filters))
	for i, filter := range data.Filters {
		layer.Filters[i] = mat.NewMatrixWithValues(filter)
	}
	layer.Activation = activationFunctionOfType(data.Activation)
	layer.inputShape = LayerShape{data.InputRows, data.InputCols, data.InputLength}
	layer.outputShape = LayerShape{data.InputRows, data.InputCols, outputLength}
	return nil
}
