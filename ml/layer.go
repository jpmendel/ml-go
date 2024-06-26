package ml

import "fmt"

// Layer is a stage of data computation in a neural network.
type Layer interface {
	Copy() Layer
	InputShape() LayerShape
	OutputShape() LayerShape
	FeedForward(inputs *Tensor) (*Tensor, error)
	BackPropagate(outputs *Tensor, learningRate float32, momentum float32) (*Tensor, error)
}

// LayerType represents the type of layer.
type LayerType string

const (
	// LayerTypeDense is a fully connected layer.
	LayerTypeDense = LayerType("dense")

	// LayerTypeConvolution is a convolutional layer.
	LayerTypeConvolution = LayerType("convolution")

	// LayerTypePooling is a layer where data is reduced into pools.
	LayerTypePooling = LayerType("pooling")

	// LayerTypeFlatten flattens the frames and rows of the data to 1.
	LayerTypeFlatten = LayerType("flatten")
)

// LayerShape is the rows, columns and frames of the data used in the layer.
type LayerShape struct {
	Rows   int
	Cols   int
	Frames int
}

func layerForType(layerType LayerType) (Layer, error) {
	switch layerType {
	case LayerTypeDense:
		return &DenseLayer{}, nil
	case LayerTypeConvolution:
		return &ConvolutionLayer{}, nil
	case LayerTypePooling:
		return &PoolingLayer{}, nil
	case LayerTypeFlatten:
		return &FlattenLayer{}, nil
	default:
		return nil, fmt.Errorf("invalid layer type: %s", layerType)
	}
}
