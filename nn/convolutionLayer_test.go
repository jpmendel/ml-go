package nn

import (
	"testing"

	tsr "../tensor"
)

func TestConvolutionLayer(t *testing.T) {
	filters := []*tsr.Tensor{FilterVerticalEdges}
	layer := NewConvolutionLayer(5, 5, 1, filters, ActivationRELU)

	inputs := tsr.NewValueTensor2D([][]float32{
		{0, 1, 0.5, 1, 0},
		{0.5, 1, 0.5, 1, 0.5},
		{0, 0.5, 0.5, 0.5, 0},
		{0.5, 1, 1, 1, 0.5},
		{0, 0, 0.5, 0, 0},
	})

	solution := tsr.NewValueTensor2D([][]float32{
		{2, 0.5, 0, 0, 0},
		{2.5, 1, 0, 0, 0},
		{2.5, 1, 0, 0, 0},
		{1.5, 1.5, 0, 0, 0},
		{1, 1, 0, 0, 0},
	})

	convolutions, err := layer.FeedForward(inputs)
	if err != nil {
		t.Fatalf("Error in FeedForward: %s", err.Error())
	}
	if convolutions.Frames != inputs.Frames*len(filters) {
		t.Fatalf("Convolution outputs have incorrect length: %d != %d", convolutions.Frames, inputs.Frames*len(filters))
	}

	if !convolutions.Equals(solution) {
		t.Errorf("Matrix after feed forward should be:\n%swhen result is:\n%s", solution.String(), convolutions.String())
	}

	deconvolutions, err := layer.BackPropagate(convolutions, 0.0, 0.0)
	if err != nil {
		t.Fatalf("Error in BackPropagate: %s", err.Error())
	}
	if deconvolutions.Frames != inputs.Frames {
		t.Fatalf("Convolution inputs have incorrect length: %d != %d", deconvolutions.Frames, inputs.Frames)
	}

	if !deconvolutions.Equals(inputs) {
		t.Errorf("Matrix after back propagate should be:\n%swhen result is:\n%s", inputs.String(), deconvolutions.String())
	}
}
