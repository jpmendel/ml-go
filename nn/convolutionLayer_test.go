package nn

import (
	"testing"

	"../mat"
)

func TestConvolutionLayer(t *testing.T) {
	layer := NewConvolutionLayer(5, 5, 1, []*mat.Matrix{FilterVerticalEdges}, ActivationRELU)

	inputs := []*mat.Matrix{
		mat.NewMatrixWithValues([][]float32{
			{0, 1, 0.5, 1, 0},
			{0.5, 1, 0.5, 1, 0.5},
			{0, 0.5, 0.5, 0.5, 0},
			{0.5, 1, 1, 1, 0.5},
			{0, 0, 0.5, 0, 0},
		}),
	}

	solution := mat.NewMatrixWithValues([][]float32{
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
	if len(convolutions) != 1 {
		t.Fatalf("Convolution outputs have incorrect length: %d != %d", len(convolutions), 1)
	}

	if !convolutions[0].Equals(solution) {
		t.Errorf("Matrix after feed forward should be:\n%swhen result is:\n%s", solution.String(), convolutions[0].String())
	}

	deconvolutions, err := layer.BackPropagate(convolutions, 0.0, 0.0)
	if err != nil {
		t.Fatalf("Error in BackPropagate: %s", err.Error())
	}
	if len(deconvolutions) != 1 {
		t.Fatalf("Convolution inputs have incorrect length: %d != %d", len(convolutions), 1)
	}

	if !deconvolutions[0].Equals(inputs[0]) {
		t.Errorf("Matrix after back propagate should be:\n%swhen result is:\n%s", inputs[0].String(), deconvolutions[0].String())
	}
}
