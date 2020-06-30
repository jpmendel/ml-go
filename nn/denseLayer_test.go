package nn

import (
	"testing"

	"../mat"
)

func TestDenseLayer(t *testing.T) {
	layer := NewDenseLayer(3, 2, ActivationRELU)

	originalOutputs := layer.outputs.Copy()
	originalWeights := layer.Weights.Copy()

	inputs := []*mat.Matrix{
		mat.NewMatrixWithValues([][]float32{
			{3, 4, 5},
		}),
	}

	targets := []*mat.Matrix{
		mat.NewMatrixWithValues([][]float32{
			{1, 2},
		}),
	}

	outputs, err := layer.FeedForward(inputs)
	if err != nil {
		t.Fatalf("Error in FeedForward: %s", err.Error())
	}
	if len(outputs) != 1 {
		t.Fatalf("Outputs have incorrect length: %d != %d", len(outputs), 1)
	}

	if outputs[0].Equals(originalOutputs) {
		t.Errorf("Matrix after feed forward should have changed from:\n%swhen result is:\n%s", originalOutputs.String(), outputs[0].String())
	}

	targets[0].SubtractMatrix(outputs[0])

	_, err = layer.BackPropagate(targets, 0.5, 0.0)
	if err != nil {
		t.Fatalf("Error in BackPropagate: %s", err.Error())
	}

	if layer.Weights.Equals(originalWeights) {
		t.Errorf("Weights after back propagate should have changed from:\n%swhen result is:\n%s", originalWeights, layer.Weights.String())
	}
}
