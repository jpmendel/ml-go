package nn

import (
	"testing"

	tsr "../tensor"
)

func TestDenseLayer(t *testing.T) {
	layer := NewDenseLayer(3, 2, ActivationRELU)

	originalOutputs := layer.outputs.Copy()
	originalWeights := layer.Weights.Copy()

	inputs := tsr.NewValueTensor1D([]float32{3, 4, 5})
	targets := tsr.NewValueTensor1D([]float32{1, 2})

	outputs, err := layer.FeedForward(inputs)
	if err != nil {
		t.Fatalf("Error in FeedForward: %s", err.Error())
	}
	if outputs.Frames != 1 {
		t.Fatalf("Outputs have incorrect frame length: %d != %d", outputs.Frames, 1)
	}

	if outputs.Equals(originalOutputs) {
		t.Errorf("Matrix after feed forward should have changed from:\n%swhen result is:\n%s", originalOutputs.String(), outputs.String())
	}

	targets.SubtractTensor(outputs)

	_, err = layer.BackPropagate(targets, 0.5, 0.0)
	if err != nil {
		t.Fatalf("Error in BackPropagate: %s", err.Error())
	}

	if layer.Weights.Equals(originalWeights) {
		t.Errorf("Weights after back propagate should have changed from:\n%swhen result is:\n%s", originalWeights, layer.Weights.String())
	}
}
