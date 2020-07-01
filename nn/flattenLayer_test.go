package nn

import (
	"testing"

	tsr "../tensor"
)

func TestFlattenLayer(t *testing.T) {
	layer := NewFlattenLayer(2, 3, 3)

	inputs := tsr.NewValueTensor3D([][][]float32{
		{
			{1, 2, 3},
			{4, 5, 6},
		},
		{
			{7, 8, 9},
			{10, 11, 12},
		},
		{
			{13, 14, 15},
			{16, 17, 18},
		},
	})

	solution := tsr.NewValueTensor1D([]float32{
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
	})

	flattened, err := layer.FeedForward(inputs)
	if err != nil {
		t.Fatalf("Error in FeedForward: %s", err.Error())
	}
	if flattened.Frames != 1 {
		t.Fatalf("Flattened outputs have incorrect frame length: %d != %d", flattened.Frames, 1)
	}

	if !flattened.Equals(solution) {
		t.Errorf("Matrix after feed forward should be:\n%swhen result is:\n%s", solution.String(), flattened.String())
	}

	unflattened, err := layer.BackPropagate(flattened, 0.0, 0.0)
	if err != nil {
		t.Fatalf("Error in BackPropagate: %s", err.Error())
	}

	if !unflattened.Equals(inputs) {
		t.Errorf("Matrix after back propagate should be:\n%swhen result is:\n%s", inputs.String(), unflattened.String())
	}
}
