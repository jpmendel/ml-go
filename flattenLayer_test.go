package ml

import "testing"

func TestFlattenLayer(t *testing.T) {
	layer := NewFlattenLayer(2, 3, 3)

	inputs := []*Matrix{
		NewMatrixWithValues([][]float32{
			{1, 2, 3},
			{4, 5, 6},
		}),
		NewMatrixWithValues([][]float32{
			{7, 8, 9},
			{10, 11, 12},
		}),
		NewMatrixWithValues([][]float32{
			{13, 14, 15},
			{16, 17, 18},
		}),
	}

	solution := NewMatrixWithValues([][]float32{
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
	})

	flattened, err := layer.FeedForward(inputs)
	if err != nil {
		t.Fatalf("Error in FeedForward: %s", err.Error())
	}
	if len(flattened) != 1 {
		t.Fatalf("Flattened outputs have incorrect length: %d != %d", len(flattened), 1)
	}

	if !flattened[0].Equals(solution) {
		t.Errorf("Matrix after feed forward should be:\n%swhen result is:\n%s", solution.String(), flattened[0].String())
	}

	unflattened, err := layer.BackPropagate(flattened, 0.0, 0.0)
	if err != nil {
		t.Fatalf("Error in BackPropagate: %s", err.Error())
	}

	for i, input := range unflattened {
		if !input.Equals(inputs[i]) {
			t.Errorf("Matrix at index %d after back propagate should be:\n%swhen result is:\n%s", i, inputs[i].String(), input.String())
		}
	}
}
