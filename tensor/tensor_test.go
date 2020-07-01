package tensor

import "testing"

func TestTensorCopy(t *testing.T) {
	tensor := NewValueTensor2D([][]float32{
		{1, 1, 1},
		{1, 1, 1},
	})

	shallow := tensor
	deep := tensor.Copy()

	tensor.Set(0, 0, 0, 2)

	shallowValue := shallow.Get(0, 0, 0)
	if shallowValue != 2 {
		t.Errorf("Shallow copy value shoud be 2, is: %.0f", shallowValue)
	}

	deepValue := deep.Get(0, 0, 0)
	if deepValue != 1 {
		t.Errorf("Deep copy value shoud be 1, is: %.0f", deepValue)
	}
}

func TestTensorGetSet(t *testing.T) {
	tensor := NewValueTensor2D([][]float32{
		{1, 3, 2},
		{-3, 2, -1},
	})

	tensor.Set(0, 0, 0, 2)

	value := tensor.Get(0, 0, 0)
	if value != 2 {
		t.Errorf("Value shoud be 2, is: %.0f", value)
	}

	set := NewValueTensor2D([][]float32{
		{3, 2, 3},
		{2, -1, 2},
	})

	tensor.SetAll(set)

	if !tensor.Equals(set) {
		t.Errorf("Tensor after setting all values should be:\n%swhen result is:\n%s", set.String(), tensor.String())
	}

	invalid := NewValueTensor2D([][]float32{
		{3, 2},
		{2, -1},
	})

	err := tensor.SetAll(invalid)

	if err == nil {
		t.Errorf("Setting values from invalid tensor did not trigger error")
	}
}

func TestTensorAdd(t *testing.T) {
	tensor := NewValueTensor2D([][]float32{
		{1, 3, 2},
		{-3, 2, -1},
	})

	tensor.Add(5)

	solution1 := NewValueTensor2D([][]float32{
		{6, 8, 7},
		{2, 7, 4},
	})
	if !tensor.Equals(solution1) {
		t.Errorf("Tensor after adding 5 should be:\n%swhen result is:\n%s", solution1.String(), tensor.String())
	}

	added := NewValueTensor2D([][]float32{
		{3, 2, 3},
		{2, -1, 2},
	})

	tensor.AddTensor(added)

	solution2 := NewValueTensor2D([][]float32{
		{9, 10, 10},
		{4, 6, 6},
	})
	if !tensor.Equals(solution2) {
		t.Errorf("Tensor after adding other tensor should be:\n%swhen result is:\n%s", solution2.String(), tensor.String())
	}

	invalid := NewValueTensor2D([][]float32{
		{3, 2},
		{2, -1},
	})

	err := tensor.AddTensor(invalid)

	if err == nil {
		t.Errorf("Adding invalid tensor did not trigger error")
	}
}

func TestTensorSubtract(t *testing.T) {
	tensor := NewValueTensor2D([][]float32{
		{1, 3, 2},
		{-3, 2, -1},
	})

	tensor.Subtract(5)

	solution1 := NewValueTensor2D([][]float32{
		{-4, -2, -3},
		{-8, -3, -6},
	})
	if !tensor.Equals(solution1) {
		t.Errorf("Tensor after subtracting 5 should be:\n%swhen result is:\n%s", solution1.String(), tensor.String())
	}

	subtracted := NewValueTensor2D([][]float32{
		{3, 2, 3},
		{2, -1, 2},
	})

	tensor.SubtractTensor(subtracted)

	solution2 := NewValueTensor2D([][]float32{
		{-7, -4, -6},
		{-10, -2, -8},
	})
	if !tensor.Equals(solution2) {
		t.Errorf("Tensor after subtracting other tensor should be:\n%swhen result is:\n%s", solution2.String(), tensor.String())
	}

	invalid := NewValueTensor2D([][]float32{
		{3, 2},
		{2, -1},
	})

	err := tensor.SubtractTensor(invalid)

	if err == nil {
		t.Errorf("Subtracting invalid tensor did not trigger error")
	}
}

func TestTensorScale(t *testing.T) {
	tensor := NewValueTensor2D([][]float32{
		{1, 3, 2},
		{-3, 2, -1},
	})

	tensor.Scale(3)

	solution1 := NewValueTensor2D([][]float32{
		{3, 9, 6},
		{-9, 6, -3},
	})
	if !tensor.Equals(solution1) {
		t.Errorf("Tensor after scaling by 3 should be:\n%swhen result is:\n%s", solution1.String(), tensor.String())
	}

	scaled := NewValueTensor2D([][]float32{
		{3, 2, 3},
		{2, -1, 2},
	})

	tensor.ScaleTensor(scaled)

	solution2 := NewValueTensor2D([][]float32{
		{9, 18, 18},
		{-18, -6, -6},
	})
	if !tensor.Equals(solution2) {
		t.Errorf("Tensor after scaling by other tensor should be:\n%swhen result is:\n%s", solution2.String(), tensor.String())
	}

	invalid := NewValueTensor2D([][]float32{
		{3, 2},
		{2, -1},
	})

	err := tensor.ScaleTensor(invalid)

	if err == nil {
		t.Errorf("Scaling by invalid tensor did not trigger error")
	}
}

func TestTensorMultiply(t *testing.T) {
	tensor1 := NewValueTensor2D([][]float32{
		{2, 3},
		{4, 1},
		{1, 3},
	})

	tensor2 := NewValueTensor2D([][]float32{
		{2, 2, 3},
		{3, 1, 4},
	})

	result, err := MatrixMultiply(tensor1, tensor2, nil)
	if err != nil {
		t.Fatalf("Error in TensorMultiply: %s", err.Error())
	}

	solution := NewValueTensor2D([][]float32{
		{13, 7, 18},
		{11, 9, 16},
		{11, 5, 15},
	})
	if !result.Equals(solution) {
		t.Errorf("Tensor multiplication result should be:\n%swhen result is:\n%s", solution.String(), result.String())
	}

	invalid := NewValueTensor2D([][]float32{
		{2, 2, 3, 1},
	})

	_, err = MatrixMultiply(tensor1, invalid, nil)
	if err == nil {
		t.Errorf("Multiplying matrices with invalid dimensions did not trigger error")
	}
}

func TestTensorTranspose(t *testing.T) {
	tensor := NewValueTensor2D([][]float32{
		{1, 3, 2},
		{-3, 2, -1},
	})

	result, err := MatrixTranspose(tensor, nil)
	if err != nil {
		t.Fatalf("Error in TensorTranspose: %s", err.Error())
	}

	solution := NewValueTensor2D([][]float32{
		{1, -3},
		{3, 2},
		{2, -1},
	})
	if !result.Equals(solution) {
		t.Errorf("Tensor transpose result should be:\n%swhen result is:\n%s", solution.String(), result.String())
	}
}
