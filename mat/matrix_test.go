package mat

import "testing"

func TestMatrixCopy(t *testing.T) {
	matrix := NewMatrixWithValues([][]float32{
		{1, 1, 1},
		{1, 1, 1},
	})

	shallow := matrix
	deep := matrix.Copy()

	matrix.Set(0, 0, 2)

	shallowValue := shallow.Get(0, 0)
	if shallowValue != 2 {
		t.Errorf("Shallow copy value shoud be 2, is: %.0f", shallowValue)
	}

	deepValue := deep.Get(0, 0)
	if deepValue != 1 {
		t.Errorf("Deep copy value shoud be 1, is: %.0f", deepValue)
	}
}

func TestMatrixGetSet(t *testing.T) {
	matrix := NewMatrixWithValues([][]float32{
		{1, 3, 2},
		{-3, 2, -1},
	})

	matrix.Set(0, 0, 2)

	value := matrix.Get(0, 0)
	if value != 2 {
		t.Errorf("Value shoud be 2, is: %.0f", value)
	}

	set := NewMatrixWithValues([][]float32{
		{3, 2, 3},
		{2, -1, 2},
	})

	matrix.SetAll(set)

	if !matrix.Equals(set) {
		t.Errorf("Matrix after setting all values should be:\n%swhen result is:\n%s", set.String(), matrix.String())
	}

	invalid := NewMatrixWithValues([][]float32{
		{3, 2},
		{2, -1},
	})

	err := matrix.SetAll(invalid)

	if err == nil {
		t.Errorf("Setting values from invalid matrix did not trigger error")
	}
}

func TestMatrixAdd(t *testing.T) {
	matrix := NewMatrixWithValues([][]float32{
		{1, 3, 2},
		{-3, 2, -1},
	})

	matrix.Add(5)

	solution1 := NewMatrixWithValues([][]float32{
		{6, 8, 7},
		{2, 7, 4},
	})
	if !matrix.Equals(solution1) {
		t.Errorf("Matrix after adding 5 should be:\n%swhen result is:\n%s", solution1.String(), matrix.String())
	}

	added := NewMatrixWithValues([][]float32{
		{3, 2, 3},
		{2, -1, 2},
	})

	matrix.AddMatrix(added)

	solution2 := NewMatrixWithValues([][]float32{
		{9, 10, 10},
		{4, 6, 6},
	})
	if !matrix.Equals(solution2) {
		t.Errorf("Matrix after adding other matrix should be:\n%swhen result is:\n%s", solution2.String(), matrix.String())
	}

	invalid := NewMatrixWithValues([][]float32{
		{3, 2},
		{2, -1},
	})

	err := matrix.AddMatrix(invalid)

	if err == nil {
		t.Errorf("Adding invalid matrix did not trigger error")
	}
}

func TestMatrixSubtract(t *testing.T) {
	matrix := NewMatrixWithValues([][]float32{
		{1, 3, 2},
		{-3, 2, -1},
	})

	matrix.Subtract(5)

	solution1 := NewMatrixWithValues([][]float32{
		{-4, -2, -3},
		{-8, -3, -6},
	})
	if !matrix.Equals(solution1) {
		t.Errorf("Matrix after subtracting 5 should be:\n%swhen result is:\n%s", solution1.String(), matrix.String())
	}

	subtracted := NewMatrixWithValues([][]float32{
		{3, 2, 3},
		{2, -1, 2},
	})

	matrix.SubtractMatrix(subtracted)

	solution2 := NewMatrixWithValues([][]float32{
		{-7, -4, -6},
		{-10, -2, -8},
	})
	if !matrix.Equals(solution2) {
		t.Errorf("Matrix after subtracting other matrix should be:\n%swhen result is:\n%s", solution2.String(), matrix.String())
	}

	invalid := NewMatrixWithValues([][]float32{
		{3, 2},
		{2, -1},
	})

	err := matrix.SubtractMatrix(invalid)

	if err == nil {
		t.Errorf("Subtracting invalid matrix did not trigger error")
	}
}

func TestMatrixScale(t *testing.T) {
	matrix := NewMatrixWithValues([][]float32{
		{1, 3, 2},
		{-3, 2, -1},
	})

	matrix.Scale(3)

	solution1 := NewMatrixWithValues([][]float32{
		{3, 9, 6},
		{-9, 6, -3},
	})
	if !matrix.Equals(solution1) {
		t.Errorf("Matrix after scaling by 3 should be:\n%swhen result is:\n%s", solution1.String(), matrix.String())
	}

	scaled := NewMatrixWithValues([][]float32{
		{3, 2, 3},
		{2, -1, 2},
	})

	matrix.ScaleMatrix(scaled)

	solution2 := NewMatrixWithValues([][]float32{
		{9, 18, 18},
		{-18, -6, -6},
	})
	if !matrix.Equals(solution2) {
		t.Errorf("Matrix after scaling by other matrix should be:\n%swhen result is:\n%s", solution2.String(), matrix.String())
	}

	invalid := NewMatrixWithValues([][]float32{
		{3, 2},
		{2, -1},
	})

	err := matrix.ScaleMatrix(invalid)

	if err == nil {
		t.Errorf("Scaling by invalid matrix did not trigger error")
	}
}

func TestMatrixMultiply(t *testing.T) {
	matrix1 := NewMatrixWithValues([][]float32{
		{2, 3},
		{4, 1},
		{1, 3},
	})

	matrix2 := NewMatrixWithValues([][]float32{
		{2, 2, 3},
		{3, 1, 4},
	})

	result, err := MatrixMultiply(matrix1, matrix2, nil)
	if err != nil {
		t.Fatalf("Error in MatrixMultiply: %s", err.Error())
	}

	solution := NewMatrixWithValues([][]float32{
		{13, 7, 18},
		{11, 9, 16},
		{11, 5, 15},
	})
	if !result.Equals(solution) {
		t.Errorf("Matrix multiplication result should be:\n%swhen result is:\n%s", solution.String(), result.String())
	}

	invalid := NewMatrixWithValues([][]float32{
		{2, 2, 3, 1},
	})

	_, err = MatrixMultiply(matrix1, invalid, nil)
	if err == nil {
		t.Errorf("Multiplying matrices with invalid dimensions did not trigger error")
	}
}

func TestMatrixTranspose(t *testing.T) {
	matrix := NewMatrixWithValues([][]float32{
		{1, 3, 2},
		{-3, 2, -1},
	})

	result, err := MatrixTranspose(matrix, nil)
	if err != nil {
		t.Fatalf("Error in MatrixTranspose: %s", err.Error())
	}

	solution := NewMatrixWithValues([][]float32{
		{1, -3},
		{3, 2},
		{2, -1},
	})
	if !result.Equals(solution) {
		t.Errorf("Matrix transpose result should be:\n%swhen result is:\n%s", solution.String(), result.String())
	}
}
