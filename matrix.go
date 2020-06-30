package ml

import (
	"fmt"
	"log"
	"math/rand"
)

// Matrix represents a 2D matrix of values.
type Matrix struct {
	Rows   int
	Cols   int
	values [][]float32
}

// NewEmptyMatrix creates a new matrix with the given dimensions and sets all values to zero.
func NewEmptyMatrix(rows int, cols int) *Matrix {
	values := make([][]float32, rows)
	for row := 0; row < rows; row++ {
		values[row] = make([]float32, cols)
		for col := 0; col < cols; col++ {
			values[row][col] = 0.0
		}
	}
	return &Matrix{Rows: rows, Cols: cols, values: values}
}

// NewMatrixWithValues creates a new matrix with the given values.
func NewMatrixWithValues(values [][]float32) *Matrix {
	rows := len(values)
	cols := len(values[0])
	matrixValues := make([][]float32, rows)
	for row := 0; row < rows; row++ {
		matrixValues[row] = make([]float32, cols)
		for col := 0; col < cols; col++ {
			matrixValues[row][col] = values[row][col]
		}
	}
	return &Matrix{Rows: rows, Cols: cols, values: matrixValues}
}

// Copy creates a deep copy of the matrix.
func (matrix *Matrix) Copy() *Matrix {
	return NewMatrixWithValues(matrix.values)
}

// Equals checks that all the values in two matrices are equivalent.
func (matrix *Matrix) Equals(other *Matrix) bool {
	if matrix.Rows != other.Rows || matrix.Cols != other.Cols {
		return false
	}
	for row := 0; row < matrix.Rows; row++ {
		for col := 0; col < matrix.Cols; col++ {
			if matrix.Get(row, col) != other.Get(row, col) {
				return false
			}
		}
	}
	return true
}

// Get retrieves a value at a specific row and column.
func (matrix *Matrix) Get(row int, col int) float32 {
	if row < 0 || row >= matrix.Rows || col < 0 || col >= matrix.Cols {
		log.Panicf("Row or column out of bounds: (%d, %d)", row, col)
	}
	return matrix.values[row][col]
}

// Sum gets the sum of all values in the matrix.
func (matrix *Matrix) Sum() float32 {
	sum := float32(0.0)
	for row := 0; row < matrix.Rows; row++ {
		for col := 0; col < matrix.Cols; col++ {
			sum += matrix.Get(row, col)
		}
	}
	return sum
}

// Set sets a new value at a specific row and column.
func (matrix *Matrix) Set(row int, col int, value float32) error {
	if row < 0 || row >= matrix.Rows || col < 0 || col >= matrix.Cols {
		return fmt.Errorf("Row or column out of bounds: (%d, %d)", row, col)
	}
	matrix.values[row][col] = value
	return nil
}

// SetAll sets all the values of the current matrix to the values of the input matrix.
func (matrix *Matrix) SetAll(other *Matrix) error {
	if matrix.Rows != other.Rows || matrix.Cols != other.Cols {
		return fmt.Errorf("Rows and columns must match: (%d, %d) != (%d, %d)", matrix.Rows, matrix.Cols, other.Rows, other.Cols)
	}
	matrix.ApplyFunction(func(current float32, row int, col int) float32 {
		return other.Get(row, col)
	})
	return nil
}

// SetRandom sets all the values of the current matrix to random values between min and max.
func (matrix *Matrix) SetRandom(min float32, max float32) error {
	if min >= max {
		return fmt.Errorf("Minimum must be less than maximum: %f >= %f", min, max)
	}
	matrix.ApplyFunction(func(current float32, row int, col int) float32 {
		return rand.Float32()*(max-min) + min
	})
	return nil
}

// Add adds a scalar value to all the values of the matrix.
func (matrix *Matrix) Add(value float32) {
	matrix.ApplyFunction(func(current float32, row int, col int) float32 {
		return current + value
	})
}

// AddMatrix adds the values of the input matrix to the values of the current matrix.
func (matrix *Matrix) AddMatrix(other *Matrix) error {
	if matrix.Rows != other.Rows || matrix.Cols != other.Cols {
		return fmt.Errorf("Rows and columns must match: (%d, %d) != (%d, %d)", matrix.Rows, matrix.Cols, other.Rows, other.Cols)
	}
	matrix.ApplyFunction(func(current float32, row int, col int) float32 {
		return current + other.Get(row, col)
	})
	return nil
}

// Subtract subtracts a scalar value from all the values of the matrix.
func (matrix *Matrix) Subtract(value float32) {
	matrix.ApplyFunction(func(current float32, row int, col int) float32 {
		return current - value
	})
}

// SubtractMatrix subtracts the values of the input matrix from the values of the current matrix.
func (matrix *Matrix) SubtractMatrix(other *Matrix) error {
	if matrix.Rows != other.Rows || matrix.Cols != other.Cols {
		return fmt.Errorf("Rows and columns must match: (%d, %d) != (%d, %d)", matrix.Rows, matrix.Cols, other.Rows, other.Cols)
	}
	matrix.ApplyFunction(func(current float32, row int, col int) float32 {
		return current - other.Get(row, col)
	})
	return nil
}

// Scale multiplies all the values of the current matrix by a scalar value.
func (matrix *Matrix) Scale(value float32) {
	matrix.ApplyFunction(func(current float32, row int, col int) float32 {
		return current * value
	})
}

// ScaleMatrix multiplies all the values of the current matrix by the values of the input matrix.
func (matrix *Matrix) ScaleMatrix(other *Matrix) error {
	if matrix.Rows != other.Rows || matrix.Cols != other.Cols {
		return fmt.Errorf("Rows and columns must match: (%d, %d) != (%d, %d)", matrix.Rows, matrix.Cols, other.Rows, other.Cols)
	}
	matrix.ApplyFunction(func(current float32, row int, col int) float32 {
		return current * other.Get(row, col)
	})
	return nil
}

// ApplyFunction applies the input function to all the values of the matrix.
func (matrix *Matrix) ApplyFunction(function func(float32, int, int) float32) {
	for row := 0; row < matrix.Rows; row++ {
		for col := 0; col < matrix.Cols; col++ {
			matrix.Set(row, col, function(matrix.Get(row, col), row, col))
		}
	}
}

// String creates a string representation of the matrix.
func (matrix *Matrix) String() string {
	str := ""
	for row := 0; row < matrix.Rows; row++ {
		for col := 0; col < matrix.Cols; col++ {
			str += fmt.Sprintf("%.4f", matrix.Get(row, col))
			if col == matrix.Cols-1 {
				str += "\n"
			} else {
				str += " "
			}
		}
	}
	return str
}

// MatrixMultiply multiplies two matrices where the rows of the first equals the columns of the second.
func MatrixMultiply(matrix1 *Matrix, matrix2 *Matrix, target *Matrix) (*Matrix, error) {
	if matrix1.Cols != matrix2.Rows {
		return nil, fmt.Errorf("Columns of first must match rows of second: %d != %d", matrix1.Cols, matrix2.Rows)
	}
	var result *Matrix
	if target != nil {
		if target.Rows != matrix1.Rows || target.Cols != matrix2.Cols {
			return nil, fmt.Errorf("Invalid target dimensions: (%d, %d) != (%d, %d)", target.Rows, target.Cols, matrix1.Rows, matrix2.Cols)
		}
		result = target
	} else {
		result = NewEmptyMatrix(matrix1.Rows, matrix2.Cols)
	}
	for row := 0; row < matrix1.Rows; row++ {
		for col := 0; col < matrix2.Cols; col++ {
			sum := float32(0.0)
			for i := 0; i < matrix1.Cols; i++ {
				sum += matrix1.Get(row, i) * matrix2.Get(i, col)
			}
			result.Set(row, col, sum)
		}
	}
	return result, nil
}

// MatrixTranspose transposes a matrix, flipping it over its primary axis.
func MatrixTranspose(matrix *Matrix, target *Matrix) (*Matrix, error) {
	var result *Matrix
	if target != nil {
		if target.Rows != matrix.Cols || target.Cols != matrix.Rows {
			return nil, fmt.Errorf("Invalid target dimensions: (%d, %d) != (%d, %d)", target.Rows, target.Cols, matrix.Cols, matrix.Rows)
		}
		result = target
	} else {
		result = NewEmptyMatrix(matrix.Cols, matrix.Rows)
	}
	result.ApplyFunction(func(current float32, row int, col int) float32 {
		return matrix.Get(col, row)
	})
	return result, nil
}
