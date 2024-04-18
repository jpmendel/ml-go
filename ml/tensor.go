package ml

import (
	"fmt"
	"log"
	"math/rand"
)

// Tensor represents a multi-dimensional set of values.
type Tensor struct {
	Frames int
	Rows   int
	Cols   int
	values [][][]float32
}

// NewEmptyTensor1D creates a new tensor with a single frame, row and given number of columns.
func NewEmptyTensor1D(cols int) *Tensor {
	values := make([]float32, cols)
	for col := 0; col < cols; col++ {
		values[col] = 0.0
	}
	return &Tensor{Frames: 1, Rows: 1, Cols: cols, values: [][][]float32{{values}}}
}

// NewEmptyTensor2D creates a new tensor with a single frame and given number of rows and columns.
func NewEmptyTensor2D(rows int, cols int) *Tensor {
	values := make([][]float32, rows)
	for row := 0; row < rows; row++ {
		values[row] = make([]float32, cols)
		for col := 0; col < cols; col++ {
			values[row][col] = 0.0
		}
	}
	return &Tensor{Frames: 1, Rows: rows, Cols: cols, values: [][][]float32{values}}
}

// NewEmptyTensor3D creates a new tensor with the given number of rows, columns and frames.
func NewEmptyTensor3D(frames int, rows int, cols int) *Tensor {
	values := make([][][]float32, frames)
	for frame := 0; frame < frames; frame++ {
		values[frame] = make([][]float32, rows)
		for row := 0; row < rows; row++ {
			values[frame][row] = make([]float32, cols)
			for col := 0; col < cols; col++ {
				values[frame][row][col] = 0.0
			}
		}
	}
	return &Tensor{Frames: frames, Rows: rows, Cols: cols, values: values}
}

// NewValueTensor1D creates a new tensor with the given 1D array of values.
func NewValueTensor1D(values []float32) *Tensor {
	if len(values) == 0 {
		return &Tensor{Frames: 1, Rows: 1, Cols: 1, values: [][][]float32{{{0.0}}}}
	}
	cols := len(values)
	tensorValues := make([]float32, cols)
	for col := 0; col < cols; col++ {
		tensorValues[col] = values[col]
	}
	return &Tensor{Frames: 1, Rows: 1, Cols: cols, values: [][][]float32{{tensorValues}}}
}

// NewValueTensor2D creates a new tensor with the given 2D array of values.
func NewValueTensor2D(values [][]float32) *Tensor {
	if len(values) == 0 {
		return &Tensor{Frames: 1, Rows: 1, Cols: 1, values: [][][]float32{{{0.0}}}}
	}
	rows := len(values)
	cols := len(values[0])
	tensorValues := make([][]float32, rows)
	for row := 0; row < rows; row++ {
		tensorValues[row] = make([]float32, cols)
		for col := 0; col < cols; col++ {
			tensorValues[row][col] = values[row][col]
		}
	}
	return &Tensor{Frames: 1, Rows: rows, Cols: cols, values: [][][]float32{tensorValues}}
}

// NewValueTensor3D creates a new tensor with the given 3D array of values.
func NewValueTensor3D(values [][][]float32) *Tensor {
	if len(values) == 0 {
		return &Tensor{Frames: 1, Rows: 1, Cols: 1, values: [][][]float32{{{0.0}}}}
	}
	frames := len(values)
	rows := len(values[0])
	cols := len(values[0][0])
	tensorValues := make([][][]float32, frames)
	for frame := 0; frame < frames; frame++ {
		tensorValues[frame] = make([][]float32, rows)
		for row := 0; row < rows; row++ {
			tensorValues[frame][row] = make([]float32, cols)
			for col := 0; col < cols; col++ {
				tensorValues[frame][row][col] = values[frame][row][col]
			}
		}
	}
	return &Tensor{Frames: frames, Rows: rows, Cols: cols, values: tensorValues}
}

// Copy creates a deep copy of the tensor.
func (t *Tensor) Copy() *Tensor {
	return NewValueTensor3D(t.values)
}

// Equals checks that all the values in two matrices are equivalent.
func (t *Tensor) Equals(other *Tensor) bool {
	if t.Rows != other.Rows || t.Cols != other.Cols {
		return false
	}
	for frame := 0; frame < t.Frames; frame++ {
		for row := 0; row < t.Rows; row++ {
			for col := 0; col < t.Cols; col++ {
				if t.Get(frame, row, col) != other.Get(frame, row, col) {
					return false
				}
			}
		}
	}
	return true
}

// Get retrieves a value at a specific row and column.
func (t *Tensor) Get(frame int, row int, col int) float32 {
	if frame < 0 || frame >= t.Frames || row < 0 || row >= t.Rows || col < 0 || col >= t.Cols {
		log.Panicf("Dimensions out of bounds: (%d, %d, %d)", frame, row, col)
	}
	return t.values[frame][row][col]
}

// GetFrame retrieves one 2D frame of values from the tensor.
func (t *Tensor) GetFrame(frame int) [][]float32 {
	return t.values[frame]
}

// GetAll retrieves all values from the tensor.
func (t *Tensor) GetAll() [][][]float32 {
	return t.values
}

// Sum gets the sum of all values in the tensor.
func (t *Tensor) Sum() float32 {
	sum := float32(0.0)
	for frame := 0; frame < t.Frames; frame++ {
		for row := 0; row < t.Rows; row++ {
			for col := 0; col < t.Cols; col++ {
				sum += t.Get(frame, row, col)
			}
		}
	}
	return sum
}

// Set sets a new value at a specific row and column.
func (t *Tensor) Set(frame int, row int, col int, value float32) error {
	if frame < 0 || frame >= t.Frames || row < 0 || row >= t.Rows || col < 0 || col >= t.Cols {
		return fmt.Errorf("dimensions out of bounds: (%d, %d, %d)", frame, row, col)
	}
	t.values[frame][row][col] = value
	return nil
}

// SetTensor sets all the values of the current tensor to the values of the input tensor.
func (t *Tensor) SetTensor(other *Tensor) error {
	if t.Frames != other.Frames || t.Rows != other.Rows || t.Cols != other.Cols {
		return fmt.Errorf(
			"dimensions must match: (%d, %d, %d) != (%d, %d, %d)",
			t.Frames, t.Rows, t.Cols, other.Frames, other.Rows, other.Cols,
		)
	}
	t.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return other.Get(frame, row, col)
	})
	return nil
}

// SetRandom sets all the values of the current tensor to random values between min and max.
func (t *Tensor) SetRandom(min float32, max float32) error {
	if min >= max {
		return fmt.Errorf("ninimum must be less than maximum: %f >= %f", min, max)
	}
	t.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return rand.Float32()*(max-min) + min
	})
	return nil
}

// Add adds a scalar value to all the values of the tensor.
func (t *Tensor) Add(value float32) {
	t.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return current + value
	})
}

// AddTensor adds the values of the input tensor to the values of the current tensor.
func (t *Tensor) AddTensor(other *Tensor) error {
	if t.Frames != other.Frames || t.Rows != other.Rows || t.Cols != other.Cols {
		return fmt.Errorf(
			"dimensions must match: (%d, %d, %d) != (%d, %d, %d)",
			t.Frames, t.Rows, t.Cols, other.Frames, other.Rows, other.Cols,
		)
	}
	t.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return current + other.Get(frame, row, col)
	})
	return nil
}

// Subtract subtracts a scalar value from all the values of the tensor.
func (t *Tensor) Subtract(value float32) {
	t.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return current - value
	})
}

// SubtractTensor subtracts the values of the input tensor from the values of the current tensor.
func (t *Tensor) SubtractTensor(other *Tensor) error {
	if t.Frames != other.Frames || t.Rows != other.Rows || t.Cols != other.Cols {
		return fmt.Errorf(
			"dimensions must match: (%d, %d, %d) != (%d, %d, %d)",
			t.Frames, t.Rows, t.Cols, other.Frames, other.Rows, other.Cols,
		)
	}
	t.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return current - other.Get(frame, row, col)
	})
	return nil
}

// Scale multiplies all the values of the current tensor by a scalar value.
func (t *Tensor) Scale(value float32) {
	t.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return current * value
	})
}

// ScaleTensor multiplies all the values of the current tensor by the values of the input tensor.
func (t *Tensor) ScaleTensor(other *Tensor) error {
	if t.Frames != other.Frames || t.Rows != other.Rows || t.Cols != other.Cols {
		return fmt.Errorf(
			"dimensions must match: (%d, %d, %d) != (%d, %d, %d)",
			t.Frames, t.Rows, t.Cols, other.Frames, other.Rows, other.Cols,
		)
	}
	t.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return current * other.Get(frame, row, col)
	})
	return nil
}

// ApplyFunction applies the input function to all the values of the tensor.
func (t *Tensor) ApplyFunction(function func(float32, int, int, int) float32) {
	for frame := 0; frame < t.Frames; frame++ {
		for row := 0; row < t.Rows; row++ {
			for col := 0; col < t.Cols; col++ {
				newValue := function(t.Get(frame, row, col), frame, row, col)
				t.Set(frame, row, col, newValue)
			}
		}
	}
}

// String creates a string representation of the tensor.
func (t *Tensor) String() string {
	str := ""
	for frame := 0; frame < t.Frames; frame++ {
		for row := 0; row < t.Rows; row++ {
			for col := 0; col < t.Cols; col++ {
				str += fmt.Sprintf("%.4f", t.Get(frame, row, col))
				if col == t.Cols-1 {
					str += "\n"
				} else {
					str += " "
				}
			}
			if row == t.Rows-1 {
				str += "\n"
			}
		}
	}
	return str
}

// MatrixMultiply multiplies two matrices across the frames of two tensors.
func MatrixMultiply(source1 *Tensor, source2 *Tensor, dest *Tensor) (*Tensor, error) {
	if source1.Frames != source2.Frames {
		return nil, fmt.Errorf("tensor frame lengths do not match: %d != %d", source1.Frames, source2.Frames)
	}
	if source1.Cols != source2.Rows {
		return nil, fmt.Errorf("columns of first must match rows of second: %d != %d", source1.Cols, source2.Rows)
	}
	var result *Tensor
	if dest != nil {
		if dest.Frames != source1.Frames || dest.Rows != source1.Rows || dest.Cols != source2.Cols {
			return nil, fmt.Errorf(
				"invalid dest dimensions: (%d, %d, %d) != (%d, %d, %d)",
				dest.Frames, dest.Rows, dest.Cols, source1.Frames, source1.Rows, source2.Cols,
			)
		}
		result = dest
	} else {
		result = NewEmptyTensor3D(source1.Frames, source1.Rows, source2.Cols)
	}
	for frame := 0; frame < source1.Frames; frame++ {
		for row := 0; row < source1.Rows; row++ {
			for col := 0; col < source2.Cols; col++ {
				sum := float32(0.0)
				for i := 0; i < source1.Cols; i++ {
					sum += source1.Get(frame, row, i) * source2.Get(frame, i, col)
				}
				result.Set(frame, row, col, sum)
			}
		}
	}
	return result, nil
}

// MatrixTranspose transposes all the matrices across the frames of a tensor.
func MatrixTranspose(source *Tensor, dest *Tensor) (*Tensor, error) {
	var result *Tensor
	if dest != nil {
		if dest.Frames != source.Frames || dest.Rows != source.Cols || dest.Cols != source.Rows {
			return nil, fmt.Errorf(
				"invalid dest dimensions: (%d, %d, %d) != (%d, %d, %d)",
				dest.Frames, dest.Rows, dest.Cols, source.Frames, source.Cols, source.Rows,
			)
		}
		result = dest
	} else {
		result = NewEmptyTensor3D(source.Frames, source.Cols, source.Rows)
	}
	result.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return source.Get(frame, col, row)
	})
	return result, nil
}
