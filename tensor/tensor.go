package tensor

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
func (tensor *Tensor) Copy() *Tensor {
	return NewValueTensor3D(tensor.values)
}

// Equals checks that all the values in two matrices are equivalent.
func (tensor *Tensor) Equals(other *Tensor) bool {
	if tensor.Rows != other.Rows || tensor.Cols != other.Cols {
		return false
	}
	for frame := 0; frame < tensor.Frames; frame++ {
		for row := 0; row < tensor.Rows; row++ {
			for col := 0; col < tensor.Cols; col++ {
				if tensor.Get(frame, row, col) != other.Get(frame, row, col) {
					return false
				}
			}
		}
	}
	return true
}

// Get retrieves a value at a specific row and column.
func (tensor *Tensor) Get(frame int, row int, col int) float32 {
	if frame < 0 || frame >= tensor.Frames || row < 0 || row >= tensor.Rows || col < 0 || col >= tensor.Cols {
		log.Panicf("Dimensions out of bounds: (%d, %d, %d)", frame, row, col)
	}
	return tensor.values[frame][row][col]
}

// GetFrame retrieves one 2D frame of values from the tensor.
func (tensor *Tensor) GetFrame(frame int) [][]float32 {
	return tensor.values[frame]
}

// GetAll retrieves all values from the tensor.
func (tensor *Tensor) GetAll() [][][]float32 {
	return tensor.values
}

// Sum gets the sum of all values in the tensor.
func (tensor *Tensor) Sum() float32 {
	sum := float32(0.0)
	for frame := 0; frame < tensor.Frames; frame++ {
		for row := 0; row < tensor.Rows; row++ {
			for col := 0; col < tensor.Cols; col++ {
				sum += tensor.Get(frame, row, col)
			}
		}
	}
	return sum
}

// Set sets a new value at a specific row and column.
func (tensor *Tensor) Set(frame int, row int, col int, value float32) error {
	if frame < 0 || frame >= tensor.Frames || row < 0 || row >= tensor.Rows || col < 0 || col >= tensor.Cols {
		return fmt.Errorf("Dimensions out of bounds: (%d, %d, %d)", frame, row, col)
	}
	tensor.values[frame][row][col] = value
	return nil
}

// SetAll sets all the values of the current tensor to the values of the input tensor.
func (tensor *Tensor) SetAll(other *Tensor) error {
	if tensor.Frames != other.Frames || tensor.Rows != other.Rows || tensor.Cols != other.Cols {
		return fmt.Errorf(
			"Dimensions must match: (%d, %d, %d) != (%d, %d, %d)",
			tensor.Frames, tensor.Rows, tensor.Cols, other.Frames, other.Rows, other.Cols,
		)
	}
	tensor.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return other.Get(frame, row, col)
	})
	return nil
}

// SetRandom sets all the values of the current tensor to random values between min and max.
func (tensor *Tensor) SetRandom(min float32, max float32) error {
	if min >= max {
		return fmt.Errorf("Minimum must be less than maximum: %f >= %f", min, max)
	}
	tensor.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return rand.Float32()*(max-min) + min
	})
	return nil
}

// Add adds a scalar value to all the values of the tensor.
func (tensor *Tensor) Add(value float32) {
	tensor.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return current + value
	})
}

// AddTensor adds the values of the input tensor to the values of the current tensor.
func (tensor *Tensor) AddTensor(other *Tensor) error {
	if tensor.Frames != other.Frames || tensor.Rows != other.Rows || tensor.Cols != other.Cols {
		return fmt.Errorf(
			"Dimensions must match: (%d, %d, %d) != (%d, %d, %d)",
			tensor.Frames, tensor.Rows, tensor.Cols, other.Frames, other.Rows, other.Cols,
		)
	}
	tensor.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return current + other.Get(frame, row, col)
	})
	return nil
}

// Subtract subtracts a scalar value from all the values of the tensor.
func (tensor *Tensor) Subtract(value float32) {
	tensor.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return current - value
	})
}

// SubtractTensor subtracts the values of the input tensor from the values of the current tensor.
func (tensor *Tensor) SubtractTensor(other *Tensor) error {
	if tensor.Frames != other.Frames || tensor.Rows != other.Rows || tensor.Cols != other.Cols {
		return fmt.Errorf(
			"Dimensions must match: (%d, %d, %d) != (%d, %d, %d)",
			tensor.Frames, tensor.Rows, tensor.Cols, other.Frames, other.Rows, other.Cols,
		)
	}
	tensor.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return current - other.Get(frame, row, col)
	})
	return nil
}

// Scale multiplies all the values of the current tensor by a scalar value.
func (tensor *Tensor) Scale(value float32) {
	tensor.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return current * value
	})
}

// ScaleTensor multiplies all the values of the current tensor by the values of the input tensor.
func (tensor *Tensor) ScaleTensor(other *Tensor) error {
	if tensor.Frames != other.Frames || tensor.Rows != other.Rows || tensor.Cols != other.Cols {
		return fmt.Errorf(
			"Dimensions must match: (%d, %d, %d) != (%d, %d, %d)",
			tensor.Frames, tensor.Rows, tensor.Cols, other.Frames, other.Rows, other.Cols,
		)
	}
	tensor.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return current * other.Get(frame, row, col)
	})
	return nil
}

// ApplyFunction applies the input function to all the values of the tensor.
func (tensor *Tensor) ApplyFunction(function func(float32, int, int, int) float32) {
	for frame := 0; frame < tensor.Frames; frame++ {
		for row := 0; row < tensor.Rows; row++ {
			for col := 0; col < tensor.Cols; col++ {
				newValue := function(tensor.Get(frame, row, col), frame, row, col)
				tensor.Set(frame, row, col, newValue)
			}
		}
	}
}

// String creates a string representation of the tensor.
func (tensor *Tensor) String() string {
	str := ""
	for frame := 0; frame < tensor.Frames; frame++ {
		for row := 0; row < tensor.Rows; row++ {
			for col := 0; col < tensor.Cols; col++ {
				str += fmt.Sprintf("%.4f", tensor.Get(frame, row, col))
				if col == tensor.Cols-1 {
					str += "\n"
				} else {
					str += " "
				}
			}
			if row == tensor.Rows-1 {
				str += "\n"
			}
		}
	}
	return str
}

// MatrixMultiply multiplies two matrices across the frames of two tensors.
func MatrixMultiply(tensor1 *Tensor, tensor2 *Tensor, target *Tensor) (*Tensor, error) {
	if tensor1.Frames != tensor2.Frames {
		return nil, fmt.Errorf("Tensor frame lengths do not match: %d != %d", tensor1.Frames, tensor2.Frames)
	}
	if tensor1.Cols != tensor2.Rows {
		return nil, fmt.Errorf("Columns of first must match rows of second: %d != %d", tensor1.Cols, tensor2.Rows)
	}
	var result *Tensor
	if target != nil {
		if target.Frames != tensor1.Frames || target.Rows != tensor1.Rows || target.Cols != tensor2.Cols {
			return nil, fmt.Errorf(
				"Invalid target dimensions: (%d, %d, %d) != (%d, %d, %d)",
				target.Frames, target.Rows, target.Cols, tensor1.Frames, tensor1.Rows, tensor2.Cols,
			)
		}
		result = target
	} else {
		result = NewEmptyTensor3D(tensor1.Frames, tensor1.Rows, tensor2.Cols)
	}
	for frame := 0; frame < tensor1.Frames; frame++ {
		for row := 0; row < tensor1.Rows; row++ {
			for col := 0; col < tensor2.Cols; col++ {
				sum := float32(0.0)
				for i := 0; i < tensor1.Cols; i++ {
					sum += tensor1.Get(frame, row, i) * tensor2.Get(frame, i, col)
				}
				result.Set(frame, row, col, sum)
			}
		}
	}
	return result, nil
}

// MatrixTranspose transposes all the matrices across the frames of a tensor.
func MatrixTranspose(tensor *Tensor, target *Tensor) (*Tensor, error) {
	var result *Tensor
	if target != nil {
		if target.Frames != tensor.Frames || target.Rows != tensor.Cols || target.Cols != tensor.Rows {
			return nil, fmt.Errorf(
				"Invalid target dimensions: (%d, %d, %d) != (%d, %d, %d)",
				target.Frames, target.Rows, target.Cols, tensor.Frames, tensor.Cols, tensor.Rows,
			)
		}
		result = target
	} else {
		result = NewEmptyTensor3D(tensor.Frames, tensor.Cols, tensor.Rows)
	}
	result.ApplyFunction(func(current float32, frame int, row int, col int) float32 {
		return tensor.Get(frame, col, row)
	})
	return result, nil
}
