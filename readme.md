# mat

a extremely simple and tiny matrix library for rust

it just provides a simple matrix struct and some basic operations

## Examples

```rust
let matrix1 = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 5.0, 9.0]]);
let vector = Vector::<3>::new([1.0, 2.0, 3.0]);

let result: Vector<3> = matrix1.solve(&vector).expect("failed to solve");

let matrix2 = Matrix3::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 5.0, 9.0]]);
let vector = Vector3::new([1.0, 2.0, 3.0]);

let result = matrix2 * vector; //! compile time error, because the matrix and vector are not compatible

// indexing
let row = matrix1[0];
let item = matrix1[(0, 0)];
```
