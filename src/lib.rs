use std::ops::Mul;

const EPSILON: f32 = 1e-6;

#[derive(Debug, Clone, Copy)]
pub struct Matrix<const H: usize, const W: usize> {
    pub data: [[f32; W]; H],
}

impl<const H: usize, const W: usize> Matrix<H, W> {
    pub const fn new(data: [[f32; W]; H]) -> Self {
        Self { data }
    }

    pub const fn zero() -> Self {
        Self {
            data: [[0.0; W]; H],
        }
    }

    pub fn transpose(&self) -> Matrix<W, H> {
        let mut data = [[0.0; H]; W];
        unsafe {
            for i in 0..H {
                for j in 0..W {
                    *data.get_unchecked_mut(j).get_unchecked_mut(i) =
                        *self.data.get_unchecked(i).get_unchecked(j);
                }
            }
        }
        Matrix { data }
    }

    pub fn reshape<const NH: usize, const NW: usize>(&self) -> Matrix<NH, NW> {
        assert_eq!(H * W, NH * NW, "Cannot reshape matrix");

        let mut data = [[0.0; NW]; NH];
        unsafe {
            for i in 0..NH {
                for j in 0..NW {
                    let index = i * NW + j;
                    let row = index / W;
                    let col = index % W;
                    *data.get_unchecked_mut(i).get_unchecked_mut(j) =
                        *self.data.get_unchecked(row).get_unchecked(col);
                }
            }
        }
        Matrix { data }
    }

    pub fn dot<const N: usize>(&self, rhs: &Matrix<W, N>) -> Matrix<H, N> {
        let mut data = [[0.0; N]; H];
        unsafe {
            for i in 0..H {
                for j in 0..N {
                    for k in 0..W {
                        *data.get_unchecked_mut(i).get_unchecked_mut(j) +=
                            *self.data.get_unchecked(i).get_unchecked(k)
                                * *rhs.data.get_unchecked(k).get_unchecked(j);
                    }
                }
            }
        }
        Matrix { data }
    }

    pub fn dot_vector<'a, R>(&self, rhs: &'a R) -> Option<Matrix<1, H>>
    where
        &'a [f32; H]: From<&'a R>,
    {
        let rhs = <&[f32; H]>::from(rhs);

        let mut data = [0.0; H];
        for i in 0..H {
            for j in 0..W {
                data[i] += self.data[i][j] * rhs[j];
            }
        }

        Some(Matrix { data: [data] })
    }
}

impl<const N: usize> Matrix<N, N> {
    // solve A * X = B for X
    pub fn solve<'a, R>(&self, rhs: &'a R) -> Option<Matrix<1, N>>
    where
        &'a [f32; N]: From<&'a R>,
    {
        let inv = self.inverse()?;
        inv.dot_vector(rhs)
    }

    pub fn identity() -> Self {
        let mut data = [[0.0; N]; N];
        unsafe {
            for i in 0..N {
                *data.get_unchecked_mut(i).get_unchecked_mut(i) = 1.0;
            }
        }
        Self { data }
    }

    pub fn inverse(&self) -> Option<Matrix<N, N>> {
        let mut data = self.data;
        let mut inv = Self::identity().data;

        for i in 0..N {
            let mut pivot = data[i][i];
            if pivot == 0.0 {
                for j in i + 1..N {
                    if data[j][i] != 0.0 {
                        data.swap(i, j);
                        inv.swap(i, j);
                        pivot = data[i][i];
                        break;
                    }
                }
            }

            if pivot == 0.0 {
                return None;
            }

            for j in 0..N {
                data[i][j] /= pivot;
                inv[i][j] /= pivot;
            }

            for j in 0..N {
                if i == j {
                    continue;
                }

                let factor = data[j][i];
                for k in 0..N {
                    data[j][k] -= factor * data[i][k];
                    inv[j][k] -= factor * inv[i][k];
                }
            }
        }

        Some(Matrix { data: inv })
    }
}

impl<const H: usize, const W: usize> From<[[f32; W]; H]> for Matrix<H, W> {
    fn from(data: [[f32; W]; H]) -> Self {
        Self { data }
    }
}

impl<const H: usize, const W: usize> From<Matrix<H, W>> for [[f32; W]; H] {
    fn from(matrix: Matrix<H, W>) -> Self {
        matrix.data
    }
}

impl<const W: usize> From<[f32; W]> for Matrix<1, W> {
    fn from(data: [f32; W]) -> Self {
        Self { data: [data] }
    }
}

impl<const W: usize> From<Matrix<1, W>> for [f32; W] {
    fn from(matrix: Matrix<1, W>) -> Self {
        matrix.data[0]
    }
}

impl<'a, const W: usize> From<&'a Matrix<1, W>> for &'a [f32; W] {
    fn from(matrix: &'a Matrix<1, W>) -> Self {
        &matrix.data[0]
    }
}

impl<const H: usize, const W: usize, const N: usize> Mul<Matrix<W, N>> for Matrix<H, W> {
    type Output = Matrix<H, N>;

    fn mul(self, rhs: Matrix<W, N>) -> Self::Output {
        self.dot(&rhs)
    }
}

impl<const H: usize, const W: usize> std::ops::Index<usize> for Matrix<H, W> {
    type Output = [f32; W];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const H: usize, const W: usize> std::ops::IndexMut<usize> for Matrix<H, W> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<const W: usize, const H: usize> std::fmt::Display for Matrix<W, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for row in self.data.iter() {
            for cell in row.iter() {
                write!(f, "{:.2} ", cell)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl<const H: usize, const W: usize> std::cmp::PartialEq for Matrix<H, W> {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..H {
            for j in 0..W {
                if (self.data[i][j] - other.data[i][j]).abs() > EPSILON {
                    return false;
                }
            }
        }
        true
    }
}

#[test]
fn test_matrix() {
    let m1 = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let m2 = Matrix::<1, 3>::new([[1.0, 2.0, 3.0]]);

    let m3 = m1.dot(&m2.transpose()).transpose();
    let m4 = (m1 * m2.transpose()).transpose();
    assert_eq!(m3, m4);

    let m5 = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let m6 = m5.inverse();
    assert_eq!(m6, None);

    let m7 = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]);
    let m8 = m7.inverse().unwrap();

    let m9 = m7 * m8;
    assert_eq!(m9, Matrix::<3, 3>::identity());

    let m10 = Matrix::<3, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 5.0, 9.0]]);
    let m11 = Matrix::<1, 3>::new([[1.0, 2.0, 3.0]]);
    let m12 = m10.solve(&m11).unwrap();
    assert_eq!(m12, Matrix::<1, 3>::new([[-5.9604645e-8, 0.0, 0.3333333]]));
}
