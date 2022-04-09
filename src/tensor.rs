use core::fmt;
use std::ops;

use failure::bail;


pub struct Mat {
    width: usize,
    height: usize,
    data: Vec<f32>
}

impl Mat {
    pub fn constant(consts: &[&[f32]]) -> Result<Mat, failure::Error> {
        if consts.len() == 0 {
            return Ok(Mat{
                width: 0,
                height: 0,
                data: Vec::new()
            })
        }

        let height = consts.len();
        let width = consts[0].len();
        let mut data = Vec::with_capacity(height * width);

        for c in consts {
            if c.len() != width {
                return bail!("All columns need to be same width");
            }

            data.extend(*c);
        }

        return Ok(Mat{
            width: width,
            height: height,
            data: data
        })
    }

    pub fn width(&self) -> usize {
        return self.width
    }

    pub fn height(&self) -> usize {
        return self.height
    }
}

impl ops::Mul for Mat {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut tensor = Mat{
            width: self.width,
            height: self.height,
            data: Vec::with_capacity(self.data.len())
        };

        for i in 0..self.data.len() {
            tensor.data[i] = self.data[i] * rhs.data[i];
        }

        tensor
    }
}

impl<'a, 'b> ops::Mul<&'b Mat> for Mat {
    type Output = Self;

    fn mul(self, rhs: &'b Mat) -> Self::Output {
        let mut tensor = Mat{
            width: self.width,
            height: self.height,
            data: Vec::with_capacity(self.data.len())
        };

        for i in 0..self.data.len() {
            tensor.data[i] = self.data[i] * rhs.data[i];
        }

        tensor
    }
}

impl ops::MulAssign for Mat {
    fn mul_assign(&mut self, rhs: Self) {
        if self.data.len() != rhs.data.len() {
            panic!("Two tensors need to be same size to multiply");
        }

        for i in 0..self.data.len() {
            self.data[i] *= rhs.data[i];
        }
    }
}

impl<'a, 'b> ops::MulAssign<&'b Mat> for Mat {
    fn mul_assign(&mut self, rhs: &'b Mat) {
        if self.data.len() != rhs.data.len() {
            panic!("Two tensors need to be same size to multiply");
        }

        for i in 0..self.data.len() {
            self.data[i] *= rhs.data[i];
        }
    }
}

impl ops::Add for Mat {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!("Two tensors need to be same size to add");
        }

        let mut tensor = Mat{
            width: self.width,
            height: self.height,
            data: Vec::with_capacity(self.data.len()),
        };

        for i in 0..self.data.len() {
            tensor.data[i] = self.data[i] + rhs.data[i];
        }

        tensor
    }
}

impl<'a, 'b> ops::Add<&'b Mat> for Mat {
    type Output = Self;

    fn add(self, rhs: &'b Mat) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!("Two tensors need to be same size to add");
        }

        let mut tensor = Mat{
            width: self.width,
            height: self.height,
            data: Vec::with_capacity(self.data.len()),
        };

        for i in 0..self.data.len() {
            tensor.data[i] = self.data[i] + rhs.data[i];
        }

        tensor
    }
}

impl ops::AddAssign for Mat {
    fn add_assign(&mut self, rhs: Mat) {
        if self.data.len() != rhs.data.len() {
            panic!("Two tensors need to be same size to add");
        }

        for i in 0..self.data.len() {
            self.data[i] += rhs.data[i];
        }
    }
}

impl<'a, 'b> ops::AddAssign<&'b Mat> for Mat {
    fn add_assign(&mut self, rhs: &'b Mat) {
        if self.data.len() != rhs.data.len() {
            panic!("Two tensors need to be same size to add");
        }

        for i in 0..self.data.len() {
            self.data[i] += rhs.data[i];
        }
    }
}

impl fmt::Display for Mat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut i = 0;
        for n in &self.data {
            write!(f, "{} ", n)?;

            i += 1;

            if i >= self.width {
                i = 0;
                write!(f, "\n")?;
            }
        }
        Ok(())
    }
}