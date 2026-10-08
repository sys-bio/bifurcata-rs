//! Complex arithmetic (specification §3A.1).
//!
//! A small type of our own rather than `num_complex`, for one reason that
//! matters: division uses **Smith's algorithm**, so dividing by a number with
//! components near 1e200 does not overflow in `|b|²` and return NaN. The
//! normal-form formulas of §6.4 divide by eigenvector inner products whose
//! scale is not under our control.

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

/// `re + i·im`
pub const fn cx(re: f64, im: f64) -> Complex {
    Complex { re, im }
}

pub const ZERO: Complex = cx(0.0, 0.0);
pub const ONE: Complex = cx(1.0, 0.0);
pub const I: Complex = cx(0.0, 1.0);

impl Complex {
    /// `|z|`, without overflow for large components.
    pub fn modulus(self) -> f64 {
        self.re.hypot(self.im)
    }

    pub fn modulus_squared(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn conjugate(self) -> Complex {
        cx(self.re, -self.im)
    }

    pub fn reciprocal(self) -> Complex {
        ONE / self
    }

    /// The argument in (−π, π].
    pub fn argument(self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn scale(self, s: f64) -> Complex {
        cx(self.re * s, self.im * s)
    }

    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl From<f64> for Complex {
    fn from(re: f64) -> Complex {
        cx(re, 0.0)
    }
}

impl Add for Complex {
    type Output = Complex;
    fn add(self, b: Complex) -> Complex {
        cx(self.re + b.re, self.im + b.im)
    }
}

impl Sub for Complex {
    type Output = Complex;
    fn sub(self, b: Complex) -> Complex {
        cx(self.re - b.re, self.im - b.im)
    }
}

impl Mul for Complex {
    type Output = Complex;
    fn mul(self, b: Complex) -> Complex {
        cx(self.re * b.re - self.im * b.im, self.re * b.im + self.im * b.re)
    }
}

impl Div for Complex {
    type Output = Complex;
    /// Smith's algorithm (Smith 1962): divide through by the larger component
    /// of the divisor, so nothing is squared that could overflow.
    fn div(self, b: Complex) -> Complex {
        if b.re.abs() >= b.im.abs() {
            let r = b.im / b.re;
            let d = b.re + b.im * r;
            cx((self.re + self.im * r) / d, (self.im - self.re * r) / d)
        } else {
            let r = b.re / b.im;
            let d = b.re * r + b.im;
            cx((self.re * r + self.im) / d, (self.im * r - self.re) / d)
        }
    }
}

impl Neg for Complex {
    type Output = Complex;
    fn neg(self) -> Complex {
        cx(-self.re, -self.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, b: Complex) {
        *self = *self + b;
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, b: Complex) {
        *self = *self - b;
    }
}

impl MulAssign for Complex {
    fn mul_assign(&mut self, b: Complex) {
        *self = *self * b;
    }
}

/// The Hermitian inner product `⟨p, q⟩ = Σ conj(p_i) q_i` — conjugate-linear
/// in its **first** argument, the convention the normal-form formulas of §6.4
/// assume.
pub fn dot(p: &[Complex], q: &[Complex]) -> Complex {
    assert_eq!(p.len(), q.len(), "dot: vectors of different lengths");
    p.iter().zip(q).fold(ZERO, |s, (a, b)| s + a.conjugate() * *b)
}

/// The Euclidean norm `sqrt(Σ |z_i|²)`.
pub fn norm2(v: &[Complex]) -> f64 {
    v.iter().map(|z| z.modulus_squared()).sum::<f64>().sqrt()
}

/// Multiply every element by `s`.
pub fn scale(v: &mut [Complex], s: Complex) {
    for z in v {
        *z *= s;
    }
}
