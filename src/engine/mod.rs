use anyhow::{bail, Result};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_NODE_ID: AtomicUsize = AtomicUsize::new(1);

#[derive(Clone, Debug, PartialEq)]
struct Storage {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Storage {
    fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        let expected = Self::numel_from_shape(&shape)?;
        if data.len() != expected {
            bail!(
                "data length {} does not match shape {:?} (numel={expected})",
                data.len(),
                shape
            );
        }
        Ok(Self { data, shape })
    }

    fn scalar(value: f32) -> Self {
        Self {
            data: vec![value],
            shape: vec![1],
        }
    }

    fn numel_from_shape(shape: &[usize]) -> Result<usize> {
        if shape.is_empty() {
            return Ok(1);
        }
        shape.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or_else(|| anyhow::anyhow!("shape {:?} overflows numel", shape))
        })
    }

    fn numel(&self) -> usize {
        self.data.len()
    }

    fn is_scalar(&self) -> bool {
        self.numel() == 1
    }

    fn scalar_value(&self) -> Result<f32> {
        if !self.is_scalar() {
            bail!("tensor with shape {:?} is not scalar", self.shape);
        }
        Ok(self.data[0])
    }
}

#[derive(Clone, Debug)]
enum OpKind {
    Leaf,
    Add,
    Sub,
    Mul,
    MatMul,
    Relu,
    Tanh,
    Sum,
    Mean,
}

struct Node {
    id: usize,
    storage: Storage,
    grad: Option<Storage>,
    requires_grad: bool,
    parents: Vec<Tensor>,
    op: OpKind,
}

#[derive(Clone)]
pub struct Tensor(Rc<RefCell<Node>>);

impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Result<Self> {
        let storage = Storage::from_vec(data, shape)?;
        Ok(Self::leaf(storage, requires_grad))
    }

    pub fn scalar(value: f32, requires_grad: bool) -> Self {
        Self::leaf(Storage::scalar(value), requires_grad)
    }

    fn leaf(storage: Storage, requires_grad: bool) -> Self {
        Self(Rc::new(RefCell::new(Node {
            id: NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed),
            storage,
            grad: None,
            requires_grad,
            parents: Vec::new(),
            op: OpKind::Leaf,
        })))
    }

    fn from_op(storage: Storage, parents: Vec<Tensor>, op: OpKind) -> Self {
        let requires_grad = parents.iter().any(|p| p.requires_grad());
        Self(Rc::new(RefCell::new(Node {
            id: NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed),
            storage,
            grad: None,
            requires_grad,
            parents,
            op,
        })))
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.borrow().storage.shape.clone()
    }

    pub fn data(&self) -> Vec<f32> {
        self.0.borrow().storage.data.clone()
    }

    pub fn grad(&self) -> Option<Vec<f32>> {
        self.0.borrow().grad.as_ref().map(|g| g.data.clone())
    }

    pub fn item(&self) -> Result<f32> {
        self.0.borrow().storage.scalar_value()
    }

    pub fn grad_item(&self) -> Result<f32> {
        let grad = self
            .0
            .borrow()
            .grad
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no grad available"))?
            .clone();
        grad.scalar_value()
    }

    pub fn requires_grad(&self) -> bool {
        self.0.borrow().requires_grad
    }

    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = None;
    }

    pub fn set_data(&self, data: Vec<f32>) -> Result<()> {
        let mut node = self.0.borrow_mut();
        if data.len() != node.storage.data.len() {
            bail!(
                "new data len {} does not match tensor numel {}",
                data.len(),
                node.storage.data.len()
            );
        }
        node.storage.data = data;
        Ok(())
    }

    pub fn step(&self, lr: f32) -> Result<()> {
        let mut node = self.0.borrow_mut();
        if !node.requires_grad {
            return Ok(());
        }
        let grad = node
            .grad
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("parameter step called before backward"))?
            .clone();
        if grad.shape != node.storage.shape {
            bail!(
                "grad shape {:?} does not match parameter shape {:?}",
                grad.shape,
                node.storage.shape
            );
        }
        for (value, g) in node.storage.data.iter_mut().zip(grad.data.iter()) {
            *value -= lr * g;
        }
        Ok(())
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let a = self.storage();
        let b = other.storage();
        let out = binary_forward(&a, &b, |x, y| x + y)?;
        Ok(Tensor::from_op(
            out,
            vec![self.clone(), other.clone()],
            OpKind::Add,
        ))
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let a = self.storage();
        let b = other.storage();
        let out = binary_forward(&a, &b, |x, y| x - y)?;
        Ok(Tensor::from_op(
            out,
            vec![self.clone(), other.clone()],
            OpKind::Sub,
        ))
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let a = self.storage();
        let b = other.storage();
        let out = binary_forward(&a, &b, |x, y| x * y)?;
        Ok(Tensor::from_op(
            out,
            vec![self.clone(), other.clone()],
            OpKind::Mul,
        ))
    }

    pub fn relu(&self) -> Result<Tensor> {
        let input = self.storage();
        let out = Storage {
            data: input
                .data
                .iter()
                .map(|v| if *v > 0.0 { *v } else { 0.0 })
                .collect(),
            shape: input.shape,
        };
        Ok(Tensor::from_op(out, vec![self.clone()], OpKind::Relu))
    }

    pub fn tanh(&self) -> Result<Tensor> {
        let input = self.storage();
        let out = Storage {
            data: input.data.iter().map(|v| v.tanh()).collect(),
            shape: input.shape,
        };
        Ok(Tensor::from_op(out, vec![self.clone()], OpKind::Tanh))
    }

    pub fn sum(&self) -> Result<Tensor> {
        let input = self.storage();
        let value = input.data.iter().sum::<f32>();
        Ok(Tensor::from_op(
            Storage::scalar(value),
            vec![self.clone()],
            OpKind::Sum,
        ))
    }

    pub fn mean(&self) -> Result<Tensor> {
        let input = self.storage();
        let count = input.numel() as f32;
        let value = input.data.iter().sum::<f32>() / count;
        Ok(Tensor::from_op(
            Storage::scalar(value),
            vec![self.clone()],
            OpKind::Mean,
        ))
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let a = self.storage();
        let b = other.storage();
        let out = matmul2d(&a, &b)?;
        Ok(Tensor::from_op(
            out,
            vec![self.clone(), other.clone()],
            OpKind::MatMul,
        ))
    }

    pub fn backward(&self) -> Result<()> {
        let output = self.storage();
        if !output.is_scalar() {
            bail!(
                "backward expects scalar output, got shape {:?}",
                output.shape
            );
        }

        let order = topo_sort(self);
        accumulate_grad(self, Storage::scalar(1.0))?;

        for tensor in order.into_iter().rev() {
            let (op, parents, grad, node_storage) = {
                let node = tensor.0.borrow();
                (
                    node.op.clone(),
                    node.parents.clone(),
                    node.grad.clone(),
                    node.storage.clone(),
                )
            };
            let Some(grad) = grad else {
                continue;
            };

            match op {
                OpKind::Leaf => {}
                OpKind::Add => backward_add(&parents, &grad, &node_storage)?,
                OpKind::Sub => backward_sub(&parents, &grad, &node_storage)?,
                OpKind::Mul => backward_mul(&parents, &grad)?,
                OpKind::MatMul => backward_matmul(&parents, &grad)?,
                OpKind::Relu => backward_relu(&parents, &grad)?,
                OpKind::Tanh => backward_tanh(&parents, &grad, &node_storage)?,
                OpKind::Sum => backward_sum(&parents, &grad)?,
                OpKind::Mean => backward_mean(&parents, &grad)?,
            }
        }

        Ok(())
    }

    fn storage(&self) -> Storage {
        self.0.borrow().storage.clone()
    }
}

fn binary_forward<F>(a: &Storage, b: &Storage, f: F) -> Result<Storage>
where
    F: Fn(f32, f32) -> f32,
{
    if a.shape == b.shape {
        return Ok(Storage {
            data: a
                .data
                .iter()
                .zip(b.data.iter())
                .map(|(x, y)| f(*x, *y))
                .collect(),
            shape: a.shape.clone(),
        });
    }
    if a.is_scalar() {
        let scalar = a.data[0];
        return Ok(Storage {
            data: b.data.iter().map(|x| f(scalar, *x)).collect(),
            shape: b.shape.clone(),
        });
    }
    if b.is_scalar() {
        let scalar = b.data[0];
        return Ok(Storage {
            data: a.data.iter().map(|x| f(*x, scalar)).collect(),
            shape: a.shape.clone(),
        });
    }
    bail!(
        "shape mismatch for binary op: left={:?}, right={:?}; only equal shape or scalar broadcast is supported",
        a.shape,
        b.shape
    );
}

fn reduce_grad_to_parent(
    grad: &[f32],
    output_shape: &[usize],
    parent: &Storage,
) -> Result<Storage> {
    if parent.shape == output_shape {
        return Storage::from_vec(grad.to_vec(), parent.shape.clone());
    }
    if parent.is_scalar() {
        return Ok(Storage::scalar(grad.iter().sum()));
    }
    bail!(
        "cannot reduce grad from shape {:?} to parent shape {:?}",
        output_shape,
        parent.shape
    );
}

fn backward_add(parents: &[Tensor], grad: &Storage, node_storage: &Storage) -> Result<()> {
    let a = parents[0].storage();
    let b = parents[1].storage();
    let ga = reduce_grad_to_parent(&grad.data, &node_storage.shape, &a)?;
    let gb = reduce_grad_to_parent(&grad.data, &node_storage.shape, &b)?;
    accumulate_grad(&parents[0], ga)?;
    accumulate_grad(&parents[1], gb)?;
    Ok(())
}

fn backward_sub(parents: &[Tensor], grad: &Storage, node_storage: &Storage) -> Result<()> {
    let a = parents[0].storage();
    let b = parents[1].storage();
    let ga = reduce_grad_to_parent(&grad.data, &node_storage.shape, &a)?;
    let neg_grad: Vec<f32> = grad.data.iter().map(|v| -*v).collect();
    let gb = reduce_grad_to_parent(&neg_grad, &node_storage.shape, &b)?;
    accumulate_grad(&parents[0], ga)?;
    accumulate_grad(&parents[1], gb)?;
    Ok(())
}

fn backward_mul(parents: &[Tensor], grad: &Storage) -> Result<()> {
    let a = parents[0].storage();
    let b = parents[1].storage();

    if a.shape == b.shape {
        let ga = Storage::from_vec(
            grad.data
                .iter()
                .zip(b.data.iter())
                .map(|(g, bv)| g * bv)
                .collect(),
            a.shape.clone(),
        )?;
        let gb = Storage::from_vec(
            grad.data
                .iter()
                .zip(a.data.iter())
                .map(|(g, av)| g * av)
                .collect(),
            b.shape.clone(),
        )?;
        accumulate_grad(&parents[0], ga)?;
        accumulate_grad(&parents[1], gb)?;
        return Ok(());
    }

    if a.is_scalar() {
        let a_value = a.data[0];
        let ga_scalar = grad
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(g, bv)| g * bv)
            .sum::<f32>();
        let gb = Storage::from_vec(
            grad.data.iter().map(|g| g * a_value).collect(),
            b.shape.clone(),
        )?;
        accumulate_grad(&parents[0], Storage::scalar(ga_scalar))?;
        accumulate_grad(&parents[1], gb)?;
        return Ok(());
    }

    if b.is_scalar() {
        let b_value = b.data[0];
        let ga = Storage::from_vec(
            grad.data.iter().map(|g| g * b_value).collect(),
            a.shape.clone(),
        )?;
        let gb_scalar = grad
            .data
            .iter()
            .zip(a.data.iter())
            .map(|(g, av)| g * av)
            .sum::<f32>();
        accumulate_grad(&parents[0], ga)?;
        accumulate_grad(&parents[1], Storage::scalar(gb_scalar))?;
        return Ok(());
    }

    bail!(
        "shape mismatch for mul backward: left={:?} right={:?}",
        a.shape,
        b.shape
    )
}

fn backward_relu(parents: &[Tensor], grad: &Storage) -> Result<()> {
    let input = parents[0].storage();
    let in_grad = Storage::from_vec(
        grad.data
            .iter()
            .zip(input.data.iter())
            .map(|(g, v)| if *v > 0.0 { *g } else { 0.0 })
            .collect(),
        input.shape.clone(),
    )?;
    accumulate_grad(&parents[0], in_grad)?;
    Ok(())
}

fn backward_tanh(parents: &[Tensor], grad: &Storage, node_storage: &Storage) -> Result<()> {
    let in_grad = Storage::from_vec(
        grad.data
            .iter()
            .zip(node_storage.data.iter())
            .map(|(g, out)| g * (1.0 - out * out))
            .collect(),
        node_storage.shape.clone(),
    )?;
    accumulate_grad(&parents[0], in_grad)?;
    Ok(())
}

fn backward_sum(parents: &[Tensor], grad: &Storage) -> Result<()> {
    let input = parents[0].storage();
    let scale = grad.scalar_value()?;
    let in_grad = Storage::from_vec(vec![scale; input.numel()], input.shape.clone())?;
    accumulate_grad(&parents[0], in_grad)?;
    Ok(())
}

fn backward_mean(parents: &[Tensor], grad: &Storage) -> Result<()> {
    let input = parents[0].storage();
    let scale = grad.scalar_value()? / (input.numel() as f32);
    let in_grad = Storage::from_vec(vec![scale; input.numel()], input.shape.clone())?;
    accumulate_grad(&parents[0], in_grad)?;
    Ok(())
}

fn matmul2d(a: &Storage, b: &Storage) -> Result<Storage> {
    let (m, k_left) = shape2d(&a.shape)?;
    let (k_right, n) = shape2d(&b.shape)?;
    if k_left != k_right {
        bail!(
            "matmul shape mismatch: left={:?}, right={:?}",
            a.shape,
            b.shape
        );
    }
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k_left {
                sum += a.data[i * k_left + p] * b.data[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    Storage::from_vec(out, vec![m, n])
}

fn backward_matmul(parents: &[Tensor], grad: &Storage) -> Result<()> {
    let a = parents[0].storage();
    let b = parents[1].storage();
    let (m, k) = shape2d(&a.shape)?;
    let (k_b, n) = shape2d(&b.shape)?;
    let (g_m, g_n) = shape2d(&grad.shape)?;

    if k != k_b || m != g_m || n != g_n {
        bail!(
            "matmul backward shape mismatch: a={:?}, b={:?}, grad={:?}",
            a.shape,
            b.shape,
            grad.shape
        );
    }

    let mut grad_a = vec![0.0f32; m * k];
    for i in 0..m {
        for p in 0..k {
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += grad.data[i * n + j] * b.data[p * n + j];
            }
            grad_a[i * k + p] = sum;
        }
    }

    let mut grad_b = vec![0.0f32; k * n];
    for p in 0..k {
        for j in 0..n {
            let mut sum = 0.0f32;
            for i in 0..m {
                sum += a.data[i * k + p] * grad.data[i * n + j];
            }
            grad_b[p * n + j] = sum;
        }
    }

    accumulate_grad(&parents[0], Storage::from_vec(grad_a, a.shape.clone())?)?;
    accumulate_grad(&parents[1], Storage::from_vec(grad_b, b.shape.clone())?)?;
    Ok(())
}

fn shape2d(shape: &[usize]) -> Result<(usize, usize)> {
    if shape.len() != 2 {
        bail!("expected 2D shape, got {:?}", shape);
    }
    Ok((shape[0], shape[1]))
}

fn accumulate_grad(tensor: &Tensor, incoming: Storage) -> Result<()> {
    let mut node = tensor.0.borrow_mut();
    if !node.requires_grad {
        return Ok(());
    }
    if incoming.shape != node.storage.shape {
        bail!(
            "incoming grad shape {:?} does not match tensor shape {:?}",
            incoming.shape,
            node.storage.shape
        );
    }

    if let Some(existing) = node.grad.as_mut() {
        for (dst, src) in existing.data.iter_mut().zip(incoming.data.iter()) {
            *dst += src;
        }
    } else {
        node.grad = Some(incoming);
    }
    Ok(())
}

fn topo_sort(root: &Tensor) -> Vec<Tensor> {
    fn dfs(node: &Tensor, seen: &mut HashSet<usize>, out: &mut Vec<Tensor>) {
        let (id, parents) = {
            let n = node.0.borrow();
            (n.id, n.parents.clone())
        };
        if !seen.insert(id) {
            return;
        }
        for parent in parents {
            dfs(&parent, seen, out);
        }
        out.push(node.clone());
    }

    let mut seen = HashSet::new();
    let mut out = Vec::new();
    dfs(root, &mut seen, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::Tensor;
    use anyhow::Result;

    #[test]
    fn scalar_autograd_matches_expected_derivative() -> Result<()> {
        let x = Tensor::scalar(3.0, true);
        let y = x.mul(&x)?.add(&x)?;
        y.backward()?;

        let grad = x.grad_item()?;
        assert!((grad - 7.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn matmul_mean_backward_is_correct() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)?;
        let b = Tensor::from_vec(vec![5.0, 6.0], vec![2, 1], true)?;

        let out = a.matmul(&b)?.mean()?;
        out.backward()?;

        let ga = a.grad().expect("a grad");
        let gb = b.grad().expect("b grad");

        let expected_a = [2.5, 3.0, 2.5, 3.0];
        for (got, exp) in ga.iter().zip(expected_a.iter()) {
            assert!((got - exp).abs() < 1e-5);
        }

        let expected_b = [2.0, 3.0];
        for (got, exp) in gb.iter().zip(expected_b.iter()) {
            assert!((got - exp).abs() < 1e-5);
        }

        Ok(())
    }

    #[test]
    fn linear_regression_demo_learns_reasonable_params() -> Result<()> {
        let x = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![4], false)?;
        let y = Tensor::from_vec(vec![1.0, 3.0, 5.0, 7.0], vec![4], false)?;
        let w = Tensor::scalar(0.0, true);
        let b = Tensor::scalar(0.0, true);

        let initial_loss = mse(&x, &y, &w, &b)?.item()?;

        for _ in 0..300 {
            w.zero_grad();
            b.zero_grad();

            let loss = mse(&x, &y, &w, &b)?;
            loss.backward()?;

            w.step(0.1)?;
            b.step(0.1)?;
        }

        let final_loss = mse(&x, &y, &w, &b)?.item()?;
        assert!(final_loss < initial_loss * 0.01);
        assert!((w.item()? - 2.0).abs() < 1e-2);
        assert!((b.item()? - 3.0).abs() < 1e-2);
        Ok(())
    }

    fn mse(x: &Tensor, y: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
        let pred = x.mul(w)?.add(b)?;
        let diff = pred.sub(y)?;
        diff.mul(&diff)?.mean()
    }
}
