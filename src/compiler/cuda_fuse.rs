//! Fuse expressions without moving reads across mutable-buffer writes.
use super::*;

pub(super) fn epilogues(
    g: &Graph,
    order: &[Value],
    outputs: &[Value],
    consumers: &HashMap<Value, Vec<Value>>,
    contractions: &mut HashMap<Value, (Value, Value, Vec<usize>)>,
    skip: &mut HashSet<Value>,
) -> Result<HashMap<Value, Value>> {
    let positions: HashMap<_, _> = order.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut claimed = HashSet::new();
    let mut result = HashMap::new();
    for &start in order {
        if !contractions.contains_key(&start) || claimed.contains(&start) {
            continue;
        }
        // Include small diamonds such as x * sigmoid(x), where x has two
        // internal uses but only one final result. Independent operands are
        // ready at the end; externally observed intermediates remain stored.
        let mut reachable = HashSet::from([start]);
        let mut pending = vec![start];
        let mut candidates = Vec::new();
        while let Some(v) = pending.pop() {
            if outputs.contains(&v) || reachable.len() >= 32 {
                continue;
            }
            for &next in consumers.get(&v).into_iter().flatten() {
                if reachable.len() >= 32 {
                    break;
                }
                let node = g.node(next)?;
                if pointwise(node.op())
                    && node.dtype() == DType::F32
                    && node.shape() == g.node(start)?.shape()
                    && !skip.contains(&next)
                    && !claimed.contains(&next)
                    && reachable.insert(next)
                {
                    candidates.push(next);
                    pending.push(next);
                }
            }
        }
        candidates.sort_by_key(|v| std::cmp::Reverse(positions[v]));
        for end in candidates {
            if order[positions[&start]..=positions[&end]]
                .iter()
                .any(|&v| g.node(v).unwrap().op() == Op::Store)
            {
                continue;
            }
            let mut region = HashSet::new();
            let mut pending = vec![end];
            while let Some(v) = pending.pop() {
                if reachable.contains(&v) && region.insert(v) && v != start {
                    pending.extend(g.node(v)?.src());
                }
            }
            if !region.contains(&start)
                || region.iter().any(|v| {
                    *v != end
                        && (outputs.contains(v)
                            || consumers
                                .get(v)
                                .into_iter()
                                .flatten()
                                .any(|c| !region.contains(c)))
                })
            {
                continue;
            }
            let mut costs = HashMap::from([(start, 0usize)]);
            for &v in &order[positions[&start] + 1..=positions[&end]] {
                if region.contains(&v) {
                    let cost = 1 + g
                        .node(v)?
                        .src()
                        .iter()
                        .map(|s| costs.get(s).copied().unwrap_or(1))
                        .sum::<usize>()
                        .min(64);
                    costs.insert(v, cost);
                }
            }
            if costs[&end] > 32 {
                continue;
            }
            let contraction = contractions.remove(&start).unwrap();
            contractions.insert(end, contraction);
            claimed.extend(region.iter().copied());
            skip.extend(region.into_iter().filter(|&v| v != end));
            result.insert(end, start);
            break;
        }
    }
    Ok(result)
}

pub(super) fn store_stacks(
    g: &Graph,
    order: &[Value],
    outputs: &[Value],
    consumers: &HashMap<Value, Vec<Value>>,
    skip: &HashSet<Value>,
    inline: &mut HashSet<Value>,
) -> Result<()> {
    // Follow only expressions that will actually be read in this kernel.
    // Materialized producers are snapshots; writable parameters are not.
    fn immutable_read(g: &Graph, v: Value, inline: &HashSet<Value>) -> Result<bool> {
        let node = g.node(v)?;
        if let Arg::Param(p) = node.arg() {
            return Ok(!p.writable);
        }
        if is_view(node.op()) {
            return immutable_read(g, node.src()[0], inline);
        }
        if inline.contains(&v) || node.op() == Op::Const {
            for &s in node.src() {
                if !immutable_read(g, s, inline)? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
    let positions: HashMap<_, _> = order.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    for &store in order {
        let node = g.node(store)?;
        if node.op() != Op::Store {
            continue;
        }
        let mut v = node.src()[1];
        while is_view(g.node(v)?.op())
            && !outputs.contains(&v)
            && consumers.get(&v).is_some_and(|cs| cs.len() == 1)
        {
            v = g.node(v)?.src()[0];
        }
        let stack = g.node(v)?;
        if stack.op() != Op::Stack
            || !(2..=4).contains(&stack.src().len())
            || outputs.contains(&v)
            || skip.contains(&v)
            || inline.contains(&v)
            || !consumers.get(&v).is_some_and(|cs| cs.len() == 1)
            || order[positions[&v] + 1..positions[&store]]
                .iter()
                .any(|&v| g.node(v).unwrap().op() == Op::Store)
        {
            continue;
        }
        if stack
            .src()
            .iter()
            .all(|&v| immutable_read(g, v, inline).unwrap_or(false))
        {
            inline.insert(v);
        }
    }
    Ok(())
}

fn is_view(op: Op) -> bool {
    matches!(
        op,
        Op::Reshape
            | Op::Expand
            | Op::Permute
            | Op::Flip
            | Op::Shrink
            | Op::Load
            | Op::After
            | Op::Window
    )
}

// Only literal integer sequences: no constant evaluation with different
// rounding, overflow or transcendental semantics from generated device code.
pub(super) fn integer_range(g: &Graph, v: Value) -> Option<(i64, i64)> {
    fn literal(g: &Graph, v: Value) -> Option<i64> {
        let node = g.node(v).ok()?;
        match node.arg() {
            Arg::Scalar(Scalar::Int(n)) => Some(*n),
            _ if node.op() == Op::Cast && node.dtype() == DType::I32 => {
                let n = literal(g, node.src()[0])?;
                i32::try_from(n).ok().map(i64::from)
            }
            _ => None,
        }
    }
    let node = g.node(v).ok()?;
    if node.op() != Op::Stack || node.src().is_empty() {
        return None;
    }
    let values = node
        .src()
        .iter()
        .map(|&v| literal(g, v))
        .collect::<Option<Vec<_>>>()?;
    let first = values[0];
    let step = if values.len() > 1 {
        values[1].checked_sub(first)?
    } else {
        0
    };
    values
        .iter()
        .enumerate()
        .all(|(i, &n)| {
            step.checked_mul(i as i64)
                .and_then(|x| first.checked_add(x))
                == Some(n)
        })
        .then_some((first, step))
}
