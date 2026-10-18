//! Graph rewrites. A pass takes a graph and returns a new one; node ids are renumbered.

use std::collections::BTreeSet;

use crate::ir::{Arg, Graph, Node, Value};

fn map_arg(a: &Arg, f: &dyn Fn(&Arg) -> Option<Arg>) -> Arg {
    if let Some(r) = f(a) {
        return r;
    }
    match a {
        Arg::List(l) => Arg::List(l.iter().map(|x| map_arg(x, f)).collect()),
        other => other.clone(),
    }
}

fn collect_nodes(a: &Arg, out: &mut Vec<usize>) {
    match a {
        Arg::Node(n) => out.push(*n),
        Arg::List(l) => l.iter().for_each(|x| collect_nodes(x, out)),
        _ => {}
    }
}

fn collect_params<'a>(a: &'a Arg, out: &mut BTreeSet<&'a str>) {
    match a {
        Arg::Param(p) => {
            out.insert(p);
        }
        Arg::List(l) => l.iter().for_each(|x| collect_params(x, out)),
        _ => {}
    }
}

/// Removes nodes that no output depends on, renumbers the rest, and drops parameters that are no
/// longer used.
pub fn eliminate_dead(g: &Graph) -> Graph {
    let mut live = vec![false; g.nodes.len()];
    let mut work = Vec::new();
    g.outputs.iter().for_each(|a| collect_nodes(a, &mut work));
    while let Some(n) = work.pop() {
        if !live[n] {
            live[n] = true;
            let node = &g.nodes[n];
            node.args.iter().chain(node.kwargs.iter().map(|(_, v)| v)).for_each(|a| collect_nodes(a, &mut work));
        }
    }
    let mut new_id = vec![usize::MAX; g.nodes.len()];
    let mut next = 0;
    for (i, l) in live.iter().enumerate() {
        if *l {
            new_id[i] = next;
            next += 1;
        }
    }
    let renumber = |a: &Arg| -> Option<Arg> {
        match a {
            Arg::Node(n) => Some(Arg::Node(new_id[*n])),
            _ => None,
        }
    };
    let nodes: Vec<Node> = g
        .nodes
        .iter()
        .enumerate()
        .filter(|(i, _)| live[*i])
        .map(|(_, n)| Node {
            op: n.op.clone(),
            args: n.args.iter().map(|a| map_arg(a, &renumber)).collect(),
            kwargs: n.kwargs.iter().map(|(k, v)| (k.clone(), map_arg(v, &renumber))).collect(),
        })
        .collect();
    let outputs: Vec<Arg> = g.outputs.iter().map(|a| map_arg(a, &renumber)).collect();
    let mut used = BTreeSet::new();
    for n in &nodes {
        n.args.iter().chain(n.kwargs.iter().map(|(_, v)| v)).for_each(|a| collect_params(a, &mut used));
    }
    outputs.iter().for_each(|a| collect_params(a, &mut used));
    let params: Vec<Value> = g.params.iter().filter(|p| used.contains(p.name.as_str())).cloned().collect();
    Graph { inputs: g.inputs.clone(), params, nodes, outputs }
}

/// Rewrites a graph exported with Hugging Face conventions to run on GGUF weights:
///
/// * `lev::rmsnorm` loses `zero_centered` (GGUF stores `1 + w`),
/// * the key/value head expansion becomes `lev::tile_heads` (the converter tiles the value heads
///   where Hugging Face groups them),
/// * `neg(exp(to(A_log)))` is folded into the parameter `...linear_attn.ssm_a` (GGUF stores it),
/// * nodes that no output depends on are removed.
pub fn retarget_gguf(g: &Graph) -> Result<Graph, String> {
    let mut nodes = g.nodes.clone();
    let mut params = g.params.clone();

    for n in &mut nodes {
        match n.op.as_str() {
            "lev::rmsnorm.default" => {
                let slot = n.args.get_mut(3).ok_or("rmsnorm without zero_centered")?;
                *slot = Arg::Bool(false);
            }
            "aten::repeat_interleave.self_int" => n.op = "lev::tile_heads.default".into(),
            _ => {}
        }
    }

    // neg(exp(to(Param A_log))) -> Param ssm_a
    let mut replace: Vec<Option<Arg>> = vec![None; nodes.len()];
    for i in 0..nodes.len() {
        if nodes[i].op != "aten::neg.default" {
            continue;
        }
        let Some(Arg::Node(e)) = nodes[i].args.first() else { continue };
        if nodes[*e].op != "aten::exp.default" {
            continue;
        }
        let Some(Arg::Node(t)) = nodes[*e].args.first() else { continue };
        if nodes[*t].op != "aten::to.dtype" {
            continue;
        }
        let Some(Arg::Param(p)) = nodes[*t].args.first() else { continue };
        let Some(base) = p.strip_suffix(".A_log") else { continue };
        let name = format!("{base}.ssm_a");
        if !params.iter().any(|v| v.name == name) {
            let shape = params.iter().find(|v| &v.name == p).map(|v| v.shape.clone()).unwrap_or_default();
            params.push(Value { name: name.clone(), dtype: "float32".into(), shape });
        }
        replace[i] = Some(Arg::Param(name));
    }
    let subst = |a: &Arg| -> Option<Arg> {
        match a {
            Arg::Node(n) => replace[*n].clone(),
            _ => None,
        }
    };
    for n in &mut nodes {
        n.args = n.args.iter().map(|a| map_arg(a, &subst)).collect();
        n.kwargs = n.kwargs.iter().map(|(k, v)| (k.clone(), map_arg(v, &subst))).collect();
    }
    let outputs = g.outputs.iter().map(|a| map_arg(a, &subst)).collect();
    Ok(eliminate_dead(&Graph { inputs: g.inputs.clone(), params, nodes, outputs }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Graph;

    fn small() -> Graph {
        let text = r#"{"format":"lev-model-ir-1","inputs":[{"name":"x","dtype":"float32","shape":[4]}],
          "params":[{"name":"m.linear_attn.A_log","dtype":"float32","shape":[4]},{"name":"w","dtype":"float32","shape":[4]}],
          "nodes":[
            {"op":"aten::to.dtype","args":[{"p":"m.linear_attn.A_log"},{"dtype":"float32"}],"kwargs":{}},
            {"op":"aten::exp.default","args":[{"n":0}],"kwargs":{}},
            {"op":"aten::neg.default","args":[{"n":1}],"kwargs":{}},
            {"op":"aten::_assert_tensor_metadata.default","args":[{"i":"x"}],"kwargs":{}},
            {"op":"lev::rmsnorm.default","args":[{"i":"x"},{"p":"w"},{"f":1e-6},{"b":true}],"kwargs":{}},
            {"op":"aten::mul.Tensor","args":[{"n":4},{"n":2}],"kwargs":{}}],
          "outputs":[{"n":5}]}"#;
        Graph::parse(text).unwrap()
    }

    #[test]
    fn retarget_folds_a_log_and_clears_the_norm_flag() {
        let g = retarget_gguf(&small()).unwrap();
        g.validate().unwrap();
        let ops: Vec<&str> = g.nodes.iter().map(|n| n.op.as_str()).collect();
        assert_eq!(ops, ["lev::rmsnorm.default", "aten::mul.Tensor"]);
        assert_eq!(g.nodes[0].args[3], Arg::Bool(false));
        assert_eq!(g.nodes[1].args, vec![Arg::Node(0), Arg::Param("m.linear_attn.ssm_a".into())]);
        let names: Vec<&str> = g.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["w", "m.linear_attn.ssm_a"]);
    }

    #[test]
    fn dead_nodes_are_removed_and_ids_renumbered() {
        let g = eliminate_dead(&small());
        assert_eq!(g.nodes.len(), 5);
        assert_eq!(g.outputs, vec![Arg::Node(4)]);
        g.validate().unwrap();
    }
}
