//! The model IR graph (format `lev-model-ir-1`), see tools/model_ir/ir_export.py.

use crate::json::{self, Json};

#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
    /// Output of an earlier node.
    Node(usize),
    Param(String),
    Input(String),
    Int(i64),
    /// A symbolic size expression (informational; sizes are taken from the runtime tensors).
    Sym(String),
    Float(f64),
    Bool(bool),
    Str(String),
    Dtype(String),
    None,
    List(Vec<Arg>),
}

#[derive(Debug, Clone)]
pub struct Node {
    pub op: String,
    pub args: Vec<Arg>,
    pub kwargs: Vec<(String, Arg)>,
}

impl Node {
    pub fn kwarg(&self, name: &str) -> Option<&Arg> {
        self.kwargs.iter().find(|(k, _)| k == name).map(|(_, v)| v)
    }
}

#[derive(Debug, Clone)]
pub struct Value {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Graph {
    pub inputs: Vec<Value>,
    pub params: Vec<Value>,
    pub nodes: Vec<Node>,
    pub outputs: Vec<Arg>,
}

pub const FORMAT: &str = "lev-model-ir-1";

fn shape_of(j: &Json) -> Vec<String> {
    match j {
        Json::Arr(a) => a
            .iter()
            .map(|d| match d {
                Json::Int(i) => i.to_string(),
                Json::Str(s) => s.clone(),
                _ => "?".into(),
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn values(j: Option<&Json>, what: &str) -> Result<Vec<Value>, String> {
    let arr = j.and_then(Json::as_arr).ok_or(format!("missing {what}"))?;
    arr.iter()
        .map(|v| {
            Ok(Value {
                name: v.get("name").and_then(Json::as_str).ok_or(format!("{what}: missing name"))?.to_string(),
                dtype: v.get("dtype").and_then(Json::as_str).unwrap_or("float32").to_string(),
                shape: v.get("shape").map(shape_of).unwrap_or_default(),
            })
        })
        .collect()
}

fn arg(j: &Json) -> Result<Arg, String> {
    let Json::Obj(kv) = j else {
        return Err(format!("argument is not an object: {j:?}"));
    };
    let Some((k, v)) = kv.first() else {
        return Err("empty argument".into());
    };
    Ok(match (k.as_str(), v) {
        ("n", Json::Int(i)) => Arg::Node(*i as usize),
        ("p", Json::Str(s)) => Arg::Param(s.clone()),
        ("i", Json::Str(s)) => Arg::Input(s.clone()),
        ("int", Json::Int(i)) => Arg::Int(*i),
        ("sym", Json::Str(s)) => Arg::Sym(s.clone()),
        ("f", v) => Arg::Float(v.as_f64().ok_or("bad float")?),
        ("b", Json::Bool(b)) => Arg::Bool(*b),
        ("str", Json::Str(s)) => Arg::Str(s.clone()),
        ("dtype", Json::Str(s)) => Arg::Dtype(s.clone()),
        ("none", _) => Arg::None,
        ("list", Json::Arr(a)) => Arg::List(a.iter().map(arg).collect::<Result<_, _>>()?),
        _ => return Err(format!("unknown argument encoding {k}")),
    })
}

impl Graph {
    pub fn parse(text: &str) -> Result<Graph, String> {
        let root = json::parse(text)?;
        if root.get("format").and_then(Json::as_str) != Some(FORMAT) {
            return Err(format!("not a {FORMAT} file"));
        }
        let mut nodes = Vec::new();
        for n in root.get("nodes").and_then(Json::as_arr).ok_or("missing nodes")? {
            nodes.push(Node {
                op: n.get("op").and_then(Json::as_str).ok_or("node without op")?.to_string(),
                args: n.get("args").and_then(Json::as_arr).ok_or("node without args")?.iter().map(arg).collect::<Result<_, _>>()?,
                kwargs: match n.get("kwargs") {
                    Some(Json::Obj(kv)) => kv.iter().map(|(k, v)| Ok((k.clone(), arg(v)?))).collect::<Result<_, String>>()?,
                    _ => Vec::new(),
                },
            });
        }
        let outputs = root.get("outputs").and_then(Json::as_arr).ok_or("missing outputs")?.iter().map(arg).collect::<Result<_, _>>()?;
        Ok(Graph { inputs: values(root.get("inputs"), "inputs")?, params: values(root.get("params"), "params")?, nodes, outputs })
    }

    /// Structural checks: references point backwards, parameters and inputs exist.
    pub fn validate(&self) -> Result<(), String> {
        fn check(a: &Arg, idx: usize, g: &Graph) -> Result<(), String> {
            match a {
                Arg::Node(n) if *n >= idx => Err(format!("node {idx} refers forward to node {n}")),
                Arg::Param(p) if !g.params.iter().any(|v| &v.name == p) => Err(format!("node {idx}: unknown parameter {p}")),
                Arg::Input(i) if !g.inputs.iter().any(|v| &v.name == i) => Err(format!("node {idx}: unknown input {i}")),
                Arg::List(l) => l.iter().try_for_each(|x| check(x, idx, g)),
                _ => Ok(()),
            }
        }
        for (i, n) in self.nodes.iter().enumerate() {
            n.args.iter().chain(n.kwargs.iter().map(|(_, v)| v)).try_for_each(|a| check(a, i, self))?;
        }
        self.outputs.iter().try_for_each(|a| check(a, self.nodes.len(), self))
    }

    /// Number of decoder layers (the highest `model.layers.N` index among the parameters, plus one).
    pub fn layer_count(&self) -> usize {
        self.params
            .iter()
            .filter_map(|p| p.name.strip_prefix("model.layers.")?.split('.').next()?.parse::<usize>().ok())
            .max()
            .map_or(0, |m| m + 1)
    }

    /// Node range `[first, end)` of decoder layer `l`: from the first node that reads one of its
    /// parameters to the first node that reads a parameter of layer `l + 1` (or of the final norm
    /// and output projection for the last layer).
    pub fn layer_segment(&self, l: usize) -> Option<(usize, usize)> {
        fn reads(a: &Arg, prefixes: &[String]) -> bool {
            match a {
                Arg::Param(p) => prefixes.iter().any(|x| p.starts_with(x.as_str())),
                Arg::List(v) => v.iter().any(|x| reads(x, prefixes)),
                _ => false,
            }
        }
        let find = |from: usize, prefixes: &[String]| self.nodes.iter().enumerate().skip(from).find(|(_, n)| n.args.iter().any(|a| reads(a, prefixes))).map(|(i, _)| i);
        let first = find(0, &[format!("model.layers.{l}.")])?;
        let end = find(first, &[format!("model.layers.{}.", l + 1), "model.norm.".to_string(), "lm_head.".to_string()]).unwrap_or(self.nodes.len());
        Some((first, end))
    }
    /// Operators of the graph with their counts, sorted by name.
    pub fn op_counts(&self) -> Vec<(String, usize)> {
        let mut m = std::collections::BTreeMap::new();
        for n in &self.nodes {
            *m.entry(n.op.clone()).or_insert(0usize) += 1;
        }
        m.into_iter().collect()
    }
}
