//! Immutable component contracts and finite decision domains, prepared once.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Choice {
    pub variable: usize,
    pub option: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Activation {
    /// Any clause activates the component; all choices in a clause must hold.
    /// An empty clause is unconditional, while no clauses means never.
    pub clauses: Vec<Vec<Choice>>,
}
impl Activation {
    pub fn always() -> Self {
        Self {
            clauses: vec![vec![]],
        }
    }
    pub fn choice(variable: usize, option: usize) -> Self {
        Self {
            clauses: vec![vec![Choice { variable, option }]],
        }
    }
    pub(super) fn active(&self, selected: &[usize]) -> bool {
        self.clauses
            .iter()
            .any(|clause| clause.iter().all(|c| selected[c.variable] == c.option))
    }
    pub(super) fn guaranteed(&self, domains: &[Vec<usize>]) -> bool {
        self.clauses
            .iter()
            .any(|clause| clause.iter().all(|c| domains[c.variable] == [c.option]))
    }
    pub(super) fn local_variable(&self) -> Option<usize> {
        let variable = self.clauses.first()?.first()?.variable;
        self.clauses
            .iter()
            .all(|c| c.len() == 1 && c[0].variable == variable)
            .then_some(variable)
    }
}

#[derive(Debug, Clone)]
pub enum Variable {
    Attention {
        layer: usize,
        values: Vec<LayerMode>,
    },
    Prefill(Vec<PrefillShape>),
    Lanes(Vec<u64>),
    /// One unit may be coupled across GPUs (TP2); its layers count only once.
    /// An empty destination is the absent option, legal only for optional units.
    Onboard {
        layers: Vec<usize>,
        destinations: Vec<Vec<u8>>,
        mandatory: bool,
    },
}
impl Variable {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Attention { values, .. } => values.len(),
            Self::Prefill(values) => values.len(),
            Self::Lanes(values) => values.len(),
            Self::Onboard { destinations, .. } => destinations.len(),
        }
    }
    pub(super) fn onboard(&self, option: usize) -> u64 {
        match self {
            Self::Onboard {
                layers,
                destinations,
                ..
            } if !destinations[option].is_empty() => layers.len() as u64,
            _ => 0,
        }
    }
    pub(super) fn optional_onboard(&self) -> bool {
        matches!(
            self,
            Self::Onboard {
                mandatory: false,
                ..
            }
        )
    }
}

#[derive(Debug, Clone)]
pub struct Requires {
    pub when: Choice,
    pub any_of: Vec<Choice>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BytePiece {
    pub first: u64,
    pub last: u64,
    pub bytes_at_first: u64,
    pub bytes_per_page: u64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteCurve(pub Vec<BytePiece>);
impl ByteCurve {
    pub fn fixed(bytes: u64, maximum: u64) -> Self {
        Self::linear(bytes, 0, maximum)
    }
    pub fn linear(bytes_at_first: u64, bytes_per_page: u64, maximum: u64) -> Self {
        Self(vec![BytePiece {
            first: 1,
            last: maximum,
            bytes_at_first,
            bytes_per_page,
        }])
    }
    pub fn at(&self, pages: u64) -> Result<u64, Refusal> {
        let piece = self.piece(pages)?;
        piece
            .bytes_per_page
            .checked_mul(pages - piece.first)
            .and_then(|n| piece.bytes_at_first.checked_add(n))
            .ok_or(Refusal::Overflow("component curve"))
    }
    pub(super) fn piece(&self, pages: u64) -> Result<&BytePiece, Refusal> {
        self.0
            .iter()
            .find(|p| p.first <= pages && pages <= p.last)
            .ok_or_else(|| Refusal::InvalidFact("component curve has no page interval".into()))
    }
    pub(super) fn validate(&self, maximum: u64) -> Result<(), Refusal> {
        let mut next = 1;
        for piece in &self.0 {
            if piece.first != next || piece.first > piece.last || piece.last > maximum {
                return Err(Refusal::InvalidFact(
                    "component curve has a gap, overlap or invalid interval".into(),
                ));
            }
            self.at(piece.last)?;
            next = piece
                .last
                .checked_add(1)
                .ok_or(Refusal::Overflow("curve interval"))?;
        }
        if next
            != maximum
                .checked_add(1)
                .ok_or(Refusal::Overflow("maximum pages"))?
        {
            return Err(Refusal::InvalidFact(
                "component curve does not cover page domain".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum ComponentKind {
    Allocation {
        template: Allocation,
        shape: Vec<ByteCurve>,
        offset: Option<ByteCurve>,
        deferred: bool,
    },
    Allowance(Allowance),
}
#[derive(Debug, Clone)]
pub struct Component {
    pub activation: Activation,
    pub bytes: ByteCurve,
    pub kind: ComponentKind,
}
impl Component {
    pub fn allocation(
        template: Allocation,
        bytes: ByteCurve,
        deferred: bool,
        maximum: u64,
    ) -> Self {
        let shape = template
            .shape
            .iter()
            .map(|&n| ByteCurve::fixed(n, maximum))
            .collect();
        let offset = template
            .view
            .as_ref()
            .map(|v| ByteCurve::fixed(v.offset, maximum));
        Self {
            activation: Activation::always(),
            bytes,
            kind: ComponentKind::Allocation {
                template,
                shape,
                offset,
                deferred,
            },
        }
    }
    /// Raw byte buffers keep their sole shape dimension equal to their extent.
    pub fn byte_extent(&mut self, bytes: ByteCurve) -> Result<(), Refusal> {
        let ComponentKind::Allocation {
            template, shape, ..
        } = &mut self.kind
        else {
            return Err(Refusal::InvalidFact("byte extent for an allowance".into()));
        };
        if template.dtype != "bytes" || shape.len() != 1 {
            return Err(Refusal::InvalidFact(
                "byte extent for a typed tensor".into(),
            ));
        }
        *shape = vec![bytes.clone()];
        self.bytes = bytes;
        Ok(())
    }
    pub(super) fn charge(&self) -> Option<(Location, Lifetime, bool)> {
        match &self.kind {
            ComponentKind::Allocation { template, .. } if template.view.is_none() => {
                Some((template.location, template.lifetime, false))
            }
            ComponentKind::Allocation { .. } => None,
            ComponentKind::Allowance(a) => Some((a.location, a.lifetime, true)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Contract {
    pub page_rows: u64,
    pub maximum_pages: u64,
    pub gpus: usize,
    pub variables: Vec<Variable>,
    pub requires: Vec<Requires>,
    pub components: Vec<Component>,
}
impl Contract {
    pub fn fixed(page_rows: u64, maximum_pages: u64, gpus: usize, facts: Facts) -> Self {
        let mut components: Vec<_> = facts
            .allocations
            .into_iter()
            .map(|a| {
                Component::allocation(
                    a.clone(),
                    ByteCurve::fixed(a.bytes, maximum_pages),
                    false,
                    maximum_pages,
                )
            })
            .chain(facts.deferred.into_iter().map(|a| {
                Component::allocation(
                    a.clone(),
                    ByteCurve::fixed(a.bytes, maximum_pages),
                    true,
                    maximum_pages,
                )
            }))
            .collect();
        components.extend(facts.allowances.into_iter().map(|a| Component {
            activation: Activation::always(),
            bytes: ByteCurve::fixed(a.bytes, maximum_pages),
            kind: ComponentKind::Allowance(a),
        }));
        Self {
            page_rows,
            maximum_pages,
            gpus,
            variables: vec![],
            requires: vec![],
            components,
        }
    }
    pub fn set_byte_extent(&mut self, id: &str, curve: ByteCurve) -> Result<(), Refusal> {
        let component = self
            .components
            .iter_mut()
            .find(|c| {
                matches!(&c.kind,
            ComponentKind::Allocation { template, .. } if template.id == id)
            })
            .ok_or_else(|| Refusal::InvalidFact(format!("unknown component {id}")))?;
        component.byte_extent(curve)
    }
    pub fn facts(&self, selected: &[usize], pages: u64) -> Result<Facts, Refusal> {
        self.decisions(selected)?;
        let mut result = Facts {
            allocations: vec![],
            deferred: vec![],
            allowances: vec![],
        };
        for component in &self.components {
            if !component.activation.active(selected) {
                continue;
            }
            let bytes = component.bytes.at(pages)?;
            if bytes == 0 {
                continue;
            }
            match &component.kind {
                ComponentKind::Allocation {
                    template,
                    shape,
                    offset,
                    deferred,
                } => {
                    let mut allocation = template.clone();
                    allocation.bytes = bytes;
                    allocation.shape = shape
                        .iter()
                        .map(|curve| curve.at(pages))
                        .collect::<Result<_, _>>()?;
                    if let (Some(view), Some(offset)) = (&mut allocation.view, offset) {
                        view.offset = offset.at(pages)?;
                    }
                    if *deferred {
                        result.deferred.push(allocation);
                    } else {
                        result.allocations.push(allocation);
                    }
                }
                ComponentKind::Allowance(template) => {
                    let mut allowance = template.clone();
                    allowance.bytes = bytes;
                    result.allowances.push(allowance);
                }
            }
        }
        Ok(result)
    }
    pub fn decisions(&self, selected: &[usize]) -> Result<Decisions, Refusal> {
        if selected.len() != self.variables.len()
            || selected
                .iter()
                .zip(&self.variables)
                .any(|(&o, v)| o >= v.len())
        {
            return Err(Refusal::InvalidFact("invalid contract assignment".into()));
        }
        let mut attention = BTreeMap::new();
        let mut prefill = None;
        let mut lanes = None;
        let mut onboard_layers = vec![vec![]; self.gpus];
        for (variable, &option) in self.variables.iter().zip(selected) {
            match variable {
                Variable::Attention { layer, values } => {
                    attention.insert(*layer, values[option]);
                }
                Variable::Prefill(values) => {
                    prefill = Some(values[option]);
                }
                Variable::Lanes(values) => {
                    lanes = Some(values[option]);
                }
                Variable::Onboard {
                    layers,
                    destinations,
                    ..
                } => {
                    for &gpu in &destinations[option] {
                        onboard_layers[gpu as usize].extend(layers);
                    }
                }
            }
        }
        if attention.keys().copied().ne(0..attention.len()) {
            return Err(Refusal::InvalidFact(
                "noncontiguous attention layers".into(),
            ));
        }
        for layers in &mut onboard_layers {
            layers.sort_unstable();
        }
        Ok(Decisions {
            attention: attention.into_values().collect(),
            onboard_layers,
            prefill: prefill
                .ok_or_else(|| Refusal::InvalidFact("missing prefill domain".into()))?,
            lanes: lanes.ok_or_else(|| Refusal::InvalidFact("missing lane domain".into()))?,
        })
    }
    pub(super) fn validate(&self) -> Result<(), Refusal> {
        if self.page_rows == 0
            || self.maximum_pages == 0
            || self.maximum_pages == u64::MAX
            || self.gpus == 0
        {
            return Err(Refusal::InvalidFact(
                "zero or overflowing contract extent".into(),
            ));
        }
        let mut attention = BTreeSet::new();
        let mut onboard = BTreeSet::new();
        let (mut prefill, mut lanes) = (0, 0);
        for variable in &self.variables {
            if variable.len() == 0 {
                return Err(Refusal::Unsupported);
            }
            match variable {
                Variable::Attention { layer, values } => {
                    if !attention.insert(*layer)
                        || values.iter().any(|v| {
                            matches!(v,
                        LayerMode::Whole { gpu, .. } if usize::from(*gpu) >= self.gpus)
                        })
                    {
                        return Err(Refusal::InvalidFact("invalid attention domain".into()));
                    }
                }
                Variable::Prefill(values) => {
                    prefill += 1;
                    if values.iter().any(|v| v.rows == 0 || v.waves == 0) {
                        return Err(Refusal::InvalidPin("zero prefill shape"));
                    }
                }
                Variable::Lanes(values) => {
                    lanes += 1;
                    if values.contains(&0) {
                        return Err(Refusal::InvalidFact("zero lanes".into()));
                    }
                }
                Variable::Onboard {
                    layers,
                    destinations,
                    mandatory,
                } => {
                    if layers.is_empty()
                        || layers.iter().any(|l| !onboard.insert(*l))
                        || destinations.iter().any(|gpus| {
                            (*mandatory && gpus.is_empty())
                                || gpus.iter().any(|&g| usize::from(g) >= self.gpus)
                                || gpus.iter().collect::<BTreeSet<_>>().len() != gpus.len()
                        })
                    {
                        return Err(Refusal::InvalidFact("invalid onboard unit".into()));
                    }
                }
            }
        }
        if prefill != 1 || lanes != 1 {
            return Err(Refusal::InvalidFact(
                "prefill/lane domains must be unique".into(),
            ));
        }
        let choice_valid = |c: &Choice| {
            self.variables
                .get(c.variable)
                .is_some_and(|v| c.option < v.len())
        };
        for constraint in &self.requires {
            if !choice_valid(&constraint.when) || !constraint.any_of.iter().all(&choice_valid) {
                return Err(Refusal::InvalidFact(
                    "invalid compatibility constraint".into(),
                ));
            }
        }
        for component in &self.components {
            if !component
                .activation
                .clauses
                .iter()
                .flatten()
                .all(&choice_valid)
            {
                return Err(Refusal::InvalidFact("invalid component activation".into()));
            }
            component.bytes.validate(self.maximum_pages)?;
            if let ComponentKind::Allocation { shape, offset, .. } = &component.kind {
                for curve in shape.iter().chain(offset) {
                    curve.validate(self.maximum_pages)?;
                }
            }
        }
        Ok(())
    }
}
