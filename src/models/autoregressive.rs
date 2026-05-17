pub trait KvCache {
    fn seq_len(&self) -> usize;

    fn max_seq_len(&self) -> usize;

    fn clear(&mut self);
}

pub trait AutoregressiveDecoder {
    type Cache: KvCache;
    type Logits;
    type Error;

    fn max_context_len(&self) -> usize;

    fn new_cache(&self) -> Result<Self::Cache, Self::Error>;

    fn prefill(
        &self,
        input_ids: &[usize],
        cache: &mut Self::Cache,
    ) -> Result<Self::Logits, Self::Error>;

    fn forward_one(
        &self,
        token_id: usize,
        cache: &mut Self::Cache,
    ) -> Result<Self::Logits, Self::Error>;

    fn select_next_token(&self, logits: &Self::Logits) -> Result<usize, Self::Error>;
}

pub trait ConditionalAutoregressiveDecoder {
    type Condition;
    type Logits;
    type Error;

    fn max_context_len(&self) -> usize;

    fn prefill(
        &mut self,
        condition: &Self::Condition,
        input_ids: &[usize],
    ) -> Result<Self::Logits, Self::Error>;

    fn forward(
        &mut self,
        condition: &Self::Condition,
        input_ids: &[usize],
    ) -> Result<Self::Logits, Self::Error>;

    fn select_next_token(
        &mut self,
        logits: &Self::Logits,
        history: &[usize],
    ) -> Result<usize, Self::Error>;

    fn should_stop(&self, token_id: usize) -> bool;
}

pub fn generate<M, F, E>(
    model: &M,
    input_ids: &[usize],
    max_new_tokens: usize,
    mut on_token: F,
) -> Result<Vec<usize>, E>
where
    M: AutoregressiveDecoder,
    F: FnMut(usize) -> Result<(), E>,
    E: From<M::Error>,
{
    let mut cache = model.new_cache()?;
    let mut logits = model.prefill(input_ids, &mut cache)?;
    let mut tokens = input_ids.to_vec();

    for _ in 0..max_new_tokens {
        if tokens.len() >= model.max_context_len() {
            break;
        }
        let next = model.select_next_token(&logits)?;
        tokens.push(next);
        on_token(next)?;
        logits = model.forward_one(next, &mut cache)?;
    }

    Ok(tokens)
}

pub fn generate_conditional<M, F, E>(
    model: &mut M,
    condition: &M::Condition,
    input_ids: &[usize],
    max_new_tokens: usize,
    mut on_token: F,
) -> Result<Vec<usize>, E>
where
    M: ConditionalAutoregressiveDecoder,
    F: FnMut(usize) -> Result<(), E>,
    E: From<M::Error>,
{
    let mut tokens = input_ids.to_vec();
    let mut logits = model.prefill(condition, &tokens)?;

    for _ in 0..max_new_tokens {
        if tokens.len() >= model.max_context_len() {
            break;
        }
        let next = model.select_next_token(&logits, &tokens)?;
        if model.should_stop(next) {
            break;
        }
        tokens.push(next);
        on_token(next)?;
        logits = model.forward(condition, &tokens)?;
    }

    Ok(tokens)
}
