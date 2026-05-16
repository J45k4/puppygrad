pub trait AutoregressiveDecoder {
    type Cache;
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
