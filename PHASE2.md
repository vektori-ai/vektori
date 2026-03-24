# Phase 2 — Retrieval & Ingestion Fixes

## Root Causes

**A. Extraction too eager / too dumb**
- Fires on every `add()` with no batching → junk facts from low-signal turns
- No write-time semantic dedup → near-duplicate facts pile up
- No hard limit → LLM bloats tables
- No cross-turn accumulation → insights can't form real patterns

**B. Retrieval too dumb / too aggressive**
- Facts-first entry point, no fallback → returns empty when extraction still pending
- `mentions_boost` in scoring is dead code (facts table has no `mentions` column)
- No retrieval gate → noise injected on every query turn
- No entity-aware filtering → cross-entity bleed (Chrome extension bug root cause)

---

## What We're Building

### 1. Sentence fallback when no facts — `search.py:135`

When `seed_facts` is empty (extraction still pending), fall through to `search_sentences()` instead of returning `_empty(depth)`. Sentences are always ready (stored synchronously in `add()`). This makes async extraction safe — retrieval degrades gracefully.

```
current:  no facts → return {}
fixed:    no facts → search_sentences() → return sentence-based results
```

### 2. Fix `mentions` on facts — schema + scoring

Facts table has no `mentions` column. `mentions_boost` in `scoring.py` always returns 1.0. Add column to schema, increment during write-time semantic dedup (step 3).

### 3. Write-time semantic dedup — `extractor.py` + storage

Before inserting each new fact, query top-3 nearest neighbors for that user.

| Condition | Action |
|---|---|
| Same user + same session + sim > 0.92 | Skip insert, increment `mentions` on existing |
| Same user + different session + sim > 0.85 | Insert but link as `related_to`, increment `mentions` on older |
| Everything else | Insert normally |

Kills the Chrome extension hallucination problem at the source — similar facts were getting merged aggressively because there was no provenance check.

### 4. Extraction count limits — `extractor.py` + config

Add `max_facts` / `max_insights` to `VektoriConfig`. Enforce in prompt. Insights should always be fewer than facts (they're cross-turn patterns, not per-message observations).

Rough defaults: `max_facts=8`, `max_insights=3` per extraction batch.

### 5. Token-threshold batching — `async_worker.py` + `client.py`

Inspired by Honcho's ~1000 token batching. Instead of scheduling extraction on every `add()`:
- Keep a pending buffer + token counter per `(user_id, agent_id)` in the worker
- Accumulate until tokens > threshold (default ~800)
- Then trigger extraction on the full batch

Benefits:
- No extraction from "ok" / "sounds good" alone
- Natural cross-turn batching → insights actually span turns (cross-session patterns become possible)
- Cheaper (fewer LLM calls)
- Race condition becomes structurally less likely (extraction intentionally deferred)

### 6. Retrieval gate — thin layer above `search.py`

Cheap heuristic before touching the DB. No LLM needed.

```
retrieve if:
  - personal reference ("I", "my", "prefer", "remember", "do I", "what do I")
  - named entity in query (capitalized non-first word)
  - explicit memory question ("what do you know about", "do you remember")

skip retrieval if:
  - very short follow-up with pronoun whose referent is in recent context
  - generic factual question with no personal anchor ("what is X", "explain Y")
```

Cuts unnecessary retrievals ~40-60% for typical usage. Prevents the most common hallucination path: marginally relevant facts get injected and the LLM stitches them into something wrong.

---

## Deferred to Phase 3

**Entity extraction + `fact_entities` table + 1-hop graph traversal**

The right long-term fix for cross-entity bleed. Requires:
- New `fact_entities` join table in schema
- Prompt change (extract subject entities per fact alongside text)
- New storage methods on all 3 backends
- New retrieval path: extract entities from query → seed facts via entity tags → 1-hop expand → vector search within candidate set → rank

Don't use subject as a hard filter — use it as a graph seed. 1-hop expansion preserves relational context (fact about "Laxman works with Manit" is reachable from a query about either person).

---

## Implementation Order

```
1. Sentence fallback          ← fixes broken retrieval immediately, critical
2. Fix mentions on facts      ← fixes dead scoring code
3. Extraction count limits    ← one config + one prompt line
4. Write-time semantic dedup  ← kills duplication at write time
5. Token-threshold batching   ← improves extraction quality, solves race properly
6. Retrieval gate             ← reduces noise at query time
```

1–3 are small changes. 4–6 are moderate. All six = Phase 2.


❯ okay so ef minimal_deriver_prompt(                                                                                                                             
    peer_id: str,                                                                                                                                                
    messages: str,                                                                                                                                               
) -> str:                                                                                                                                                        
    """                                                                                                                                                          
    Generate minimal prompt for fast observation extraction.                                                                                                     
                                                                                                                                                                 
    Args:                                                                                                                                                        
        peer_id: The ID of the user being analyzed.                                                                                                              
        messages: All messages in the range (interleaving messages and new turns combined).                                                                      
                                                                                                                                                                 
    Returns:                                                                                                                                                     
        Formatted prompt string for observation extraction.                                                                                                      
    """                                                                                                                                                          
    return c(                                                                                                                                                    
        f"""                                                                                                                                                     
Analyze messages from {peer_id} to extract **explicit atomic facts** about them.                                                                                 
                                                                                                                                                                 
[EXPLICIT] DEFINITION: Facts about {peer_id} that can be derived directly from their messages.                                                                   
   - Transform statements into one or multiple conclusions                                                                                                       
   - Each conclusion must be self-contained with enough context                                                                                                  
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")                                                                               
                                                                                                                                                                 
RULES:                                                                                                                                                           
- Properly attribute observations to the correct subject: if it is about {peer_id}, say so. If {peer_id} is referencing someone or something else, make that     
clear.                                                                                                                                                           
- Observations should make sense on their own. Each observation will be used in the future to better understand {peer_id}.                                       
- Extract ALL observations from {peer_id} messages, using others as context.                                                                                     
- Contextualize each observation sufficiently (e.g. "Ann is nervous about the job interview at the pharmacy" not just "Ann is nervous")                          
                                                                                                                                                                 
EXAMPLES:                                                                                                                                                        
- EXPLICIT: "I just had my 25th birthday last Saturday" → "{peer_id} is 25 years old", "{peer_id}'s birthday is June 21st"                                       
- EXPLICIT: "I took my dog for a walk in NYC" → "{peer_id} has a dog", "{peer_id} lives in NYC"                                                                  
- EXPLICIT: "{peer_id} attended college" + general knowledge → "{peer_id} completed high school or equivalent"                                                   
                                                                                                                                                                 
Messages to analyze:                                                                                                                                             
<messages>                                                                                                                                                       
{messages}                                                                                                                                                       
</messages>                                                                                                                                                      
"""                                                                                                                                                              
    )                                                                                                                                                            
                                                                                                                                                                 
                                                                                                                                                                 
@cache                                                                                                                                                           
                                                                                                                                                                 
                                                                                                                                                                 
plastic-labs/honcho                                                                                                                                              
tests/bench/coverage.py                                                                                                                                          
                                                                                                                                                                 
                                                                                                                                                                 
# =============================================================================                                                                                  
                                                                                                                                                                 
                                                                                                                                                                 
GOLD_EXTRACTION_PROMPT = """You are extracting ALL facts that could reasonably be derived from conversation messages.                                            
                                                                                                                                                                 
## Your Task                                                                                                                                                     
                                                                                                                                                                 
Given messages from a conversation, extract every piece of factual information about the speaker.                                                                
Be EXHAUSTIVE - capture everything that a careful reader could learn about the speaker.                                                                          
                                                                                                                                                                 
## Fact Categories                                                                                                                                               
                                                                                                                                                                 
**EXPLICIT**: Directly stated facts                                                                                                                              
- "I work at Google" → "Speaker works at Google"                                                                                                                 
                                                                                                                                                                 
**IMPLICIT**: Clearly implied by context (not speculation)                                                                                                       
- "My commute to the Googleplex is 30 minutes" → "Speaker lives within 30 minutes of Googleplex"                                                                 
                                                                                                                                                                 
**RELATIONAL**: Facts about relationships                                                                                                                        
- "My sister's wedding was beautiful" → "Speaker has a sister", "Speaker's sister is married"                                                                    
                                                                                                                                                                 
**TEMPORAL**: Time-bound information                                                                                                                             
- "I started this job last year" → "Speaker started current job within the past year"                                                                            
                                                                                                                                                                 
**PREFERENCE**: Likes, dislikes, opinions                                                                                                                        
- "I love hiking on weekends" → "Speaker enjoys hiking"                                                                                                          
                                                                                                                                                                 
**BIOGRAPHICAL**: Personal details                                                                                                                               
- "I'm 32 years old" → "Speaker is 32 years old"                                                                                                                 
                                                                                                                                                                 
**BEHAVIORAL**: Habits and patterns                                                                                                                              
- "I usually wake up at 6am" → "Speaker typically wakes up at 6am"                                                                                               
                                                                                                                                                                 
## Importance Levels                                                                                                                                             
                                                                                                                                                                 
**CRITICAL**: Core identifying info (name, occupation, location, key relationships)                                                                              
**IMPORTANT**: Significant details that build a clear picture                                                                                                    
**MINOR**: Nice-to-have details                                                                                                                                  
**TRIVIAL**: Marginal information that adds little value                                                                                                         
                                                                                                                                                                 
## Rules                                                                                                                                                         
                                                                                                                                                                 
1. Extract ATOMIC facts (one claim per fact)                                                                                                                     
2. Resolve all pronouns and references to be standalone                                                                                                          
3. Include the source span that supports each fact                                                                                                               
4. Mark whether inference was required                                                                                                                           
5. Do NOT speculate beyond what the text clearly implies                                                                                                         
6. Do NOT include facts about the assistant/system, only about the human speaker"""                                                                              
                                                                                                                                                                 
                                                                                                                                                                 
COVERAGE_MATCHING_PROMPT = """You are evaluating how well extracted facts cover the gold standard facts.                                                         
                                                                                                                                                                 
                                                                                                                                                                 
            logger.exception("Unexpected error in LLM call")                                                                                                     
            raise                                                                                                                                                
                                                                                                                                                                 
    async def extract_gold_facts(                                                                                                                                
        self, source_messages: list[dict[str, Any]]                                                                                                              
    ) -> list[GoldFact]:                                                                                                                                         
        """                                                                                                                                                      
        Stage 1: Extract all facts that SHOULD be extractable from source.go thorugh these prompts. so go throiugh this for the prompts also tyr to implement    
the input/output token length idea. also implement the facts to sentence linking for now...                                                                      
remove the cross session feature for now(like let it be there.. but dont use it) also how can we link the facts and insights.. what method is there.. cuz im     
assuming fatsare connected to sentences and ingishts are also connected to sentences.. but not directly... is it even needed?                                    
so if we try to do both act and insight generation in 1 llm call instead of 2.. wont we solve the above problem.. give pros n cons                               
  ⎿  ⧉ Selected 1 lines from vektori/ingestion/extractor.py in Visual Studio Code

● Let me read the current extractor state before rewriting:

  Read 1 file (ctrl+o to expand)    
                                   
● Okay — answers first, then implementing all of it.
                                                         
  ---
  Combined 1-call: pros & cons                                                                                                                                   
  
  Pros:                                                                                                                                                          
  1. ID-based insight linking works perfectly in-call — the LLM numbers its own facts [F1]...[FN] in the output, then references them in insights. You have the
  UUID mapping in memory immediately. Zero text matching, zero paraphrase risk.                                                                                  
  2. LLM sees raw conversation for insights — richer than seeing only fact texts. It can pick up tone, hesitation, what wasn't said.
  3. One fewer API call per session.                                                                                                                             
  4. Simpler flow — one parse, one pass.                                                                                                                         
  
  Cons:                                                                                                                                                          
  1. Per-session insights can't detect recurrence — "user has cancelled 3 appointments in a row" requires seeing multiple sessions. One session can only observe,
   not pattern-match across time.                                                                                                                                
  2. Fail-together risk — bad JSON loses both facts and insights. Currently bad JSON only loses facts.                                   
  3. Output schema is more complex → harder to prompt-tune quality of facts independently of insights.                                                           
                                                                                                                                                                 
  Verdict: Combined call is correct right now since cross-session is disabled. Per-session insights from raw conversation > no per-session insights. And the ID  
  linking solves the connection problem cleanly.                                                                                                                 
                                                                                                                                                                 
  ---                                                                                                                                                            
  On insight_sources — dead table, not needed. Sentences are surfaced via fact → fact_sources → sentence. Insights ride along via insight_facts to those same  
  facts. Two-hop is sufficient, the direct path is redundant. Leave the schema/interface alone (migration cost), just never write to it.                         
  
  ---                                                                                                                                                            
  Now implementing: combined prompt, honcho-inspired fact rules, sentence_ids piped through from pipeline, conv truncation,cross-session disabled.            
                                                                                                                                                                 
● Update(vektori/utils/async_worker.py)    
  ⎿  User rejected update to vektori/utils/async_worker.py                                                                                                       
      17      session_id: str                                                                                                                                    
      18      user_id: str              
      19      agent_id: str | None = None                                                                                                                        
      20 +    sentence_ids: list[str] | None = None  # IDs of sentences stored in this session                                                                 
      21  
      22                                           
      23  class ExtractionWorker:
                                                           
❯ the ideas make snese, however other than this method of combined prompt/using 1 pormot fpr both... is there no other way to actually do this linking of facts  
and insights? epxlain                                                                                                                                            
  ⎿  ⧉ Selected 1 lines from vektori/client.py in Visual Studio Code                                                                                             
                                                                                                                                                                 
● Yes — there are a few other ways. None are as clean as the combined call but they're worth understanding:                                                      
                                                                                                                                                                 
  ---                                                                                                                                                            
  Option A: Post-hoc embedding similarity                                                                                                                        
                                                                                                                                                                 
  After insights are generated and stored, embed each insight and run search_facts() to find the top-K most cosine-similar facts. Link those.                  
                                                                                                                                                                 
  No prompt changes. No LLM involvement. Fully automated.
                                                                                                                                                                 
  Problem: insights are inferred/synthesized statements, not paraphrases of facts. "User may benefit from encouragement before difficult conversations" doesn't  
  cosine-match well to "user is nervous about the interview" — they're semantically adjacent but not similar vectors. You'dget false positives and miss real
  derivations. Exactly the reason we moved fact search to L0 and made insights graph-discovered rather than vector-searched.                                     
                  
  ---
  Option B: Two calls, but pass fact IDs to the second call
                                                                                                                                                                 
  Call 1 → extracts facts, numbers them [F1], [F2]... in the response
  Call 2 → receives the numbered fact list + conversation, returns insights with derived_from_fact_ids: ["F1"]                                                   
                                                                                                                                                                 
  Same ID-based reliability as the combined call, separation of concerns preserved.                                                                              
                                                                                                                                                                 
  Problem: still two API calls. And you have to pipe the numbered facts from call 1's response into call 2's prompt — which means you need to parse call 1 before
   building call 2's prompt. The combined call is strictly simpler for the same reliability.
                                                                                                                                                                 
  ---             
  Option C: Session-scoped link (structural, no LLM)
                                                                                                                                                                 
  Don't link insights to specific facts at all. Instead link every insight generated from session X to every fact from session X — purely via session_id. Since
  both came from the same conversation, the relationship is structurally guaranteed.                                                                             
                  
  Problem: breaks for cross-session insights which by definition span multiple sessions. And within one session it's imprecise — if you extract 8 facts and 2    
  insights, the insights probably relate to 2–3 of those facts, not all 8. This makes the insight_facts table noisy.
                                                                                                                                                                 
  ---             
  Option D: Structured output / function calling
                                                
  Enforce the schema at the API level. Pass facts as tool parameters with valid IDs, LLM MUST return IDs from the enum. Schema validation catches bad references
  before they hit the DB.                                                                                                                                        
  
  Problem: requires changing LLMProvider.generate() from str → str to support tool/function call schemas. Anthropic, OpenAI, and Ollama all have different APIs  
  for this. Big interface change, not worth it unless you're also getting other benefits from structured output.
                                                                                                                                                                 
  ---             
  Bottom line
             
  ┌────────────────────────────┬────────────────────┬───────────┬─────────────────────────┐
  │           Option           │    Reliability     │ API calls │       Complexity        │                                                                      
  ├────────────────────────────┼────────────────────┼───────────┼─────────────────────────┤
  │ Combined prompt (proposed) │ ✓ self-referential │ 1         │ low                     │                                                                      
  ├────────────────────────────┼────────────────────┼───────────┼─────────────────────────┤
  │ Two calls + pass IDs (B)   │ ✓ same             │ 2         │ medium                  │
  ├────────────────────────────┼────────────────────┼───────────┼─────────────────────────┤                                                                      
  │ Post-hoc embedding (A)     │ ✗ imprecise        │ 0 extra   │ low                     │
  ├────────────────────────────┼────────────────────┼───────────┼─────────────────────────┤                                                                      
  │ Session-scoped (C)         │ ~ structural only  │ 0 extra   │ lowest                  │
  ├────────────────────────────┼────────────────────┼───────────┼─────────────────────────┤                                                                      
  │ Structured output (D)      │ ✓ enforced         │ 1         │ high (interface change) │
  └────────────────────────────┴────────────────────┴───────────┴─────────────────────────┘                                                                      
                  
  Combined prompt wins on every axis except "insights can't span sessions" — which is fine since you've disabled cross-session for now. Option B is the closest  
  alternative if you ever want to split fact/insight tuning. Option C is a reasonable fallback if the combined prompt quality is bad in practice (store gross
  links, fix precision later).                                                                                                                                   
                  
✻ Churned for 44s              

❯ okay lets use combined prompt. (link all facts to sentences for now) and then ensure u keep a limit on the token input prompt and output.                      
                                                                                                                                            
  Read 4 files (ctrl+o to expand)                                                                                                                                
                                                                                                                                                                 
● Now I have everything. Implementing in one shot across all files:                                                                                              
                                                      
  Plan: add max_tokens to LLMProvider.generate() + wire all providers → add sentence_ids to ExtractionRequest → pass through pipeline → rewrite extractor with   
  combined prompt, conv truncation, all-sentences linking.                                                                                                       
                                                                                                                                                                 
● Update(vektori/models/base.py)                                                                                                                                 
  ⎿  Added 2 lines, removed 1 line                                                                                                                               
      32      """Abstract LLM provider for fact and insight extraction."""                                                                                       
      33                                
      34      @abstractmethod                                                                                                                                    
      35 -    async def generate(self, prompt: str) -> str:                                                                                                    
      35 +    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:                                                                       
      36          """                                                                          
      37          Generate a completion for the given prompt.                                   
      38          Should return valid JSON for extraction prompts.                                                                                               
      39 +        max_tokens: cap output length. None = provider default.                                                                                        
      40          """                                                                                                                                            
      41          ...                                                                                                                                            
                                                                                                                                                                 
● Update(vektori/models/openai.py) 
  ⎿  Added 10 lines, removed 7 lines
      67              self._client = AsyncOpenAI(api_key=self._api_key)                                                                                          
      68          return self._client
      69                                                                                                                                                         
      70 -    async def generate(self, prompt: str) -> str:                                                                                                    
      70 +    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:                                                                       
      71          client = self._get_client()                                                  
      72 -        response = await client.chat.completions.create(                                                                                             
      73 -            model=self.model,                                                                                                                        
      74 -            messages=[{"role": "user", "content": prompt}],                                                                                     
      75 -            response_format={"type": "json_object"},                                                                                            
      76 -            temperature=0.1,                                                                                                                    
      77 -        )                                                                                                                                       
      72 +        kwargs: dict = {                                                                                                                        
      73 +            "model": self.model,                                                                                                                
      74 +            "messages": [{"role": "user", "content": prompt}],                                                                                  
      75 +            "response_format": {"type": "json_object"},                                                                                         
      76 +            "temperature": 0.1,                                                                                                                 
      77 +        }                                                                                                                                       
      78 +        if max_tokens is not None:                                                                                                              
      79 +            kwargs["max_tokens"] = max_tokens                                                                                                   
      80 +        response = await client.chat.completions.create(**kwargs)                                                                               
      81          return response.choices[0].message.content or ""

● Update(vektori/models/anthropic.py)
  ⎿  Added 2 lines, removed 2 lines
      72              self._client = anthropic.AsyncAnthropic(api_key=self._api_key)                                                                             
      73          return self._client
      74                                                                                                                                                         
      75 -    async def generate(self, prompt: str) -> str:                                                                                                    
      75 +    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:                                                                       
      76          client = self._get_client()                                                  
      77          message = await client.messages.create(                                   
      78              model=self.model,                                   
      79 -            max_tokens=2048,                                                                                                                    
      79 +            max_tokens=max_tokens or 2048,                                                                                                      
      80              messages=[{"role": "user", "content": prompt}],
      81          )
      82          return message.content[0].text

● Update(vektori/models/ollama.py)  
  ⎿  Added 5 lines, removed 2 lines
      76          self.model = model or DEFAULT_LLM_MODEL                                                                                                        
      77          self.base_url = base_url.rstrip("/")
      78                                                                                                                                                         
      79 -    async def generate(self, prompt: str) -> str:                                                                                                    
      79 +    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:                                                                       
      80 +        payload: dict = {"model": self.model, "prompt": prompt, "stream": False}                                                                       
      81 +        if max_tokens is not None:                                                                                                                     
      82 +            payload["options"] = {"num_predict": max_tokens}                                                                                           
      83          async with httpx.AsyncClient() as client:                                                                                                      
      84              response = await client.post(
      85                  f"{self.base_url}/api/generate",
      83 -                json={"model": self.model, "prompt": prompt, "stream": False},                                                                  
      86 +                json=payload,                                                                                                                   
      87                  timeout=120.0,
      88              )
      89              response.raise_for_status()

● Update(vektori/models/litellm_provider.py)
  ⎿  Added 5 lines, removed 2 lines
      39          self.model = model or DEFAULT_MODEL                                                                                                            
      40          self._kwargs = kwargs  # pass-through to litellm (api_key, api_base, etc.)
      41                                                                                                                                                         
      42 -    async def generate(self, prompt: str) -> str:                                                                                                    
      42 +    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:                                                                       
      43          try:                                                                                                                                           
      44              import litellm                                   
      45          except ImportError as e:
      46              raise ImportError("litellm required: pip install litellm") from e
      47  
      48 +        kwargs = dict(self._kwargs)                                                                                                             
      49 +        if max_tokens is not None:                                                                                                              
      50 +            kwargs["max_tokens"] = max_tokens                                                                                                   
      51          response = await litellm.acompletion(
      52              model=self.model,
      53              messages=[{"role": "user", "content": prompt}],
      54              temperature=0.1,
      52 -            **self._kwargs,                                                                                                                     
      55 +            **kwargs,                                                                                                                           
      56          )
      57          return response.choices[0].message.content or ""

● Now sentence_ids through the worker and pipeline:

● Update(vektori/utils/async_worker.py)                                                                                                                          
  ⎿  Added 1 line                             
      17      session_id: str                                                                                                                                    
      18      user_id: str                                                                    
      19      agent_id: str | None = None                                                                                                                        
      20 +    sentence_ids: list[str] | None = None  # IDs of sentences stored in this session                                                                 
      21                                                                                                                                                         
      22                                                                    
      23  class ExtractionWorker:                                   

● Update(vektori/utils/async_worker.py)
  ⎿  Added 2 lines, removed 1 line
      92              async with semaphore:                                                                                                                      
      93                  try:                                                     
      94                      await self._extractor.extract(                                                                                                     
      95 -                        req.messages, req.session_id, req.user_id, req.agent_id                                                                      
      95 +                        req.messages, req.session_id, req.user_id, req.agent_id,                                                                       
      96 +                        sentence_ids=req.sentence_ids,                                                                                               
      97                      )                                                                                                                                
      98                  except Exception as e:                                   
      99                      logger.error(