# AI Website Link Analysis API

FastAPI service that accepts a user-supplied public company website, extracts useful page content, and returns validated business, competitor, search-query, SEO, service, and audience insights from a configurable LLM.

> **Evidence note:** This application fetches the submitted page only. It does not independently crawl competitor websites, check search rankings, or query keyword-volume services. Competitor names are therefore returned as AI-generated, unverified suggestions unless supporting evidence is available in the submitted page. No search-volume or benchmark claims are fabricated.

## Features

- `POST /link-analysis` accepts any syntactically valid public HTTP(S) website.
- Extracts title, meta description, headings, paragraphs, useful same-site links, and service/product descriptions.
- DNS/IP and redirect checks, DNS-pinned outbound connections, private-address blocking, response-size limits, and bounded timeouts.
- Five typed analysis sections with Pydantic response validation.
- Groq, OpenAI, Google Gemini, and Anthropic adapters, selectable through environment variables.
- Provider token-usage reporting when supplied by the API, plus bounded in-memory TTL caching.
- Clear API errors, `/health`, and interactive Swagger UI at `/docs`.
- Tests use mocked network and LLM results; no paid API key is needed to run tests.
- [Local verification log](./verification.log) records the executed test and server smoke checks.

## Architecture

```text
app/
  main.py                     FastAPI routes, dependency injection, and error mapping
  config.py                   Environment-based settings
  schemas.py                  Validated request/response data models
  services/
    website_extractor.py      SSRF-aware retrieval and HTML extraction
    llm_service.py            Provider adapters and token usage
    analysis_service.py       Orchestration and response cache
    competitor_service.py     Evidence-aware prompt construction
  utils/
    validators.py             URL and public-address validation
tests/
  test_api.py                 API contract and error integration tests
  test_analysis_service.py    Orchestration and cache tests
  test_extractor.py           URL, SSRF, timeout, redirect, and HTML tests
  test_llm_service.py         Provider output and failure tests
```

The extractor requests only the submitted page; it does not follow links to crawl a whole site. Redirect destinations are checked before each request. The transport resolves DNS again immediately before TCP connection, rejects any non-public result, and connects to the vetted IP while preserving HTTPX TLS verification.

## Requirements and installation

- Python 3.12+
- An API key for one provider for live analysis; tests do not need one.

Windows PowerShell:

```powershell
cd ai-link-analysis
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `LLM_API_KEY` in `.env` to the secret API key for the selected provider. For Groq, you may instead set `GROQ_API_KEY`. Do not commit `.env`. The `.gitignore` excludes it. The API returns `503` if the configured key is missing.

### Provider configuration

| `LLM_PROVIDER` | Default model | Key expected |
|---|---|---|
| `groq` | `openai/gpt-oss-20b` | `GROQ_API_KEY` or `LLM_API_KEY` |
| `anthropic` | `claude-haiku-5-5` | Anthropic API key |
| `openai` | Set `LLM_MODEL` to `gpt-5.4-mini` | OpenAI API key |
| `gemini` | Set `LLM_MODEL` to `gemini-4-argon` | Google AI Studio / Gemini API key |

Set both `LLM_PROVIDER` and `LLM_MODEL` together when switching providers. Provider IDs and pricing change; confirm them against official documentation before deploying.

Other environment settings:

| Variable | Default | Description |
|---|---:|---|
| `LLM_TIMEOUT_SECONDS` | `45` | Provider request timeout |
| `WEBSITE_TIMEOUT_SECONDS` | `12` | Website connect/read timeout |
| `MAX_RESPONSE_BYTES` | `2000000` | Maximum downloaded page size |
| `MAX_CONTENT_CHARS` | `18000` | Approximate upper character bound for LLM input content |
| `CACHE_TTL_SECONDS` | `1800` | In-memory response cache lifetime; zero disables caching |
| `CACHE_MAX_ENTRIES` | `256` | Maximum in-memory cached analyses |
| `APIFY_API_TOKEN` | Empty | Optional Apify token for Instagram profile fallback |
| `APIFY_INSTAGRAM_ACTOR_ID` | Empty | Optional Apify Instagram profile actor ID |

## Supported link types

- Ordinary HTTP(S) websites use secure HTTPX and BeautifulSoup extraction.
- Instagram profiles use public HTML metadata and report login/bot/access restrictions explicitly. An optional Apify actor can provide richer public profile data when `APIFY_API_TOKEN` and `APIFY_INSTAGRAM_ACTOR_ID` are configured.
- YouTube channels, videos, and Shorts use the public YouTube oEmbed metadata endpoint.
- Facebook pages/posts and X/Twitter profiles/posts use public HTML metadata when accessible; login walls and blocks are reported.
- Social extraction does not infer followers, engagement, post counts, or products unless public source evidence supports them.
- If Groq is rate-limited or unavailable, social and website requests return an explicitly labeled evidence-safe degraded analysis rather than invented brand facts.

## Run locally

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 4000
```

- API root: http://localhost:4000/
- Health: http://localhost:4000/health
- Swagger: http://localhost:4000/docs

Example request:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:4000/link-analysis `
  -ContentType 'application/json' `
  -Body '{"url":"https://example.com"}'
```

Equivalent curl:

```sh
curl -X POST http://localhost:4000/link-analysis \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com"}'
```

## API contract

Request:

```json
{"url": "https://example.com"}
```

Response shape (illustrative structure; values are generated from the submitted page and configured model):

```json
{
  "success": true,
  "website": "https://example.com",
  "brand": {
    "name": "Example",
    "category": "Industry inferred from the page",
    "summary": "Summary based on extracted page content"
  },
  "analysis": {
    "competitor_analysis": {
      "brand_overview": "Evidence-led overview",
      "industry": "Industry",
      "competitors": [],
      "brand_strengths": [],
      "opportunities": [],
      "research_limitations": "External research was not performed"
    },
    "target_search_queries": [
      {
        "query": "Example service for business customers",
        "platforms": ["Google"],
        "intent": "Service discovery",
        "business_relevance": "Based on an observed page service",
        "target_audience": "Business customers"
      },
      {
        "query": "Example service provider for businesses",
        "platforms": ["Facebook"],
        "intent": "Business discovery",
        "business_relevance": "Based on an observed page service",
        "target_audience": "Business customers"
      },
      {
        "query": "Example product solutions",
        "platforms": ["Instagram"],
        "intent": "Product discovery",
        "business_relevance": "Based on an observed page product",
        "target_audience": "Potential buyers"
      },
      {
        "query": "Example services for organizations",
        "platforms": ["LinkedIn"],
        "intent": "B2B service discovery",
        "business_relevance": "Based on an observed page service",
        "target_audience": "Organizations"
      },
      {
        "query": "Example service recommendations",
        "platforms": ["X"],
        "intent": "Service discovery",
        "business_relevance": "Based on an observed page service",
        "target_audience": "Potential customers"
      }
    ],
    "seo_improvements": {
      "primary_keyword": "Relevant primary keyword",
      "secondary_keywords": [],
      "long_tail_keywords": [],
      "recommended_title": "Suggested SEO title",
      "meta_description": "Suggested description",
      "heading_improvements": [],
      "content_optimization": [],
      "supporting_content": [],
      "blog_topics": [],
      "faq_recommendations": [],
      "technical_seo": []
    },
    "top_services": [
      {
        "service_name": "Example observed service",
        "description": "Summary of a service supported by website content",
        "target_industry": "Industry",
        "target_customer": "Customer",
        "main_benefit": "Benefit",
        "source_status": "observed"
      },
      {
        "service_name": "Example recommendation 2",
        "description": "A clearly labeled business service suggestion",
        "target_industry": "Industry",
        "target_customer": "Customer",
        "main_benefit": "Potential benefit",
        "source_status": "recommended"
      },
      {
        "service_name": "Example recommendation 3",
        "description": "A clearly labeled business service suggestion",
        "target_industry": "Industry",
        "target_customer": "Customer",
        "main_benefit": "Potential benefit",
        "source_status": "recommended"
      },
      {
        "service_name": "Example recommendation 4",
        "description": "A clearly labeled business service suggestion",
        "target_industry": "Industry",
        "target_customer": "Customer",
        "main_benefit": "Potential benefit",
        "source_status": "recommended"
      },
      {
        "service_name": "Example recommendation 5",
        "description": "A clearly labeled business service suggestion",
        "target_industry": "Industry",
        "target_customer": "Customer",
        "main_benefit": "Potential benefit",
        "source_status": "recommended"
      }
    ],
    "target_audience": {
      "primary_audience": [],
      "secondary_audience": [],
      "industries": [],
      "customer_profiles": [],
      "customer_pain_points": [],
      "customer_interests": [],
      "geographic_markets": [],
      "potential_requirements": []
    }
  },
  "usage": {
    "provider": "anthropic",
    "model": "claude-haiku-5-5",
    "input_tokens": 1000,
    "output_tokens": 700,
    "total_tokens": 1700,
    "cache_hit": false
  }
}
```

The response schema requires exactly five search queries and five service entries, with Google, Facebook, Instagram, LinkedIn, and X all represented. Services can be marked `observed` or `recommended`. Competitors must carry an evidence status and evidence note. No example response above represents a real completed analysis.

### HTTP errors

- `400`: unsafe/private network destination
- `422`: invalid request or insufficient readable page content
- `502`: website or LLM upstream error, or malformed model output
- `503`: missing or unsupported provider configuration
- `504`: website timeout
- `500`: unexpected internal error (details are logged server-side)

## LLM provider comparison and cost

Published API rates below are USD per 1 million text tokens, checked against official model/pricing documentation on 2026-11-19. They are list prices, not measured per-request costs. Rates may vary by tier, region, prompt length, caching, batch mode, or provider changes.

| Provider / model | Input / output | Context | Published speed information | Website-analysis quality (not benchmarked here) | JSON support | Integration complexity |
|---|---:|---:|---|---|---|---|
| OpenAI `gpt-5.4-mini` | $0.75 / $4.50 | 400k tokens | Described as a fast, efficient mini model; no latency measured here | No comparative quality evaluation was run; general-purpose candidate to test | JSON mode in this adapter; model documentation lists structured outputs | Low: one authenticated JSON REST request |
| Google Gemini `gemini-4-argon` | $2.00 / $10.00 through Nov 30, 2026; scheduled $4 / $20 from Dec 1, 2026 | 1M tokens | Provider describes it as faster/lower-token-spend than Gemini 4 Carbon; no latency measured here | No comparative quality evaluation was run; large-context option | JSON MIME output in this adapter; API documents JSON Schema output | Low: one authenticated JSON REST request |
| Anthropic `claude-haiku-5-5` | $0.10 / $0.50 up to 100k input tokens | 1M tokens | Anthropic labels it fastest in the current lineup; no latency measured here | No comparative quality evaluation was run; reasonable low-cost model to evaluate for bounded extraction/analysis | Schema-constrained JSON output in this adapter | Medium: Messages API plus schema-output configuration |

**Recommendation:** Groq is the configured default for this project, using `openai/gpt-oss-20b` and `GROQ_API_KEY`. Claude Haiku 5.5 remains a cost-sensitive alternative, while `gpt-5.4-mini` may justify its higher published rate if a quality evaluation shows an advantage. Gemini offers a large context window and JSON Schema support, but is currently more expensive for this workload. These are engineering recommendations, not measured accuracy/latency rankings; this project includes no provider benchmark results.

The illustrative budget assumes 2,000 input and 1,000 output tokens per analysis; this is not measured usage. At that assumption, un-cached token charges based on the cited rates are approximately:

- OpenAI GPT-5.4 mini: `$0.006`
- Gemini 4 Argon: `$0.014` at the rate through November 30, 2026
- Claude Haiku 5.5: `$0.0007`

Actual usage is reported in the API response when the provider returns it. Expected wall-clock latency is not estimated here because it has not been measured; it depends on model, output length, provider load, and network conditions. This estimate excludes any separate tool, search-grounding, tax, or account-level charges.

Official references:

- [OpenAI GPT-5.4 mini pricing and model details](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
- [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)
- [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini model catalog](https://ai.google.dev/gemini-api/docs/models)
- [Gemini structured outputs](https://ai.google.dev/gemini-api/docs/structured-output)
- [Anthropic Haiku 5.5 model overview and pricing](https://platform.claude.com/docs/en/models/haiku-5-5/overview)
- [Anthropic API pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)

### Cost and prompt controls

- Strip navigation, scripts, styles, repeated text, and non-HTML pages before prompting.
- Cap the extracted page and response sizes.
- Request all five sections in one provider call and validate once rather than chaining prompts.
- Cache identical page/model analyses in a bounded TTL cache; cache is process-local and clears on restart.
- Use provider-reported token usage and re-evaluate model choice against an actual workload before changing production defaults.

## Testing

```powershell
pytest -q
```

Tests mock website and LLM traffic; no paid API calls are made. They cover valid/invalid/missing URLs, local/private targets, redirect safety, extraction and empty pages, timeouts, HTTP errors, malformed model JSON, provider failures, token usage, endpoints, and unexpected errors.

## Deployment (Render)

`render.yaml` includes a Python 3.12 web service, `/health` check, and production command:

```sh
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

1. Push this project to a public GitHub repository.
2. In Render, create a new Blueprint from the repository, or create a Python web service with `pip install -r requirements.txt` as the build command and the startup command above.
3. Configure `LLM_PROVIDER`, `LLM_MODEL`, and for Groq, `GROQ_API_KEY` (or `LLM_API_KEY`) as secret environment variables in Render.
4. Wait for a successful deploy, then verify `/health`, `/docs`, and a live POST request with a valid provider key.
5. Add the real deployment links to your submission only after the service is live.

- Public API URL: https://ai-link-analysis-wci2.onrender.com
- Health: https://ai-link-analysis-wci2.onrender.com/health
- Swagger: https://ai-link-analysis-wci2.onrender.com/docs
- Analysis endpoint: https://ai-link-analysis-wci2.onrender.com/link-analysis

The current deployment is hosted on Render and deployed from the `main` branch of this GitHub repository.

## Limitations

- Only the requested public URL is fetched. Multi-page crawling, robots/sitemap parsing, browser-rendered application data, followers, engagement counts, and external competitor-site verification are not guaranteed.
- Competitor output is not market research. Suggestions are labeled unverified and should be validated before business decisions.
- Search terms are model-generated recommendations, not search-volume or ranking results.
- Technical SEO observations are limited to fetched HTML; no Lighthouse, schema, canonical, robots, backlink, or Search Console audits are run.
- In-memory cache is not shared across workers or restarts.
- The default Groq model and published provider prices can change; confirm provider docs and account availability before production.
- Website and LLM calls can contain personal/business data. Review each provider's data terms and configure retention/region to meet your requirements.
- The SSRF-safe website transport intentionally bypasses environment proxies so host DNS can be validated and pinned. Run it where direct outbound HTTP(S) egress is permitted; a mandatory proxy needs a separately designed proxy integration that preserves the same destination guarantees.
- A production deployment should add distributed caching, rate limiting, authentication, observability, and an external threat/penetration review.

## GitHub submission

```powershell
git init
git add .
git status
git commit -m "Build AI website link analysis API"
git branch -M main
git remote add origin https://github.com/hemanthganiga/ai-link-analysis.git
git push -u origin main
```

Ensure `.env` is not staged and configure secret keys only in the deployment platform.
