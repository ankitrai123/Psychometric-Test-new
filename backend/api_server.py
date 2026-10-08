"""FastAPI server for the hybrid assessment system.

Run:  uvicorn api_server:app --reload        (or: python api_server.py)
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import secrets
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

import exporters
from config import Settings, settings as default_settings
from database_models import Assessment, Database, DatabaseCacheStore, InviteUnavailable
from hybrid_assessment_engine import (AssessmentInput, HybridReportGenerator, HybridScoringEngine,
                                      LLMInterpretationCache, public_report)
from llm_providers import PROVIDERS, LLMUnavailable, build_provider, catalog, mask_key

log = logging.getLogger("assessment.api")
ADMIN_PAGE = Path(__file__).resolve().parent / "static" / "admin.html"


class AssessRequest(BaseModel):
    test_taker_id: str = Field(min_length=1, max_length=64, examples=["165943163"])
    name: str = Field(min_length=1, max_length=200, examples=["Ankit Kumar"])
    email: str | None = Field(default=None, max_length=320, examples=["ankit@example.com"])
    responses: dict[str, int | None] = Field(
        description="Question id -> response (1..scale_points). Unanswered questions may be omitted or null.",
        examples=[{"1": 2, "2": 4, "3": 5}],
        max_length=1000,
    )
    premium: bool = Field(default=False, description="Also generate the LLM-enhanced premium report.")
    invite_token: str | None = Field(default=None, max_length=64,
                                     description="Token from the candidate's invite link. When given, the "
                                                 "candidate's identity comes from the invite, not this body.")


class InviteRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200, examples=["Ananya Sharma"])
    email: str | None = Field(default=None, max_length=320, examples=["ananya@example.com"])
    candidate_id: str | None = Field(default=None, max_length=64,
                                     description="Leave empty to generate one (CND-<year>-<6 hex>).")
    role: str | None = Field(default=None, max_length=200, examples=["Senior Software Engineer"])
    valid_days: int | None = Field(default=None, ge=1, description="Days until the link expires.")


INVITE_ERRORS = {
    "invalid": (404, "This test link is not valid. Please check the link in your invitation."),
    "expired": (410, "This test link has expired. Please contact the recruitment team for a new one."),
    "revoked": (410, "This test link has been cancelled. Please contact the recruitment team."),
    "completed": (409, "This test has already been submitted. Each link can only be used once."),
}


def invite_error(reason: str) -> HTTPException:
    status, message = INVITE_ERRORS.get(reason, INVITE_ERRORS["invalid"])
    return HTTPException(status_code=status, detail={"code": reason, "message": message})


class LLMConfigRequest(BaseModel):
    provider: str = Field(examples=["nvidia"], description="Provider id from GET /api/admin/llm")
    model: str = Field(min_length=1, max_length=200, examples=["meta/llama-3.3-70b-instruct"])
    api_key: str | None = Field(default=None, max_length=500,
                                description="Omit to keep the key already stored for this provider.")
    base_url: str | None = Field(default=None, max_length=500, description="Only for Ollama / custom endpoints.")
    input_price_per_mtok: float | None = Field(default=None, ge=0,
                                               description="USD per million input tokens, for cost estimates. "
                                                           "Omit to keep the stored value.")
    output_price_per_mtok: float | None = Field(default=None, ge=0)
    test: bool = Field(default=True, description="Make a tiny test call before saving.")


class ModelListRequest(BaseModel):
    provider: str
    api_key: str | None = Field(default=None, max_length=500)
    base_url: str | None = Field(default=None, max_length=500)


TEST_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
               "additionalProperties": False}


class PremiumRequest(BaseModel):
    test_id: str = Field(description="Assessment id, or a test-taker id (uses their latest assessment).")


def create_app(cfg: Settings = default_settings, db: Database | None = None,
               generator: HybridReportGenerator | None = None) -> FastAPI:
    problems = cfg.validate()
    if problems and cfg.is_production:
        raise RuntimeError("invalid configuration: " + "; ".join(problems))
    for p in problems:
        log.warning("config: %s", p)
    if not cfg.admin_api_key:
        log.warning("ADMIN_API_KEY not set: results, analytics and exports are unauthenticated (development only)")

    db = db or Database(cfg)
    db.create_all()
    if generator is None:
        cache = LLMInterpretationCache(cfg.cache_max_entries, store=DatabaseCacheStore(db))
        generator = HybridReportGenerator(cfg=cfg, cache=cache)

    app = FastAPI(title="Hybrid Psychometric Assessment API", version="1.0.0",
                  description="Deterministic Sten scoring with optional LLM-enhanced premium reports "
                              "(Claude, NVIDIA NIM, OpenAI, Gemini and other OpenAI-compatible providers).")
    app.add_middleware(CORSMiddleware, allow_origins=list(cfg.cors_origins), allow_methods=["*"],
                       allow_headers=["*"])
    app.state.db, app.state.generator, app.state.cfg = db, generator, cfg

    # The interpreter's initial provider (from environment variables or an
    # injected test client) is the baseline restored when the dashboard
    # setting is removed.
    baseline = (generator.interpreter.provider, generator.interpreter.source,
                generator.interpreter.input_price, generator.interpreter.output_price)
    app.state.llm_version = None

    def sync_llm() -> None:
        """Apply the dashboard-configured provider if it changed (cheap
        timestamp check, so every worker converges on the same setting)."""
        version = db.llm_settings_version()
        if version == app.state.llm_version:
            return
        app.state.llm_version = version
        stored = db.get_llm_settings()
        if stored is None:
            generator.interpreter.set_provider(baseline[0], baseline[1], baseline[2], baseline[3])
            return
        try:
            provider = build_provider(cfg, stored.provider, stored.api_key, stored.model, stored.base_url)
        except ValueError as exc:
            log.error("stored LLM settings are invalid (%s); keeping the previous provider", exc)
            return
        generator.interpreter.set_provider(provider, "dashboard", stored.input_price_per_mtok,
                                           stored.output_price_per_mtok)

    def llm_status() -> dict[str, Any]:
        stored = db.get_llm_settings()
        interp = generator.interpreter
        info = interp.describe()
        info.update(key_hint=stored.key_hint if stored and info["source"] == "dashboard" else None,
                    base_url=stored.base_url if stored and info["source"] == "dashboard" else None,
                    input_price_per_mtok=interp.input_price, output_price_per_mtok=interp.output_price,
                    updated_at=stored.updated_at.isoformat() if stored else None)
        return info

    def resolve_key(provider: str, api_key: str | None, base_url: str | None) -> str | None:
        """A pasted key wins; otherwise reuse the stored or environment key for
        the same provider. A stored key is only reused for the same endpoint,
        so it can't be redirected to a different server."""
        if api_key:
            return api_key.strip()
        stored = db.get_llm_settings()
        if stored and stored.provider == provider and stored.api_key and stored.base_url == base_url:
            return stored.api_key
        if provider == "anthropic" and cfg.anthropic_api_key:
            return cfg.anthropic_api_key
        if provider == cfg.llm_provider and cfg.llm_api_key:
            return cfg.llm_api_key
        return None

    sync_llm()

    def require_admin(x_api_key: str | None = Header(default=None)) -> None:
        if cfg.admin_api_key and not (x_api_key and secrets.compare_digest(x_api_key, cfg.admin_api_key)):
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")

    def find_assessment(test_id: str) -> Assessment:
        a = db.get_assessment(test_id)
        if a is None:
            a = db.latest_for_test_taker(test_id)
        if a is None:
            raise HTTPException(status_code=404, detail=f"no assessment found for {test_id!r}")
        return a

    def full_report(a: Assessment) -> dict[str, Any]:
        report = dict(a.report)
        if a.premium_report:
            report = a.premium_report
        return {**report, "assessment_id": a.id, "submitted_at": a.submitted_at.isoformat()}

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "llm_available": generator.interpreter.available,
                "norms": generator.engine.norms_status, "scoring_key": generator.bank.key_status,
                "invite_required": cfg.require_invite}

    @app.get("/api/questions")
    def questions() -> dict[str, Any]:
        """Question texts and scale (no scoring key)."""
        return {"scale_points": cfg.scale_points,
                "questions": [{"id": it.id, "text": it.text} for it in generator.bank.items.values()]}

    @app.post("/api/assess")
    def assess(body: AssessRequest) -> dict[str, Any]:
        if cfg.require_invite and not body.invite_token:
            raise HTTPException(status_code=403, detail={"code": "invite_required",
                                                         "message": "A personal test link is required."})
        taker_id, name, email = body.test_taker_id, body.name, body.email
        if body.invite_token:
            # Identity comes from the invite the admin created, not the request.
            invite = db.get_invite(body.invite_token)
            if invite is None:
                raise invite_error("invalid")
            taker_id, name, email = invite["candidate_id"], invite["name"], invite["email"]
        data = AssessmentInput(taker_id, name, email, {k: v for k, v in body.responses.items() if v is not None})
        report = generator.standard_report(data)  # validates before the invite is used up
        responses = generator.validator.normalise(data.responses)
        if body.invite_token:
            try:
                db.claim_invite(body.invite_token)
            except InviteUnavailable as exc:
                raise invite_error(exc.reason) from exc
        try:
            assessment_id = db.save_assessment(
                test_taker_id=taker_id, name=name, email=email,
                instrument=generator.bank.instrument, scale_points=cfg.scale_points,
                responses=responses, report=public_report(report), proportions=report.get("_proportions", {}),
            )
        except Exception:
            if body.invite_token:
                db.release_invite(body.invite_token)
            raise
        if body.invite_token:
            db.attach_assessment(body.invite_token, assessment_id)
        if body.premium and report["status"] == "Completed":
            sync_llm()
            report = generator.premium_report(data, standard=report)
            db.save_premium_report(assessment_id, public_report(report))
        return {**public_report(report), "assessment_id": assessment_id}

    @app.get("/api/results/{test_id}", dependencies=[Depends(require_admin)])
    def results(test_id: str) -> dict[str, Any]:
        return full_report(find_assessment(test_id))

    @app.post("/api/generate-premium-report", dependencies=[Depends(require_admin)])
    def generate_premium(body: PremiumRequest) -> dict[str, Any]:
        a = find_assessment(body.test_id)
        if a.status != "Completed":
            raise HTTPException(status_code=409, detail="premium reports need a completed assessment")
        data = AssessmentInput(a.test_taker.external_id, a.test_taker.name, a.test_taker.email, {})
        sync_llm()
        report = generator.premium_report(data, standard=dict(a.report))
        db.save_premium_report(a.id, public_report(report))
        return {**public_report(report), "assessment_id": a.id}

    @app.post("/api/export/{test_id}", dependencies=[Depends(require_admin)])
    def export(test_id: str, format: Literal["pdf", "json", "text"] = Query("pdf")) -> Response:
        a = find_assessment(test_id)
        report = full_report(a)
        stem = f"assessment_{a.test_taker.external_id}_{a.id[:8]}"
        if format == "json":
            return Response(exporters.to_json(report), media_type="application/json",
                            headers={"Content-Disposition": f'attachment; filename="{stem}.json"'})
        if format == "text":
            return PlainTextResponse(exporters.to_text(report),
                                     headers={"Content-Disposition": f'attachment; filename="{stem}.txt"'})
        return Response(exporters.to_pdf(report), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{stem}.pdf"'})

    @app.get("/api/analytics", dependencies=[Depends(require_admin)])
    def analytics() -> dict[str, Any]:
        sync_llm()
        stats = db.analytics()
        stats["engine"] = generator.stats()
        stats["norms"] = generator.engine.norms_status
        return stats

    @app.get("/api/norms/suggested", dependencies=[Depends(require_admin)])
    def suggested_norms(min_sample: int = Query(30, ge=2)) -> dict[str, Any]:
        """Norms computed from stored genuine assessments. Save the result to
        NORMS_PATH (data/norms.json) and restart to use calibrated norms."""
        try:
            return HybridScoringEngine.compute_norms(db.proportion_rows(), min_sample=min_sample)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # -- LLM provider settings (admin dashboard) -----------------------------

    @app.get("/api/admin/llm", dependencies=[Depends(require_admin)])
    def get_llm_config() -> dict[str, Any]:
        """Current provider (the API key is never returned) and the provider catalog."""
        sync_llm()
        return {"current": llm_status(), "providers": catalog()}

    @app.put("/api/admin/llm", dependencies=[Depends(require_admin)])
    def put_llm_config(body: LLMConfigRequest) -> dict[str, Any]:
        if body.provider not in PROVIDERS:
            raise HTTPException(status_code=422, detail=f"unknown provider {body.provider!r}")
        base_url = (body.base_url or "").strip() or None
        api_key = resolve_key(body.provider, body.api_key, base_url)
        try:
            provider = build_provider(cfg, body.provider, api_key, body.model.strip(), base_url)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if body.test:
            try:
                result = provider.generate_json("You are a connectivity check. Reply in JSON.",
                                                'Return the JSON object {"ok": true}.', TEST_SCHEMA)
            except LLMUnavailable as exc:
                raise HTTPException(status_code=400, detail=f"Connection test failed: {exc}") from exc
            if result.data.get("ok") is not True:
                raise HTTPException(status_code=400, detail="Connection test failed: unexpected reply")
        previous = db.get_llm_settings()
        same = previous is not None and previous.provider == body.provider

        def price(given: float | None, stored: float | None) -> float:
            return given if given is not None else (stored if same and stored is not None else 0.0)

        db.save_llm_settings(provider=body.provider, model=body.model.strip(), base_url=base_url, api_key=api_key,
                             key_hint=mask_key(api_key),
                             input_price=price(body.input_price_per_mtok, previous and previous.input_price_per_mtok),
                             output_price=price(body.output_price_per_mtok, previous and previous.output_price_per_mtok))
        log.info("LLM provider set to %s / %s from the admin dashboard", body.provider, body.model)
        sync_llm()
        return {"current": llm_status(), "tested": body.test}

    @app.delete("/api/admin/llm", dependencies=[Depends(require_admin)])
    def delete_llm_config() -> dict[str, Any]:
        """Remove the dashboard setting and revert to the environment configuration."""
        db.clear_llm_settings()
        sync_llm()
        return {"current": llm_status()}

    @app.post("/api/admin/llm/models", dependencies=[Depends(require_admin)])
    def list_llm_models(body: ModelListRequest) -> dict[str, Any]:
        """List the models the given key can use (queried from the provider)."""
        if body.provider not in PROVIDERS:
            raise HTTPException(status_code=422, detail=f"unknown provider {body.provider!r}")
        base_url = (body.base_url or "").strip() or None
        try:
            provider = build_provider(cfg, body.provider, resolve_key(body.provider, body.api_key, base_url),
                                      "list-models", base_url)
            return {"models": provider.list_models()}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LLMUnavailable as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # -- Dashboard analysis ------------------------------------------------------

    @app.get("/api/admin/assessments", dependencies=[Depends(require_admin)])
    def list_assessments(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)) -> dict[str, Any]:
        return {"assessments": db.list_assessments(limit=limit, offset=offset)}

    @app.post("/api/admin/cohort-analysis", dependencies=[Depends(require_admin)])
    def cohort_analysis() -> dict[str, Any]:
        """LLM analysis of aggregated (anonymised) results across all candidates."""
        stats = db.analytics()
        if not stats["assessments"]["total"]:
            raise HTTPException(status_code=409, detail="no assessments to analyse yet")
        sync_llm()
        return generator.cohort_report(stats)

    # -- Candidate invitations --------------------------------------------------

    def with_link(invite: dict[str, Any]) -> dict[str, Any]:
        return {**invite, "link": f"{cfg.invite_base_url}/?invite={invite['token']}"}

    @app.post("/api/admin/invites", dependencies=[Depends(require_admin)], status_code=201)
    def create_invite(body: InviteRequest) -> dict[str, Any]:
        days = body.valid_days or cfg.invite_default_days
        if days > cfg.invite_max_days:
            raise HTTPException(status_code=422, detail=f"valid_days must be at most {cfg.invite_max_days}")
        invite = db.create_invite(name=body.name.strip(), email=(body.email or "").strip() or None,
                                  candidate_id=(body.candidate_id or "").strip() or None,
                                  role=(body.role or "").strip() or None, valid_days=days)
        return with_link(invite)

    @app.get("/api/admin/invites", dependencies=[Depends(require_admin)])
    def list_invites(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)) -> dict[str, Any]:
        return {"invites": [with_link(i) for i in db.list_invites(limit=limit, offset=offset)],
                "candidate_app_url": cfg.invite_base_url, "default_valid_days": cfg.invite_default_days,
                "invite_required": cfg.require_invite}

    @app.delete("/api/admin/invites/{token}", dependencies=[Depends(require_admin)])
    def revoke_invite(token: str) -> dict[str, Any]:
        invite = db.revoke_invite(token)
        if invite is None:
            raise HTTPException(status_code=404, detail="invite not found")
        return with_link(invite)

    @app.get("/api/invites/{token}")
    def open_invite(token: str) -> dict[str, Any]:
        """Candidate-facing: who the link is for and whether it can still be used."""
        invite = db.open_invite(token)
        if invite is None:
            raise invite_error("invalid")
        public = ("candidate_id", "name", "email", "role", "status", "created_at", "expires_at")
        return {k: invite[k] for k in public}

    @app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
    def admin_page() -> str:
        return ADMIN_PAGE.read_text(encoding="utf-8")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000)
