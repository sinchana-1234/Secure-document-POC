"""
AI Firewall gateway — a single shared Firewall instance used at every checkpoint:
  - document upload (input)   → scan before storing/embedding
  - user question  (input)    → guard before retrieval
  - generated answer (output) → guard before returning to the user

Mode is driven by settings.FIREWALL_MODE:
  "monitor" — detect + log only, never blocks (safe dry-run; default)
  "enforce" — raises PromptInjectionError on malicious input

Flip via .env (FIREWALL_MODE=enforce) once monitoring confirms no false positives.
"""
import logging

from ai_firewall import Firewall, Policy

from app.config import settings

logger = logging.getLogger("doc-poc.firewall")

# One configured instance, reused everywhere. Mode comes from config so monitor→enforce
# is a .env change, not a code change.
_policy = Policy(mode=settings.FIREWALL_MODE)
firewall = Firewall(_policy, logger=logger)

logger.info("AI Firewall initialized in '%s' mode.", settings.FIREWALL_MODE)