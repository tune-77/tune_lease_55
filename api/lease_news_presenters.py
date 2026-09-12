"""Stable JSON presenters shared by lease-news routes, dashboard, and chat."""
from __future__ import annotations


def lease_news_focus_to_dict(focus):
    if not focus or not getattr(focus, "available", False):
        return {"available": False}
    return {
        "available": True,
        "note_path": getattr(focus, "note_path", ""),
        "note_date": getattr(focus, "note_date", ""),
        "profile": getattr(focus, "profile", ""),
        "theme_summary": getattr(focus, "theme_summary", ""),
        "bucket_summary": getattr(focus, "bucket_summary", ""),
        "tag_summary": getattr(focus, "tag_summary", ""),
        "focus_lines": list(getattr(focus, "focus_lines", ()) or ()),
        "memo_lines": list(getattr(focus, "memo_lines", ()) or ()),
        "metrics_lines": list(getattr(focus, "metrics_lines", ()) or ()),
        "article_titles": list(getattr(focus, "article_titles", ()) or ()),
        "headline": getattr(focus, "headline", ""),
    }


def lease_news_brief_to_dict(brief):
    if not brief or not getattr(brief, "available", False):
        return {"available": False}
    return {
        "available": True,
        "prefecture": getattr(brief, "prefecture", ""),
        "region": getattr(brief, "region", ""),
        "geo_context": getattr(brief, "geo_context", ""),
        "national_headline": getattr(brief, "national_headline", ""),
        "national_focus_lines": list(getattr(brief, "national_focus_lines", ()) or ()),
        "regional_available": getattr(brief, "regional_available", False),
        "regional_title": getattr(brief, "regional_title", ""),
        "regional_summary_lines": list(getattr(brief, "regional_summary_lines", ()) or ()),
        "regional_usage_memo": getattr(brief, "regional_usage_memo", ""),
        "regional_tags": list(getattr(brief, "regional_tags", ()) or ()),
        "regional_source": getattr(brief, "regional_source", ""),
        "opening_line": getattr(brief, "opening_line", ""),
        "question_line": getattr(brief, "question_line", ""),
        "note_date": getattr(brief, "note_date", ""),
        "note_path": getattr(brief, "note_path", ""),
    }


def lease_news_reflection_to_dict(reflection):
    if not reflection or not getattr(reflection, "available", False):
        return {"available": False}
    return {
        "available": True,
        "note_path": getattr(reflection, "note_path", ""),
        "note_date": getattr(reflection, "note_date", ""),
        "theme_summary": getattr(reflection, "theme_summary", ""),
        "tag_summary": getattr(reflection, "tag_summary", ""),
        "headline": getattr(reflection, "headline", ""),
        "thought_lines": list(getattr(reflection, "thought_lines", ()) or ()),
        "tomorrow_lines": list(getattr(reflection, "tomorrow_lines", ()) or ()),
        "illustration_url": getattr(reflection, "illustration_url", ""),
        "continuity_days": getattr(reflection, "continuity_days", 0),
        "dominant_mood": getattr(reflection, "dominant_mood", ""),
        "self_narrative": getattr(reflection, "self_narrative", ""),
        "current_question": getattr(reflection, "current_question", ""),
        "memory_excerpt": getattr(reflection, "memory_excerpt", ""),
        "user_understanding": getattr(reflection, "user_understanding", ""),
        "user_curiosity": getattr(reflection, "user_curiosity", ""),
        "user_interests": list(getattr(reflection, "user_interests", ()) or ()),
        "observed_days": getattr(reflection, "observed_days", 0),
        "primary_goal": getattr(reflection, "primary_goal", ""),
        "secondary_goal": getattr(reflection, "secondary_goal", ""),
        "ultimate_goal": getattr(reflection, "ultimate_goal", ""),
        "ultimate_goal_status": getattr(reflection, "ultimate_goal_status", ""),
        "knowledge_available": getattr(reflection, "knowledge_available", False),
        "knowledge_scope": getattr(reflection, "knowledge_scope", ""),
        "indexed_notes": getattr(reflection, "indexed_notes", 0),
        "knowledge_source_count": getattr(reflection, "knowledge_source_count", 0),
        "knowledge_sources": list(getattr(reflection, "knowledge_sources", ()) or ()),
    }


def lease_news_actions_to_dict(actions):
    if not actions or not getattr(actions, "available", False):
        return {"available": False}
    return {
        "available": True,
        "date": getattr(actions, "date", ""),
        "note_path": getattr(actions, "note_path", ""),
        "json_path": getattr(actions, "json_path", ""),
        "summary": getattr(actions, "summary", ""),
        "action_items": [
            {
                "signal": getattr(item, "signal", ""),
                "affected_industries": list(getattr(item, "affected_industries", ()) or ()),
                "affected_assets": list(getattr(item, "affected_assets", ()) or ()),
                "risk_flags": list(getattr(item, "risk_flags", ()) or ()),
                "recommended_checks": list(getattr(item, "recommended_checks", ()) or ()),
                "condition_impacts": list(getattr(item, "condition_impacts", ()) or ()),
                "source_title": getattr(item, "source_title", ""),
                "source_path": getattr(item, "source_path", ""),
                "valid_until": getattr(item, "valid_until", ""),
                "confidence": getattr(item, "confidence", 0.0),
                "noise_score": getattr(item, "noise_score", 0.0),
            }
            for item in (getattr(actions, "action_items", ()) or ())
        ],
        "ignored_titles": list(getattr(actions, "ignored_titles", ()) or ()),
    }
