"""utils package init.

This package exposes submodules (eg. monitoring) and also re-exports
legacy helper functions defined in the top-level `utils.py` module so
existing imports like `from utils import _slider_and_number` continue
to work after introducing this package.
"""
from . import monitoring

__all__ = ["monitoring"]

# Attempt to load legacy top-level utils.py and re-export common helpers
import importlib.util
import pathlib
import sys
_root_utils_path = pathlib.Path(__file__).resolve().parents[1] / "utils.py"
_logger = None
try:
	import logging
	_logger = logging.getLogger(__name__)
except Exception:
	_logger = None


def _load_legacy_utils():
	"""Load top-level utils.py into a temporary namespace and expose selected names."""
	try:
		import runpy
		ns = runpy.run_path(str(_root_utils_path))
		for name in ("_slider_and_number", "_reset_shinsa_inputs"):
			if name in ns:
				globals()[name] = ns[name]
				if name not in __all__:
					__all__.append(name)
		if _logger:
			_logger.debug("Loaded legacy utils from %s", _root_utils_path)
		return True
	except Exception as e:
		if _logger:
			_logger.exception("Failed to load legacy utils: %s", e)
		return False


if _root_utils_path.exists():
	_load_legacy_utils()


def __getattr__(name: str):
	# Lazy-load legacy helpers on attribute access
	if name in ("_slider_and_number", "_reset_shinsa_inputs"):
		if name in globals():
			return globals()[name]
		if _root_utils_path.exists():
			ok = _load_legacy_utils()
			if ok and name in globals():
				return globals()[name]
	raise AttributeError(f"module 'utils' has no attribute '{name}'")


# --- Backwards-compatible copies of small UI helpers ---
try:
	import streamlit as st  # keep local import to avoid requiring at module import time in some cases
except Exception:
	st = None


def _slider_and_number(field_name, key_prefix, default, min_val, max_val, step_slider, step_num=None, fmt="{:,}", unit="千円", label_slider="売上高調整", max_val_number=None, unit_factor=1):
	if st is None:
		raise RuntimeError("streamlit is not available for _slider_and_number")
	if step_num is None:
		step_num = step_slider
	num_max = max_val_number if max_val_number is not None else max_val

	if field_name not in st.session_state:
		st.session_state[field_name] = default
	cur = st.session_state[field_name]

	use_mansen = (unit_factor != 1)
	if use_mansen:
		display_unit = "百万円"
		step_slider_d = 1.0
		step_num_d = 0.1
		min_d   = min_val / unit_factor
		max_d   = max_val / unit_factor
		num_max_d = num_max / unit_factor
		cur_d   = cur / unit_factor
	else:
		display_unit = unit
		step_slider_d = step_slider
		step_num_d = step_num
		min_d   = min_val
		max_d   = max_val
		num_max_d = num_max
		cur_d   = cur

	prev_key      = f"_san_prev_{key_prefix}"
	num_key       = f"num_{key_prefix}"
	slide_key     = f"slide_{key_prefix}"
	prev_num_key  = f"_san_prev_num_{key_prefix}"
	prev_slide_key = f"_san_prev_slide_{key_prefix}"
	externally_changed = st.session_state.get(prev_key) != cur

	if num_key not in st.session_state or externally_changed:
		st.session_state[num_key] = max(min_d, min(cur_d, num_max_d))
	if slide_key not in st.session_state or externally_changed:
		st.session_state[slide_key] = max(min_d, min(cur_d, max_d))

	c_l, c_r = st.columns([0.7, 0.3])
	with c_r:
		st.number_input("直接入力", min_value=min_d, max_value=num_max_d,
						step=step_num_d, key=num_key,
						label_visibility="collapsed")
	with c_l:
		st.slider(label_slider, min_value=min_d, max_value=max_d,
				  step=step_slider_d, key=slide_key,
				  label_visibility="collapsed")

	new_num   = st.session_state[num_key]
	new_slide = st.session_state[slide_key]
	prev_num  = st.session_state.get(prev_num_key, new_num)
	prev_slide = st.session_state.get(prev_slide_key, new_slide)

	num_changed   = new_num   != prev_num
	slide_changed = new_slide != prev_slide
	if num_changed and not slide_changed:
		adopted_d = new_num
	elif slide_changed and not num_changed:
		adopted_d = new_slide
	elif num_changed and slide_changed:
		adopted_d = new_num
	else:
		adopted_d = cur_d

	if use_mansen:
		adopted = round(adopted_d * unit_factor)
		caption_str = f"{adopted_d:.1f} {display_unit}"
	else:
		adopted = adopted_d
		caption_str = f"{fmt.format(adopted)} {display_unit}"

	st.session_state[field_name] = adopted
	st.session_state[prev_key]   = adopted
	st.session_state[prev_num_key]   = new_num
	st.session_state[prev_slide_key] = new_slide
	st.caption(f"**採用値: {caption_str}**")
	return adopted


def _reset_shinsa_inputs():
	if st is None:
		raise RuntimeError("streamlit is not available for _reset_shinsa_inputs")
	field_defaults = {
		"nenshu": 0,
		"item9_gross": 0,
		"rieki": 0,
		"item4_ord_profit": 0,
		"item5_net_income": 0,
		"item10_dep": 0,
		"item11_dep_exp": 0,
		"item8_rent": 0,
		"item12_rent_exp": 0,
		"item6_machine": 0,
		"item7_other": 0,
		"net_assets": 0,
		"total_assets": 0,
		"bank_credit": 0,
		"lease_credit": 0,
		"contracts": 1,
		"acquisition_cost": 0,
		"lease_term": 60,
		"acceptance_year": 2026,
	}
	for k, v in field_defaults.items():
		st.session_state[k] = v
	widget_prefixes = [
		"nenshuu", "sourieki", "rieki", "item4_ord_profit", "item5_net_income",
		"item10_dep", "item11_dep_exp", "item8_rent", "item12_rent_exp",
		"item6_machine", "item7_other", "net_assets", "total_assets",
		"bank_credit", "lease_credit", "contracts", "acquisition_cost",
	]
	for pfx in widget_prefixes:
		for pre in ("num_", "slide_", "_san_prev_", "_san_prev_num_", "_san_prev_slide_"):
			st.session_state.pop(f"{pre}{pfx}", None)
	for k in list(st.session_state.keys()):
		if k.startswith("qual_corr_"):
			st.session_state[k] = 0
	for k in ("last_submitted_inputs", "last_result", "current_case_id",
			  "selected_asset_index", "news_results", "selected_news_content"):
		st.session_state.pop(k, None)
	st.session_state["messages"] = []
	st.session_state["debate_history"] = []


__all__.extend(["_slider_and_number", "_reset_shinsa_inputs"])
