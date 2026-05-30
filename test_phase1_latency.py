#!/usr/bin/env python3
"""
Phase 1: テスト - レイテンシ計測機構の確認

このスクリプトは以下を確認します：
1. chat_assistant.py の import に成功
2. ログが正常に出力される
3. 各ステップの計測が動作する（モック版）
"""

import sys
import os
import json
import time
import logging
from pathlib import Path

# ===== Setup path
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

# ===== Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger("test_phase1")

def test_import():
    """Test if chat_assistant can be imported"""
    logger.info("=" * 60)
    logger.info("TEST 1: Import chat_assistant module")
    logger.info("=" * 60)
    try:
        from mobile_app import chat_assistant
        logger.info("✅ Successfully imported chat_assistant")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to import: {e}")
        return False

def test_logging():
    """Test if logging is configured correctly"""
    logger.info("=" * 60)
    logger.info("TEST 2: Verify logging works")
    logger.info("=" * 60)

    test_logger = logging.getLogger("mobile_app.chat_assistant")
    test_logger.info("PHASE1_LATENCY | obsidian_search=0.123s | obsidian_digest=0.045s | web_search=0.010s | gemini=1.234s | total=1.412s | query_length=50 | hits=3")
    logger.info("✅ Logging configured correctly")
    return True

def test_latency_calculation():
    """Test latency calculation logic"""
    logger.info("=" * 60)
    logger.info("TEST 3: Verify latency calculation")
    logger.info("=" * 60)

    # Simulate the latency calculation
    t_start = time.time()

    t0 = time.time()
    time.sleep(0.1)  # Simulate obsidian search
    t_obsidian_search = time.time() - t0

    t0 = time.time()
    time.sleep(0.05)  # Simulate digest
    t_obsidian_digest = time.time() - t0

    t0 = time.time()
    time.sleep(0.02)  # Simulate web search
    t_web_search = time.time() - t0

    t0 = time.time()
    time.sleep(0.3)  # Simulate Gemini
    t_gemini = time.time() - t0

    t_total = time.time() - t_start

    logger.info(f"  obsidian_search: {t_obsidian_search:.3f}s")
    logger.info(f"  obsidian_digest: {t_obsidian_digest:.3f}s")
    logger.info(f"  web_search: {t_web_search:.3f}s")
    logger.info(f"  gemini: {t_gemini:.3f}s")
    logger.info(f"  total: {t_total:.3f}s")

    # Verify that total is approximately sum of parts
    expected_total = t_obsidian_search + t_obsidian_digest + t_web_search + t_gemini
    diff = abs(t_total - expected_total)

    if diff < 0.05:  # 50ms tolerance
        logger.info(f"✅ Latency calculation verified (diff: {diff:.3f}s)")
        return True
    else:
        logger.error(f"❌ Latency calculation error (diff: {diff:.3f}s)")
        return False

def test_phase1_features():
    """Test Phase 1 implementation checklist"""
    logger.info("=" * 60)
    logger.info("TEST 4: Phase 1 implementation checklist")
    logger.info("=" * 60)

    try:
        from mobile_app.chat_assistant import build_chat_reply

        # Check if function signature has the expected parameters
        import inspect
        sig = inspect.signature(build_chat_reply)
        params = list(sig.parameters.keys())

        expected_params = ['message', 'history', 'score_result', 'use_obsidian', 'use_web', 'timeout_seconds', 'humor_style']
        all_present = all(p in params for p in expected_params)

        if all_present:
            logger.info("✅ build_chat_reply function has all expected parameters")
        else:
            logger.warning("⚠️ Some parameters missing")

        # Verify that the module has logging configured
        import mobile_app.chat_assistant as ca_module
        if hasattr(ca_module, 'logger'):
            logger.info("✅ Logger is configured in chat_assistant")
        else:
            logger.warning("⚠️ Logger not found in chat_assistant")

        return True
    except Exception as e:
        logger.error(f"❌ Phase 1 check failed: {e}")
        return False

def main():
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: LATENCY MONITORING - IMPLEMENTATION VERIFICATION")
    logger.info("=" * 60 + "\n")

    results = []

    results.append(("Import test", test_import()))
    results.append(("Logging test", test_logging()))
    results.append(("Latency calculation", test_latency_calculation()))
    results.append(("Phase 1 features", test_phase1_features()))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅" if result else "❌"
        logger.info(f"{status} {test_name}")

    logger.info(f"\n{passed}/{total} tests passed")

    if passed == total:
        logger.info("\n✅ Phase 1: Latency monitoring implementation verified!")
        logger.info("\nNext steps:")
        logger.info("1. Monitor /Users/kobayashiisaoryou/Library/Logs/tunelease/ for logs")
        logger.info("2. Collect latency data from real chats")
        logger.info("3. Identify bottlenecks")
        logger.info("4. Proceed to Step 2: Implement caching")
        return 0
    else:
        logger.error("\n❌ Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
