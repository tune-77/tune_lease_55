from lease_finance_knowledge import build_basic_lease_question_block
from api.main import _build_chat_basic_lease_question_context


def test_basic_lease_question_block_answers_finance_lease_without_rag():
    block = build_basic_lease_question_block("ファイナンスリースとは？")

    assert "基本リースQA" in block
    assert "ファイナンス・リース" in block
    assert "原則中途解約不可" in block
    assert "RAGやObsidian検索で直接ノートが0件でも" in block


def test_basic_lease_question_block_answers_common_residual_and_insurance_questions():
    block = build_basic_lease_question_block("残価と動産保険の見方を教えて")

    assert "残価は満了時" in block
    assert "中古市場" in block
    assert "動産総合保険" in block
    assert "盗難・火災・破損" in block


def test_basic_lease_question_block_answers_truck_useful_life():
    block = build_basic_lease_question_block("トラックの法定耐用年数は？")

    assert "トラック一般" in block
    assert "5年" in block
    assert "中型トラック" in block
    assert "4年" in block


def test_next_chat_uses_same_basic_lease_question_block():
    block = _build_chat_basic_lease_question_context("ファイナンスリースとは？")

    assert "基本リースQA" in block
    assert "ファイナンス・リース" in block
    assert "RAGやObsidian検索で直接ノートが0件でも" in block


def test_next_chat_basic_lease_question_block_answers_truck_useful_life():
    block = _build_chat_basic_lease_question_context("トラックの法定耐用年数は？")

    assert "トラック一般" in block
    assert "5年" in block
    assert "中型トラック" in block
    assert "4年" in block
