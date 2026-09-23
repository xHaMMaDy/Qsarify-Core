from services.query_intent import NEEDS_CLARIFICATION, OUT_OF_SCOPE, SUPPORTED, classify_query_intent


def test_biological_question_is_supported():
    result = classify_query_intent("Which human protein targets are associated with Alzheimer disease?")
    assert result.status == SUPPORTED
    assert "protein" in result.detected_concepts or "targets" in result.detected_concepts


def test_vague_research_question_requests_clarification():
    result = classify_query_intent("Which targets should I use?")
    assert result.status == NEEDS_CLARIFICATION
    assert result.clarifying_questions


def test_unrelated_question_is_out_of_scope():
    result = classify_query_intent("What is the weather today?")
    assert result.status == OUT_OF_SCOPE
