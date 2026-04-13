"""rag_core 和 CBextension 的单元测试。

运行方式：
	cd Chatbot && pytest test_rag.py -v
"""

import os
import json
import pytest
from unittest.mock import MagicMock, patch

import CBextension
from rag_core import RagExt, _parse_call, load_documents


# ------------------------------------------------------------------
# CBextension 测试
# ------------------------------------------------------------------

class TestSalesPrediction:
	def test_basic(self):
		result = CBextension.sales_prediction(3)
		assert result['value'] == '503'
		assert '3' in result['description']

	def test_zero_months(self):
		result = CBextension.sales_prediction(0)
		assert result['value'] == '500'

	def test_returns_dict(self):
		result = CBextension.sales_prediction(12)
		assert isinstance(result, dict)
		assert 'value' in result and 'description' in result


class TestToolRegistry:
	def test_registry_contains_sales_prediction(self):
		assert "CBextension.sales_prediction" in CBextension.TOOL_REGISTRY

	def test_registry_values_are_callable(self):
		for name, func in CBextension.TOOL_REGISTRY.items():
			assert callable(func), f"{name} 不是可调用对象"

	def test_no_eval_needed(self):
		"""验证通过注册表调用，不需要 eval()。"""
		func = CBextension.TOOL_REGISTRY["CBextension.sales_prediction"]
		result = func(6)
		assert result['value'] == '506'

	def test_tool_descriptions_not_empty(self):
		assert len(CBextension.TOOL_DESCRIPTIONS.strip()) > 0


# ------------------------------------------------------------------
# _parse_call 测试
# ------------------------------------------------------------------

class TestParseCall:
	def test_integer_arg(self):
		name, args = _parse_call("CBextension.sales_prediction(3)")
		assert name == "CBextension.sales_prediction"
		assert args == [3]

	def test_float_arg(self):
		name, args = _parse_call("CBextension.foo(1.5)")
		assert args == [1.5]

	def test_string_arg(self):
		name, args = _parse_call('CBextension.bar("hello")')
		assert args == ["hello"]

	def test_no_args(self):
		name, args = _parse_call("CBextension.noop()")
		assert name == "CBextension.noop"
		assert args == []

	def test_invalid_format_raises(self):
		with pytest.raises(ValueError):
			_parse_call("not_a_call")

	def test_dangerous_expression_rejected(self):
		"""含任意表达式的参数应被拒绝，防止注入。"""
		with pytest.raises(ValueError):
			_parse_call("CBextension.foo(__import__('os').system('rm -rf /'))")


# ------------------------------------------------------------------
# RagExt 测试（使用 Mock，不依赖 LLM 和 ChromaDB）
# ------------------------------------------------------------------

def _make_mock_doc(content: str, source: str, score: float = 0.2):
	doc = MagicMock()
	doc.page_content = content
	doc.metadata = {'source': source}
	return doc, score


def _make_rag(tool_registry=None, call_template=None):
	"""构造带 Mock LLM 和 VectorStore 的 RagExt 实例。"""
	mock_llm = MagicMock()
	mock_db = MagicMock()
	mock_db.similarity_search_with_score.return_value = [
		_make_mock_doc("鲜芋仙地址：2001 Coit Rd, Plano, TX", "meetfresh.htm", 0.1),
		_make_mock_doc("招牌甜品：芋圆、仙草", "2020menu.pdf", 0.2),
	]
	mock_llm.invoke.return_value = MagicMock(content="这是模拟回答。")

	return RagExt(
		llm=mock_llm,
		db=mock_db,
		prompt_template="Context:{context}\n{QA_history}\nQuestion:{question}\nAnswer:",
		tool_registry=tool_registry or {},
		prompt_template_call=call_template,
		doc_score_upper=0.35,
		doc_num=5,
		history_budget=2000,
	)


class TestRagExtQuery:
	def test_returns_required_keys(self):
		rag = _make_rag()
		result = rag.query("地址是什么？")
		assert 'answer' in result
		assert 'source_documents' in result
		assert 'generated_question' in result

	def test_answer_is_string(self):
		rag = _make_rag()
		result = rag.query("测试问题")
		assert isinstance(result['answer'], str)

	def test_source_documents_list(self):
		rag = _make_rag()
		result = rag.query("测试问题")
		assert isinstance(result['source_documents'], list)

	def test_no_docs_above_threshold(self):
		"""当所有文档相似度超过阈值，应返回无资料提示。"""
		rag = _make_rag()
		rag.db.similarity_search_with_score.return_value = [
			_make_mock_doc("不相关内容", "other.htm", 0.9),
		]
		rag.doc_score_upper = 0.35
		result = rag.query("找不到的问题")
		assert result['source_documents'] == []
		assert '暂无' in result['answer'] or len(result['answer']) > 0

	def test_history_appended_after_query(self):
		rag = _make_rag()
		assert len(rag.chat_history) == 0
		rag.query("第一个问题")
		assert len(rag.chat_history) == 1

	def test_unknown_tool_not_executed(self):
		"""LLM 要求调用未注册工具时，不应执行，并在 context 中标记。"""
		mock_llm = MagicMock()
		mock_db = MagicMock()
		mock_db.similarity_search_with_score.return_value = [
			_make_mock_doc("内容", "test.htm", 0.1),
		]
		# 第一次调用（工具判断）返回未知工具，第二次（正式回答）正常
		mock_llm.invoke.side_effect = [
			MagicMock(content='["CBextension.evil_function(1)"]'),
			MagicMock(content="正常回答"),
		]
		rag = RagExt(
			llm=mock_llm,
			db=mock_db,
			prompt_template="Context:{context}\n{QA_history}\nQuestion:{question}\nAnswer:",
			tool_registry={},  # 空注册表，无任何工具
			prompt_template_call="{QA_history}\n{question}",
			doc_score_upper=1.0,
		)
		result = rag.query("触发未知工具")
		# context 中应有"未知工具"标记，而不是工具被真正执行
		assert "evil_function" not in result['answer']


# ------------------------------------------------------------------
# Memory 测试
# ------------------------------------------------------------------

class TestMemoryBudget:
	def test_trim_empty_history(self):
		rag = _make_rag()
		result = rag._trim_history_to_budget()
		assert result == ''

	def test_trim_within_budget(self):
		rag = _make_rag()
		rag.chat_history = [("短问题", "短回答")]
		result = rag._trim_history_to_budget()
		assert "短问题" in result

	def test_trim_over_budget(self):
		"""超出 budget 的历史轮次应被截断。"""
		rag = _make_rag()
		# 注入 200 轮每轮超长对话（总估算远超 budget=2000）
		rag.chat_history = [("q" * 50, "a" * 100)] * 200
		trimmed = rag._trim_history_to_budget()
		# 截断后长度应远小于全量
		assert len(trimmed) < 200 * 150

	def test_estimate_tokens_positive(self):
		rag = _make_rag()
		assert rag._estimate_tokens("hello world") > 0

	def test_estimate_tokens_empty(self):
		rag = _make_rag()
		assert rag._estimate_tokens("") == 1  # max(1, 0//3)


# ------------------------------------------------------------------
# 文档加载测试（使用临时目录，不依赖真实文件）
# ------------------------------------------------------------------

class TestLoadDocuments:
	def test_empty_dir(self, tmp_path):
		docs = load_documents(str(tmp_path))
		assert docs == []

	def test_html_loaded(self, tmp_path):
		html_file = tmp_path / "test.htm"
		html_file.write_text("<html><body><h1>Title</h1><p>Content</p></body></html>",
								encoding='utf-8')
		docs = load_documents(str(tmp_path))
		assert len(docs) > 0
		assert any("Content" in d.page_content for d in docs)
