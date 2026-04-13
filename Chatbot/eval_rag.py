"""RAG 离线评估框架。

评估两个维度：
  1. Context Recall  — 检索结果是否覆盖期望来源文件
  2. Keyword Hit Rate — 生成答案是否包含期望关键词

使用方法：
	python eval_rag.py
"""

import argparse
import json
from dotenv import load_dotenv
from rag_core import RagExt, load_documents, build_vectorstore
import CBextension
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ------------------------------------------------------------------
# Golden Set：标准问答对，每条含期望来源文件和期望关键词
# ------------------------------------------------------------------
GOLDEN_SET: list[dict] = [
	{
		"question": "鲜芋仙达拉斯店的地址是什么？",
		"expected_keywords": ["Coit", "2001", "Dallas"],
		"expected_sources": ["meetfresh.htm", "meet freshenu_ prices and deliver - doordash.htm"],
	},
	{
		"question": "鲜芋仙有哪些招牌甜品？",
		"expected_keywords": ["芋圆", "仙草", "豆花", "taro"],
		"expected_sources": ["2020menu.pdf"],
	},
	{
		"question": "鲜芋仙的营业时间是什么？",
		"expected_keywords": ["pm", "am", "hour", "open"],
		"expected_sources": ["meetfresh.htm", "meet freshenu_ prices and deliver - doordash.htm"],
	},
	{
		"question": "刨冰有哪些口味可以选择？",
		"expected_keywords": ["shaved ice", "刨冰", "flavor"],
		"expected_sources": ["2020menu.pdf"],
	},
	{
		"question": "Meet Fresh 的价格范围大概是多少？",
		"expected_keywords": ["$", "price", "dollar"],
		"expected_sources": ["meet freshenu_ prices and deliver - doordash.htm", "2020menu.pdf"],
	},
]


# ------------------------------------------------------------------
# 评估函数
# ------------------------------------------------------------------

def eval_retrieval(rag: RagExt, golden_set: list[dict]) -> dict:
	"""Context Recall：检索结果是否包含至少一个期望来源文件。

	返回:
		{'context_recall': float, 'n': int, 'details': list}
	"""
	hits = 0
	details = []
	for item in golden_set:
		result = rag.query(item["question"])
		retrieved_sources = {
			d['doc'].metadata.get('source', '').lower()
			for d in result['source_documents']
		}
		expected = {s.lower() for s in item['expected_sources']}
		hit = bool(retrieved_sources & expected)
		if hit:
			hits += 1
		details.append({
			"question": item["question"],
			"hit": hit,
			"retrieved": sorted(retrieved_sources),
			"expected": sorted(expected),
		})
	return {
		"context_recall": hits / len(golden_set),
		"n": len(golden_set),
		"details": details,
	}


def eval_answer_keywords(rag: RagExt, golden_set: list[dict]) -> dict:
	"""Keyword Hit Rate：生成答案包含期望关键词的比例。

	返回:
		{'keyword_hit_rate': float, 'n': int, 'details': list}
	"""
	scores = []
	details = []
	for item in golden_set:
		result = rag.query(item["question"])
		answer_lower = result['answer'].lower()
		hits = [kw for kw in item['expected_keywords'] if kw.lower() in answer_lower]
		hit_rate = len(hits) / len(item['expected_keywords'])
		scores.append(hit_rate)
		details.append({
			"question": item["question"],
			"hit_rate": hit_rate,
			"hits": hits,
			"missed": [kw for kw in item['expected_keywords'] if kw.lower() not in answer_lower],
		})
	return {
		"keyword_hit_rate": sum(scores) / len(scores),
		"n": len(golden_set),
		"details": details,
	}


def print_report(retrieval: dict, keywords: dict) -> None:
	"""打印格式化评估报告。"""
	print("\n" + "=" * 60)
	print("RAG Evaluation Report")
	print("=" * 60)
	print(f"Context Recall    : {retrieval['context_recall']:.1%}  ({retrieval['n']} questions)")
	print(f"Keyword Hit Rate  : {keywords['keyword_hit_rate']:.1%}  ({keywords['n']} questions)")
	print()
	print("-- Context Recall Details --")
	for d in retrieval['details']:
		mark = "OK" if d['hit'] else "MISS"
		print(f"  [{mark}] {d['question']}")
		if not d['hit']:
			print(f"        retrieved: {d['retrieved']}")
			print(f"        expected : {d['expected']}")
	print()
	print("-- Keyword Hit Rate Details --")
	for d in keywords['details']:
		print(f"  [{d['hit_rate']:.0%}] {d['question']}")
		if d['missed']:
			print(f"        missed: {d['missed']}")
	print("=" * 60)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="RAG 离线评估")
	parser.add_argument('--data_dir', default='data', help='文档目录')
	parser.add_argument('--db_dir', default='./chroma_db', help='ChromaDB 目录')
	parser.add_argument('--doc_score_upper', type=float, default=0.35)
	parser.add_argument('--doc_num', type=int, default=5)
	parser.add_argument('--output', default='', help='结果输出 JSON 路径（可选）')
	args = parser.parse_args()

	# 初始化 RAG 引擎
	docs = load_documents(args.data_dir)
	db = build_vectorstore(docs, args.db_dir)
	llm = ChatGoogleGenerativeAI(
		model='gemini-1.5-pro',
		temperature=0,
		convert_system_message_to_human=True,
	)
	template = (
		"你是美国达拉斯一家鲜芋仙店铺的店长助理，"
		"请结合以下Context回答Question。\n\n"
		"Context:\n{context}\n\n{QA_history}\n\nQuestion: {question}\nAnswer:\n"
	)
	rag = RagExt(
		llm=llm,
		db=db,
		prompt_template=template,
		tool_registry=CBextension.TOOL_REGISTRY,
		doc_num=args.doc_num,
		doc_score_upper=args.doc_score_upper,
	)

	# 运行评估
	retrieval_result = eval_retrieval(rag, GOLDEN_SET)
	# 重置 history，避免两次评估相互干扰
	rag.chat_history = []
	keyword_result = eval_answer_keywords(rag, GOLDEN_SET)

	print_report(retrieval_result, keyword_result)

	if args.output:
		with open(args.output, 'w', encoding='utf-8') as f:
			json.dump(
				{"retrieval": retrieval_result, "keywords": keyword_result},
				f, ensure_ascii=False, indent=2,
			)
		print(f"Results saved to {args.output}")
