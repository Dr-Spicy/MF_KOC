"""RAG 核心模块：文档加载、向量库构建、带记忆的查询引擎。

使用方法：
	python rag_core.py --data_dir data --db_dir ./chroma_db --port 8002
"""

import os
import re
import json
import argparse
from typing import Optional

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import docx as python_docx

# 加载 .env 中的 API Key（不得硬编码）
load_dotenv()

# chunking 参数
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# HTML 按标题分割
_HEADERS_TO_SPLIT = [("h1", "H1"), ("h2", "H2"), ("h3", "H3"), ("h4", "H4")]


def load_documents(data_dir: str) -> list[Document]:
	"""从 data_dir 加载 PDF/DOCX/HTML 文件，返回结构化 Document 列表。

	PDF：先按页加载，再用 RecursiveCharacterTextSplitter 二次截断，
	     防止整页超过 context 限制。
	DOCX：按 Heading 样式分段，段落超过 CHUNK_SIZE*2 时进一步截断。
	HTML：按 h1-h4 标题标签分割。
	"""
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=CHUNK_SIZE,
		chunk_overlap=CHUNK_OVERLAP,
	)
	html_splitter = HTMLHeaderTextSplitter(_HEADERS_TO_SPLIT)
	docs: list[Document] = []

	for fname in os.listdir(data_dir):
		fpath = os.path.join(data_dir, fname)
		if not os.path.isfile(fpath):
			continue
		ext = fname.rsplit('.', 1)[-1].lower()

		if ext in ('docx', 'doc'):
			# 按 Heading 样式切段，过长时二次截断
			doc_obj = python_docx.Document(fpath)
			current_heading = ''
			current_section: list[str] = []
			for para in doc_obj.paragraphs:
				if para.style.name.startswith('Heading'):
					if current_section:
						content = '\n'.join(current_section)
						meta = {'source': fname, 'section': current_heading}
						if len(content) > CHUNK_SIZE * 2:
							docs += splitter.create_documents([content], metadatas=[meta])
						else:
							docs.append(Document(page_content=content, metadata=meta))
					current_heading = para.text
					current_section = [para.text] if para.text.strip() else []
				elif para.text.strip():
					current_section.append(para.text)
			if current_section:
				content = '\n'.join(current_section)
				meta = {'source': fname, 'section': current_heading}
				if len(content) > CHUNK_SIZE * 2:
					docs += splitter.create_documents([content], metadatas=[meta])
				else:
					docs.append(Document(page_content=content, metadata=meta))

		elif ext == 'pdf':
			# 先按页加载，再二次分割防止整页过长
			loader = PyPDFLoader(fpath)
			page_docs = loader.load()
			chunks = splitter.split_documents(page_docs)
			for chunk in chunks:
				chunk.metadata['source'] = fname
			docs += chunks

		elif ext in ('htm', 'html'):
			html_content = open(fpath, encoding='utf-8').read()
			splits = html_splitter.split_text(html_content)
			for doc in splits:
				doc.metadata['source'] = fname
			docs += splits

	return docs


def build_vectorstore(docs: list[Document], db_dir: str) -> Chroma:
	"""用 Google embedding-001 构建 ChromaDB，持久化到 db_dir。

	若 db_dir 已存在则直接加载，避免重复 embedding。
	"""
	embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
	if os.path.exists(db_dir) and os.listdir(db_dir):
		# 已有持久化索引，直接加载
		db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
	else:
		db = Chroma.from_documents(docs, embeddings, persist_directory=db_dir)
	return db


def _parse_call(call_str: str) -> tuple[str, list]:
	"""将 'CBextension.sales_prediction(3)' 解析为 (函数名, [参数列表])。

	仅支持整数、浮点数、字符串字面量参数，拒绝任意表达式，防止注入。
	"""
	match = re.fullmatch(r'([\w.]+)\((.*)\)', call_str.strip())
	if not match:
		raise ValueError(f"无法解析工具调用格式：{call_str!r}")
	func_name = match.group(1)
	raw_args = match.group(2).strip()
	args: list = []
	if raw_args:
		for part in raw_args.split(','):
			part = part.strip()
			# 只接受数字和带引号的字符串
			if re.fullmatch(r'-?\d+(\.\d+)?', part):
				args.append(float(part) if '.' in part else int(part))
			elif re.fullmatch(r'"[^"]*"|\'[^\']*\'', part):
				args.append(part[1:-1])
			else:
				raise ValueError(f"不支持的参数类型：{part!r}")
	return func_name, args


class RagExt:
	"""带对话记忆和工具调用的 RAG 查询引擎。

	参数:
		llm: LangChain LLM 实例
		db: LangChain VectorStore 实例
		prompt_template (str): 主问答提示词，含 {context}/{question}/{QA_history}
		tool_registry (dict): 工具名字符串 -> 可调用函数，替代 eval()
		prompt_template_call (str): 工具调用判断提示词模板（可选）
		history_budget (int): 历史注入到 prompt 的估算 token 上限，默认 2000
		doc_num (int): 每次检索文档数，默认 10
		doc_score_upper (float): 相似度分数上限（Chroma L2 距离越小越优）
	"""

	def __init__(
		self,
		llm,
		db,
		prompt_template: str,
		tool_registry: Optional[dict] = None,
		prompt_template_call: Optional[str] = None,
		history_budget: int = 2000,
		doc_num: int = 10,
		doc_score_upper: float = float('inf'),
	):
		self.llm = llm
		self.db = db
		self.prompt_template = prompt_template
		self.tool_registry: dict = tool_registry or {}
		self.call_prompt_template = prompt_template_call
		self.history_budget = history_budget
		self.doc_num = doc_num
		self.doc_score_upper = doc_score_upper
		self.chat_history: list[tuple[str, str]] = []

	# ------------------------------------------------------------------
	# Memory 相关方法
	# ------------------------------------------------------------------

	def _estimate_tokens(self, text: str) -> int:
		"""用字符数粗估 token 数（中英混合：约 3 字/token）。"""
		return max(1, len(text) // 3)

	def _trim_history_to_budget(self) -> str:
		"""从最新对话向前累加，不超过 history_budget 估算 token。

		保证 context window 安全，优先保留最近轮次。
		"""
		entries: list[str] = []
		used = 0
		for q, a in reversed(self.chat_history):
			entry = f"Question:{q}\nAnswer: {a}\n\n"
			cost = self._estimate_tokens(entry)
			if used + cost > self.history_budget:
				break
			entries.insert(0, entry)
			used += cost
		return ''.join(entries)

	# ------------------------------------------------------------------
	# 核心查询
	# ------------------------------------------------------------------

	def query(self, query: str) -> dict:
		"""执行一次 RAG 查询，返回答案、来源文档和最终 prompt。

		流程：
		  1. 用 token-aware 历史构建 QA_history
		  2. 检索向量库，过滤高分数文档
		  3. （可选）让 LLM 判断是否调用工具，安全执行注册表内函数
		  4. 拼接 context，生成最终答案
		"""
		qa_history = self._trim_history_to_budget()

		# 检索
		ref_docs = self.db.similarity_search_with_score(query, k=self.doc_num)
		ref_docs.sort(key=lambda x: x[1])  # 按相似度升序
		ref_docs = [
			{'doc': doc, 'score': score}
			for doc, score in ref_docs
			if score < self.doc_score_upper
		]

		if not ref_docs:
			return {
				'answer': '暂无相关参考资料，无法回答该问题。',
				'source_documents': [],
				'generated_question': query,
			}

		context = '\n\n'.join(item['doc'].page_content for item in ref_docs[:10])

		# 工具调用（可选）
		if self.call_prompt_template:
			call_prompt = (
				self.call_prompt_template
				.replace('{question}', query)
				.replace('{QA_history}', qa_history)
			)
			tool_answer = self.llm.invoke(call_prompt)
			tool_content = tool_answer.content.strip()
			if tool_content:
				results = ''
				try:
					calls = json.loads(tool_content)
					for call_str in calls:
						func_name, args = _parse_call(call_str)
						if func_name in self.tool_registry:
							result = self.tool_registry[func_name](*args)
							results += f"|{result['description']}:{result['value']}"
						else:
							results += f"|未知工具:{func_name}"
					context += '|分析结果:' + results
				except (json.JSONDecodeError, ValueError) as e:
					context += f'|工具解析失败:{e}'

		# 生成答案
		prompt = (
			self.prompt_template
			.replace('{question}', query)
			.replace('{QA_history}', qa_history)
			.replace('{context}', context)
		)
		answer = self.llm.invoke(prompt)
		self.chat_history.append((query, answer.content))

		return {
			'answer': answer.content,
			'source_documents': ref_docs[:10],
			'generated_question': prompt,
		}


# ------------------------------------------------------------------
# 独立运行入口
# ------------------------------------------------------------------

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="启动鲜芋仙 RAG Chatbot")
	parser.add_argument('--data_dir', default='data', help='文档目录')
	parser.add_argument('--db_dir', default='./chroma_db', help='ChromaDB 持久化目录')
	parser.add_argument('--port', type=int, default=8002, help='Panel 服务端口')
	parser.add_argument('--doc_score_upper', type=float, default=0.35, help='检索分数上限')
	parser.add_argument('--doc_num', type=int, default=5, help='检索文档数')
	args = parser.parse_args()

	import CBextension

	# 加载文档并构建向量库
	print("加载文档...")
	docs = load_documents(args.data_dir)
	print(f"共 {len(docs)} 个 chunk")
	db = build_vectorstore(docs, args.db_dir)

	# 初始化 LLM
	llm = ChatGoogleGenerativeAI(
		model='gemini-1.5-pro',
		temperature=0,
		convert_system_message_to_human=True,
	)

	# 提示词模板
	template_calls = (
		"如果答最后一个Question可以用Context里的function calls回答，"
		"就按照Example的格式回复，如果不能就回复空字符串。\n"
		"Example:\n[\"CBextension.sales_prediction(3)\", \"CBextension.sales_prediction(6)\"]\n\n"
		f"Context:\n{CBextension.TOOL_DESCRIPTIONS}\n\n"
		"{QA_history}\n\nQuestion: {question}\nAnswer:\n"
	)

	template = (
		"你是美国达拉斯一家鲜芋仙店铺的店长助理，"
		"请结合以下Context和市场公关知识回答最后一个Question。"
		"'|分析结果'里的内容都已经过验证，不用怀疑。\n\n"
		"Context:\n{context}\n\n{QA_history}\n\nQuestion: {question}\nAnswer:\n"
	)

	# 构建 RAG 引擎
	qa = RagExt(
		llm=llm,
		db=db,
		prompt_template=template,
		tool_registry=CBextension.TOOL_REGISTRY,
		prompt_template_call=template_calls,
		doc_num=args.doc_num,
		doc_score_upper=args.doc_score_upper,
		history_budget=2000,
	)

	# 启动 Panel UI
	import panel as pn

	pn.config.loading_spinner = 'petal'
	pn.config.loading_color = 'black'
	pn.extension()

	class ConvRag:
		panels: list = []

		def convchain(self, query: str):
			self.panels = self.panels[-15:]
			if not query:
				return pn.WidgetBox(
					pn.Row('Query:', pn.pane.Markdown("", width=1500,
					styles={'background-color': '#e6f2ff'})),
					scroll=True,
				)
			result = qa.query(query)
			db_response = [
				f"{d['doc'].metadata.get('source','')} {d['score']:.3f}"
				for d in result['source_documents']
			]
			answer = result['answer']
			if db_response:
				answer += '\nReferences:\n' + '\n'.join(db_response)
			self.panels.extend([
				pn.Row('Prompt:', pn.pane.Markdown(result['generated_question'],
					width=1300, styles={'background-color': '#cce6ff'})),
				pn.Row('Answer:', pn.pane.Markdown(answer,
					width=1300, styles={'background-color': '#cce6ff'})),
				pn.Row('Query:', pn.pane.Markdown(query)),
			])
			inp.value = ''
			return pn.WidgetBox(*reversed(self.panels), scroll=True)

	cb = ConvRag()
	inp = pn.widgets.TextInput(placeholder='输入问题…', width=500)
	conversation = pn.bind(cb.convchain, inp)
	tab1 = pn.Column(
		pn.Row(inp),
		pn.layout.Divider(),
		pn.panel(conversation, loading_indicator=True),
		pn.layout.Divider(),
	)
	dashboard = pn.Column(
		pn.Row(pn.pane.Markdown('# 鲜芋仙 AI 助理')),
		pn.Tabs(('对话', tab1)),
	)

	print(f"启动服务 http://localhost:{args.port}")
	pn.serve(dashboard, title="AI Assistant", port=args.port, show=False)
