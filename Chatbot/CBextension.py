"""鲜芋仙门店 Chatbot 工具扩展模块。

新增工具在此文件定义函数，并注册到 TOOL_REGISTRY 即可被 RagExt 安全调用，
无需修改核心 RAG 逻辑。
"""


def sales_prediction(n: int) -> dict:
	"""预测 n 个月后的月营收（美元）。

	参数:
		n: 整数，距今月份数
	返回:
		{'value': str, 'description': str}
	"""
	return {
		'value': str(500 + n),
		'description': f"{n}个月后的月营收（美元）"
	}


# 工具注册表：key 为 LLM 调用时使用的函数名字符串（含模块前缀），
# value 为实际可调用对象。仅此表中的函数才会被执行。
TOOL_REGISTRY: dict = {
	"CBextension.sales_prediction": sales_prediction,
}

# 供 prompt 使用的工具描述字符串，与 TOOL_REGISTRY 保持同步
TOOL_DESCRIPTIONS = """\
{call: CBextension.sales_prediction(n),
description: 预测 n 个月后的月营收（美元）,
return: {value: str, description: str},
parameters: {n: 整数，月份数}}"""
