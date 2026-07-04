import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

print("正在呼叫 DeepSeek 专家总部...\n")

try:
    response = client.chat.completions.create(
        model="deepseek-chat", # 如果想测试推理模型，可以换成 "deepseek-reasoner"
        messages=[
            {"role": "system", "content": "你是一个顶尖的计算生物学家。"},
            {"role": "user", "content": "请用一句话解释什么是抗菌肽（AMP）。"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    print("✅ 接收成功！模型回复：")
    print("-" * 30)
    print(response.choices[0].message.content)
    print("-" * 30)
except Exception as e:
    print(f"❌ 呼叫失败，错误信息：\n{e}")