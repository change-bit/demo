from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import torch

torch.cuda.empty_cache()  # 释放显存碎片，必备

#向量库配置
EMBEDDING_DIM = 1536
COLLECTION_NAME = "full_demo"
PATH = "./qdrant_db"
client = QdrantClient(path=PATH, allow_concurrent_reads=True)

#  依赖导入
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core.postprocessor import SimilarityPostprocessor

# 加载本地模型+绑定全局Settings
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings('ignore')  # 关闭无关日志

MODEL_PATH = r"F:\llmv1.0\inori"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# 修复torch_dtype警告：torch_dtype → dtype
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="cuda:0",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# 绑定token配置，稳定生成
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.eval()

# 定义CustomLLM类，绑定全局
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen
from typing import Any, List
import types


class CustomDeepSeekLLM(CustomLLM):
    @property
    def metadata(self):
        meta = types.SimpleNamespace()
        meta.model_name = "inori-DeepSeek1.5B"
        meta.context_window = 4096
        meta.num_output = 768
        meta.is_chat_model = True
        return meta

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        prompt = f"<｜begin▁of▁sentence｜>{prompt}\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=768,
                temperature=0.25,
                top_p=0.85,
                do_sample=True,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        answer = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return CompletionResponse(text=answer)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        yield self.complete(prompt, **kwargs)


Settings.llm = CustomDeepSeekLLM()

# 配置Embedding+文档分片
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v1",
    api_key="sk-2231d947be55426fb8ebb3057c2b7072"
)
# 中文文档最优分片配置，无冗余
Settings.transformations = [SentenceSplitter(chunk_size=300, chunk_overlap=50)]

#加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 初始化向量库
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
client.create_collection(COLLECTION_NAME, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)


# 相似度阈值0.4（过高会检索不到）
sp = SimilarityPostprocessor(similarity_cutoff=0.4)

retriever = index.as_retriever(similarity_top_k=5)


# Prompt
def build_prompt(question, retrieved_text):
    if retrieved_text:
        # 如果有知识库，强制参考知识库回答
        prompt = f"""<｜begin▁of▁sentence｜>
严格按照【已知信息】回答问题，已知信息是唯一答案来源，禁止编造内容、禁止说无关的话。
精准提炼已知信息中的答案即可，不要添加额外内容。

【已知信息】：
{retrieved_text}

【用户问题】：{question}
【回答】："""
    else:
        # 如果没有知识库，允许模型用自己的知识自由回答（恢复聊天功能）
        prompt = f"""<｜begin▁of▁sentence｜>
【用户问题】：{question}
【回答】："""
    return prompt


# 生成回答
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=768,
            temperature=0.25,
            top_p=0.85,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    answer = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return answer


# ===================== 8. 交互循环+调试日志+退出提示====================
print("=====================inori-DeepSeek1.5B 纯本地RAG =====================")
print("=====================输入 exit/quit/退出 结束对话   =====================\n")

while True:
    question = input("User: ")
    if question.strip() in ["exit", "quit", "退出"]:
        print("\n🎉 退出成功，楪祈永远陪伴你 ❤️ 🎉")
        break

    # 检索流程：极简无冗余，必出结果
    retrieved_nodes = retriever.retrieve(question)
    filtered_nodes = sp.postprocess_nodes(retrieved_nodes)
    retrieved_text = "\n".join([node.text for node in filtered_nodes])

    # 调试日志：查看检索结果（可随时删掉，不影响功能）
    print("\n" + "=" * 60)
    print("📄 检索到的知识库内容")
    print(retrieved_text if retrieved_text else "❌ 暂无匹配内容")
    print("=" * 60 + "\n")

    # 生成回答
    prompt = build_prompt(question, retrieved_text)
    answer = generate_answer(prompt)

    print(f"AI: {answer}\n")